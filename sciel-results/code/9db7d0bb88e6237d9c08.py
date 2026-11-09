import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        n_evals = 0
        X = []
        F = []
        
        def evaluate(x):
            nonlocal n_evals
            fx = problem(x)
            n_evals += 1
            return fx

        # 1. Latin Hypercube Sampling for initial coverage
        # Use 100 points for global exploration
        n_lhs = 100
        if n_lhs > self.budget:
            n_lhs = self.budget
            
        # Generate LHS points
        # For each dim, create n_lhs unique intervals
        lhs_points = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            # Create bins
            edges = np.linspace(0, 1, n_lhs + 1)
            # Random point in each bin
            points_in_bins = np.random.uniform(edges[:-1], edges[1:])
            # Shuffle to decorrelate dimensions
            np.random.shuffle(points_in_bins)
            lhs_points[:, d] = points_in_bins
        
        # Scale to bounds
        lhs_points = lo + lhs_points * span
        
        # Evaluate LHS points
        for i in range(n_lhs):
            if n_evals >= self.budget or time.time() > deadline:
                break
            x = lhs_points[i]
            fx = evaluate(x)
            X.append(x)
            F.append(fx)
            
        if n_evals >= self.budget or time.time() > deadline:
            return np.array(X)
            
        # 2. Local Search around best points
        # Convert to arrays for easy manipulation
        X = np.array(X)
        F = np.array(F)
        
        # Find non-dominated points
        # A point i is dominated if there exists j such that F[j] <= F[i] and F[j] != F[i]
        def is_dominated(F, i):
            for j in range(len(F)):
                if i == j: continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    return True
            return False
            
        nondom_indices = [i for i in range(len(F)) if not is_dominated(F, i)]
        
        if not nondom_indices:
            # Fallback: pick random
            nondom_indices = list(range(len(F)))
            
        # Select top K non-dominated points for local search
        # Sort by sum of objectives (simple heuristic)
        nondom_indices.sort(key=lambda i: np.sum(F[i]))
        k_parents = min(10, len(nondom_indices))
        parents = [X[i] for i in nondom_indices[:k_parents]]
        
        # Use remaining budget for local search
        # Distribute remaining evaluations among parents
        remaining_budget = self.budget - n_evals
        if remaining_budget > 0 and parents:
            # Use scalarized optimization to polish each parent
            # Weight vectors for 3 objectives
            # Uniform distribution on simplex
            n_weights = len(parents)
            weights = []
            for i in range(n_weights):
                w = np.random.dirichlet(np.ones(3))
                weights.append(w)
            
            # Assign each parent to a unique weight
            parent_weight_pairs = []
            for i, parent in enumerate(parents):
                w = weights[i % len(weights)]
                parent_weight_pairs.append((parent, w))
            
            # Custom local search
            for parent, w in parent_weight_pairs:
                if n_evals >= self.budget or time.time() > deadline:
                    break
                    
                # Start from parent
                x_current = parent.copy()
                
                # We need the current objective vector to start
                # Since we already evaluated parent in LHS, we can find it in F
                # But it's easier to just evaluate it again if we don't have a handle.
                # Actually, we have X and F. Let's find the index.
                # But X is a list of arrays.
                # Let's just evaluate it once to get f_current vector.
                # This costs 1 eval.
                if n_evals >= self.budget:
                    break
                f_current_vec = evaluate(x_current)
                f_current_scalar = np.dot(w, f_current_vec)
                
                # Simple local search with adaptive step
                step_size = 0.1 * span
                for step_iter in range(5):  # 5 steps per parent
                    if n_evals >= self.budget or time.time() > deadline:
                        break
                    
                    # Generate a random direction
                    direction = np.random.standard_normal(self.dim)
                    direction /= np.linalg.norm(direction)
                    
                    # Try step in positive and negative direction
                    best_x = x_current
                    best_f_vec = f_current_vec
                    best_f_scalar = f_current_scalar
                    
                    for sign in [1, -1]:
                        if n_evals >= self.budget or time.time() > deadline:
                            break
                        x_try = x_current + sign * step_size * direction
                        x_try = np.clip(x_try, lo, hi)
                        
                        if np.allclose(x_try, x_current):
                            continue
                        
                        f_try_vec = evaluate(x_try)
                        f_try_scalar = np.dot(w, f_try_vec)
                        
                        if f_try_scalar < best_f_scalar:
                            best_f_scalar = f_try_scalar
                            best_f_vec = f_try_vec
                            best_x = x_try
                    
                    if np.allclose(best_x, x_current):
                        # No improvement, reduce step size
                        step_size *= 0.5
                        continue
                    else:
                        # Improvement found
                        x_current = best_x
                        f_current_vec = best_f_vec
                        f_current_scalar = best_f_scalar
                
                # Record the best point found
                X = np.vstack([X, x_current])
                F = np.vstack([F, f_current_vec])
                
        return np.array(X)
