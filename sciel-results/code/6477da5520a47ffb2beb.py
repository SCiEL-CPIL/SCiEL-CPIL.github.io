"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using Multi-Objective Differential Evolution (MOEA/D-DE)."""

    def __init__(self, dim, n_obj, budget):
        self.dim = dim
        self.n_obj = n_obj
        self.budget = budget

    def __call__(self, problem, bounds):
        start_time = time.time()
        max_time = 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Budget management
        # We need to be very careful with evaluations.
        # Let's reserve 10 evaluations for initialization if possible, but with 200 total,
        # we might just initialize with random points and evolve.
        
        # MOEA/D parameters
        # Population size N
        # With 200 budget, N=20 allows for 10 generations (20 evals/gen).
        # Or N=10 allows for 20 generations.
        # Smaller N is better for diversity in small budgets? 
        # Let's try N=20.
        
        N = 20
        # Weights for MOEA/D: uniform distribution on the simplex
        # For 3 objectives, we can generate weights using a grid or random.
        # Random Dirichlet is good for coverage.
        rng = np.random.default_rng(42)
        weights = rng.dirichlet(np.ones(self.n_obj), size=N)
        
        # Initialize population
        # Each individual is associated with a weight vector.
        # Initialize by minimizing the weighted sum for each weight? 
        # That would cost N * (cost of local search). 
        # Better: Initialize randomly, then evolve.
        
        X = np.zeros((N, self.dim))
        F = np.zeros((N, self.n_obj))
        
        # Random initialization
        for i in range(N):
            X[i] = lo + (hi - lo) * rng.random(self.dim)
            F[i] = problem(X[i])
            
        # Record all evaluations
        all_X = [X.copy()]
        all_F = [F.copy()]
        eval_count = N
        
        # MOEA/D Evolution
        # Standard DE: x_trial = x_r1 + F_param * (x_r2 - x_r3)
        # F_param is the differential weight, usually 0.5-0.9
        # CR is crossover rate, usually 0.5-1.0
        
        F_param = 0.8
        CR = 0.9
        
        # Neighborhood size
        # For N=20, a neighborhood of 5 is reasonable.
        # Define neighbors for each weight vector based on distance in weight space.
        # Simpler: Just pick random individuals for mutation, but selection is based on weight vector.
        # To keep it simple and fast, we'll use a global DE-like mutation but local selection.
        
        # Precompute neighbors? 
        # With N=20, computing distance 20*20 is trivial.
        # dist[i,j] = ||w_i - w_j||
        w_dist = np.linalg.norm(weights[:, np.newaxis, :] - weights[np.newaxis, :, :], axis=2)
        # Sort neighbors for each i
        # neighbors[i] is a list of indices of closest weights
        K = 5  # Neighborhood size
        neighbors = []
        for i in range(N):
            # Get indices of closest K weights (excluding self)
            # argsort gives indices that would sort the row
            sorted_idx = np.argsort(w_dist[i])
            # First one is self (distance 0), so take next K
            neighbors.append(sorted_idx[1:K+1])
            
        while eval_count < self.budget:
            if time.time() - start_time > max_time:
                break
                
            # Generate offspring for each individual in population
            # We can generate all N offspring in one go, evaluating N points.
            # This uses N evaluations per generation.
            
            X_off = np.zeros_like(X)
            F_off = np.zeros_like(F)
            
            # Check if we have enough budget for a full generation
            if eval_count + N > self.budget:
                # If not, we might do partial generation or stop.
                # Let's just stop to be safe and clean.
                break
            
            for i in range(N):
                # Mutation: x_r1 + F * (x_r2 - x_r3)
                # Pick r1, r2, r3 from neighbors of i
                # Ensure they are distinct and not i
                # Simple approach: pick random indices from neighbors list
                # neighbors[i] has K indices.
                # We need 3 distinct indices.
                # If K < 3, we might have issues, but K=5.
                
                # Shuffle neighbors to get random distinct ones
                n_idx = neighbors[i]
                # Pick 3 distinct from n_idx
                # np.random.choice without replacement
                r1, r2, r3 = rng.choice(n_idx, size=3, replace=False)
                
                # DE mutation
                # x_trial = X[r1] + F_param * (X[r2] - X[r3])
                x_trial = X[r1] + F_param * (X[r2] - X[r3])
                
                # Crossover: binary crossover
                # For each dimension j:
                # if rng.random() < CR or j == j_rand: x_trial[j] = X_off[i][j] (from mutant)
                # else: x_trial[j] = X[i][j] (from parent)
                
                j_rand = rng.integers(0, self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        X_off[i][j] = x_trial[j]
                    else:
                        X_off[i][j] = X[i][j]
                        
                # Clip to bounds
                X_off[i] = np.clip(X_off[i], lo, hi)
                
                # Evaluate
                F_off[i] = problem(X_off[i])
                eval_count += 1
                
                # Record evaluation
                all_X.append(X_off[i].reshape(1, -1))
                all_F.append(F_off[i].reshape(1, -1))
                
            # Selection: For each i, compare X_off[i] with X[i] using weight vector w_i
            # If w_i . F_off[i] < w_i . F[i], replace X[i] with X_off[i]
            # This is the MOEA/D selection mechanism.
            
            for i in range(N):
                score_parent = np.dot(weights[i], F[i])
                score_off = np.dot(weights[i], F_off[i])
                
                if score_off < score_parent:
                    X[i] = X_off[i]
                    F[i] = F_off[i]
                    
        # Return all recorded points
        # The task says "Every returned objective vector is recorded".
        # The evaluator calls problem(x) and records the result.
        # My code above appends to all_X and all_F manually, but the evaluator
        # actually records based on the calls to `problem`.
        # The return value of __call__ is ignored.
        # So I just need to make sure I call `problem` correctly.
        
        # The contract says: "Every returned objective vector is recorded... return value of __call__ is ignored."
        # So I don't need to return anything specific, but I must ensure I don't exceed budget.
        
        return None

# EVOLVE-BLOCK-END
