# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        budget = self.budget
        
        # 1. Generate Weight Vectors for MOEA/D
        # We need a population size that fits within the budget.
        # Let's use a population size P. We will do generations.
        # Budget = P * Generations.
        # Let P = 40. Generations = (200 - P) / P = 4.5 -> 4 generations?
        # Or just run a loop until budget is exhausted.
        
        # Let's pick a population size. 
        # WFG4 is 12D. 
        # Standard MOEA/D uses P ~ 100-200 for 2-3 objectives.
        # With budget 200, we can have P=50 and 3 generations (50 init + 150 evolve).
        # Or P=100 and 1 generation (100 init + 100 evolve).
        # Let's try P=50.
        
        P = 50
        
        # Generate weight vectors using a simplex lattice or random uniform distribution.
        # For 3 objectives, a random uniform distribution on the simplex is good enough for initial coverage.
        # Or use a systematic grid.
        # Let's use a simple method: random points on the simplex.
        W = np.random.dirichlet(np.ones(m), size=P)
        
        # Initialize population
        # Use Latin Hypercube Sampling for initial diversity
        X = lo + (hi - lo) * np.random.random((P, n))
        F = np.array([problem(x) for x in X])
        used = P
        
        # Ideal point approximation (minimum for each objective)
        ideal = np.min(F, axis=0)
        
        # Track best scalarized value for each weight vector
        # Actually, MOEA/D usually keeps one solution per weight vector.
        # We will store X and F aligned with W.
        
        # Helper: Tchebycheff scalarization
        def tcheb(f, w, ideal):
            # f: (m,)
            # w: (m,)
            # ideal: (m,)
            # Normalize by range if possible, but we don't know range.
            # Use simple Tcheb: max(w_i * (f_i - ideal_i))
            # To handle scales, we might need normalization, but without knowing max,
            # we can use a dynamic normalization or just raw values if scales are similar.
            # WFG4 objectives are typically in [0, 10] or similar small ranges.
            # Let's assume raw Tcheb works, or normalize by current max if > 0.
            # Better: use a large constant or current max for normalization to be safe.
            # Let's use current max as a proxy for range.
            max_f = np.max(F, axis=0)
            range_f = max_f - ideal
            range_f[range_f == 0] = 1.0 # Avoid div by zero
            norm_f = (f - ideal) / range_f
            return np.max(w * norm_f)
        
        # Pre-compute neighborhoods
        # For simplicity, use a fixed neighborhood size K
        K = 20
        # Calculate distance between weight vectors
        # W is P x m
        # dist[i,j] = ||W[i] - W[j]||
        # This is P x P. P=50, so 2500 distances. Fast.
        W_flat = W.reshape(P, -1)
        dist_matrix = np.sum((W_flat[:, np.newaxis, :] - W_flat[np.newaxis, :, :])**2, axis=2)
        
        # Get K nearest neighbors for each weight vector
        neighborhoods = []
        for i in range(P):
            # Sort indices by distance
            idx = np.argsort(dist_matrix[i])
            # Exclude self
            neighbors = idx[1:K+1]
            neighborhoods.append(neighbors)
        
        # Main Evolutionary Loop
        # We have budget - P evaluations left.
        # Each generation produces P new solutions.
        # So we can do roughly (budget - P) / P generations.
        
        while used < budget:
            if time.time() > deadline - 10:
                break
                
            # For each weight vector i, generate a new solution
            # Select two parents from neighborhood of i
            for i in range(P):
                if used >= budget:
                    break
                if time.time() > deadline - 5:
                    break
                    
                # Select parents from neighborhood
                neigh = neighborhoods[i]
                # Randomly pick 2 distinct parents from neighborhood
                # Ensure we have at least 2
                if len(neigh) < 2:
                    # Fallback to random population
                    p1_idx, p2_idx = np.random.choice(P, 2, replace=False)
                else:
                    p1_idx, p2_idx = np.random.choice(neigh, 2, replace=False)
                    
                # Crossover: SBX or BLX-alpha
                # Let's use BLX-alpha for simplicity and robustness
                alpha = 0.5
                x1 = X[p1_idx]
                x2 = X[p2_idx]
                
                # BLX-Alpha
                # x_child = x1 + alpha * (x2 - x1) * (1 + U(-1, 1)) ? 
                # Standard BLX: 
                # a = min(x1, x2) - alpha * |x1 - x2|
                # b = max(x1, x2) + alpha * |x1 - x2|
                # x_child = U(a, b)
                
                a = np.minimum(x1, x2) - alpha * np.abs(x1 - x2)
                b = np.maximum(x1, x2) + alpha * np.abs(x1 - x2)
                
                x_child = a + (b - a) * np.random.random(n)
                x_child = np.clip(x_child, lo, hi)
                
                # Mutation: Gaussian
                # Step size adaptive? Let's use a fixed relative step size.
                # sigma = 0.1 * (hi - lo)
                sigma = 0.1 * (hi - lo)
                x_child = x_child + np.random.normal(0, 1, n) * sigma
                x_child = np.clip(x_child, lo, hi)
                
                # Evaluate
                f_child = problem(x_child)
                used += 1
                
                # Update Ideal Point
                ideal = np.minimum(ideal, f_child)
                
                # Update Logic
                # In MOEA/D, a new solution replaces the old one associated with weight W[i]
                # if it has a better scalarized value with respect to W[i].
                # It can also replace neighbors if it dominates them.
                
                # Calculate scalarized value for W[i]
                val_new = tcheb(f_child, W[i], ideal)
                
                # Get current solution for W[i]
                val_old = tcheb(F[i], W[i], ideal)
                
                # If new is better for this weight, replace
                if val_new < val_old:
                    X[i] = x_child
                    F[i] = f_child
                else:
                    # Check if it dominates any neighbor
                    # This is optional but improves convergence.
                    # To save time, we might skip full dominance check for all neighbors
                    # and just stick to the weight-based update.
                    # However, updating neighbors is key for MOEA/D performance.
                    # Let's do a limited check: check if it dominates the neighbor's solution.
                    for j in neighborhoods[i]:
                        if j == i:
                            continue
                        # Check if f_child dominates F[j]
                        # f_child <= F[j] and f_child != F[j]
                        if np.all(f_child <= F[j]) and np.any(f_child < F[j]):
                            # Replace neighbor j
                            X[j] = x_child
                            F[j] = f_child
                            # Note: This might break the 1-to-1 mapping of W and X if we are strict,
                            # but in MOEA/D, the solution is associated with the weight.
                            # If we replace F[j], it is now the solution for weight W[j].
                            # This is correct behavior.
        
        # Return all evaluated points? 
        # The task says "Every returned objective vector is recorded".
        # The __call__ return value is ignored.
        # But wait, "Every returned objective vector is recorded" implies the evaluator
        # records the output of `problem(x)`. 
        # "The return value of __call__ is ignored."
        # So we just need to ensure we call `problem` correctly.
        # We don't need to return anything specific, but the current code returns X.
        # Let's return F or X, it doesn't matter.
        
        return F
# EVOLVE-BLOCK-END
