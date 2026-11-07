"""Multi-objective optimizer for DTLZ1."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """MO-CMA-ES style multi-objective optimizer for DTLZ1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size for CMA-ES
        # Standard is 4 + floor(3 ln n), but we are constrained by total budget.
        # With budget=200, we can afford a population of ~20-30.
        pop_size = 20
        
        # Initialize mean at center
        mean = (lo + hi) / 2.0
        # Initialize sigma (step size)
        sigma = 0.1
        
        # Initialize covariance matrix
        C = np.eye(dim)
        # Inverse square root of C (initially identity)
        C_inv_sqrt = np.eye(dim)
        
        # Eigen decomposition for step size adaptation
        # We will update C and sigma periodically.
        # For simplicity and speed, we use a simplified update rule or 
        # re-decompose every generation.
        
        def fast_non_dominated_sort(F):
            """Return list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in _ in range(n)] # typo fix below
            S = [[] for _ in range(n)]
            n_d = np.zeros(n, dtype=int)
            fronts = [[]]

            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_d[p] += 1
                if n_d[p] == 0:
                    fronts[0].append(p)

            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_d[q] -= 1
                        if n_d[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)

            if not fronts[-1]:
                fronts.pop()
            return fronts

        used = 0
        # We will generate pop_size offspring per generation
        # We need to keep a "archive" of non-dominated points to guide the mean?
        # Actually, standard MO-CMA-ES updates the distribution based on the 
        # non-dominated offspring.
        
        # To keep it simple and robust within the budget:
        # 1. Sample pop_size points from N(mean, sigma^2 * C)
        # 2. Evaluate them
        # 3. Find non-dominated set
        # 4. Update mean to the centroid of non-dominated set (or weighted)
        # 5. Update sigma and C based on the spread of the non-dominated set
        
        # We must be careful not to exceed budget.
        # Generations = budget // pop_size = 200 // 20 = 10 generations.
        
        # Initial evaluation of the mean? No, we just start sampling.
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # Sample offspring
            # z ~ N(0, I)
            Z = np.random.randn(pop_size, dim)
            # Transform to x = mean + sigma * C_inv_sqrt @ z.T
            # Note: C_inv_sqrt is updated such that C_inv_sqrt @ C_inv_sqrt.T = C_inv
            # So x = mean + sigma * (C_inv_sqrt @ Z.T).T = mean + sigma * Z @ C_inv_sqrt.T
            # Wait, if C_inv_sqrt is such that C_inv_sqrt @ C_inv_sqrt.T = C^{-1},
            # then x = mean + sigma * (C_inv_sqrt @ Z.T).T
            # Let's use the standard formulation:
            # x_i = mean + sigma * B @ z_i, where B @ B.T = C
            # So we need B (matrix square root of C).
            
            # Let's maintain B directly.
            # Initialize B = I
            if not hasattr(self, '_B') or self._B is None:
                B = np.eye(dim)
            else:
                B = self._B
                
            # Sample
            X = mean + sigma * (Z @ B.T)
            
            # Clip to bounds
            X = np.clip(X, lo, hi)
            
            # Evaluate
            F = np.array([problem(x) for x in X])
            used += pop_size
            
            # Find non-dominated points
            fronts = fast_non_dominated_sort(F)
            if not fronts or not fronts[0]:
                # Fallback: take best 2
                break
            
            nd_indices = fronts[0]
            
            # Update mean: Centroid of non-dominated points
            if len(nd_indices) > 0:
                new_mean = np.mean(X[nd_indices], axis=0)
            else:
                new_mean = mean
            
            # Update sigma and B (Covariance)
            # Simple strategy: 
            # Sigma adaptation: Use the distance from mean to non-dominated points
            # Or use the standard CMA-ES step size control (1/5 rule) adapted for MO.
            # For simplicity and stability:
            # Estimate covariance of the non-dominated points.
            if len(nd_indices) >= 2:
                X_nd = X[nd_indices]
                # Center X_nd
                X_nd_centered = X_nd - np.mean(X_nd, axis=0)
                # Covariance
                C_new = np.cov(X_nd_centered.T)
                # Ensure positive definite
                # Add small regularization
                C_new += 1e-6 * np.eye(dim)
                
                # Matrix square root of C_new
                # Using SVD for stability
                U, s, Vt = np.linalg.svd(C_new)
                B_new = U @ np.diag(np.sqrt(s)) @ Vt
            else:
                B_new = B
                # If only 1 point, keep B, maybe shrink sigma?
                sigma *= 0.9

            # Update sigma
            # Standard CMA-ES uses a history of step sizes.
            # Here, we can scale sigma based on the spread of the front.
            # If the front is wide, increase sigma? No, we want to converge.
            # Let's use a simple rule: 
            # sigma_new = sigma * (1 + 0.01 * (len(nd_indices)/pop_size - 0.5))
            # Or just keep sigma constant and let B handle the shape?
            # In MO-CMA-ES, sigma is often adapted using the same mechanism as single-objective,
            # but the target is the non-dominated set.
            
            # Let's try to keep sigma roughly constant or slightly decreasing.
            # A better heuristic for MO: 
            # The "effective" step size is determined by the spread of the non-dominated set.
            # We want the next generation to explore around this set.
            
            # Let's update mean and B.
            mean = new_mean
            B = B_new
            
            # Simple sigma adaptation: 
            # If we found many non-dominated points, we are doing well.
            # Let's keep sigma = 0.1 fixed for now to ensure stability, 
            # as B adapts the shape.
            # However, if the front is very small, we might want to shrink.
            # Let's add a simple shrinkage if the population is not diverse enough.
            pass # sigma stays constant for simplicity in this iteration.

            # Store B for next iteration
            self._B = B

        return X
# EVOLVE-BLOCK-END
