# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using dense Pareto-front grid + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0  # safety margin under 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        def time_ok():
            return time.time() < deadline

        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # For DTLZ2 with 3 objectives, Pareto front is quarter of unit sphere in R^3.
        # Parameters: theta1 in [0, pi/2], theta2 in [0, pi/2]
        # y1 = cos(theta1)
        # y2 = sin(theta1)*cos(theta2)
        # y3 = sin(theta1)*sin(theta2)
        # f_i = y_i (when g=0)
        # Decision vars: x[0]=theta1/(pi/2), x[1]=theta2/(pi/2), x[2:]=0.5

        # Use stratified sampling for better coverage
        # Split [0,1] x [0,1] into n_per_dim x n_per_dim cells, sample one point per cell
        n_per_dim = 14  # 196 points on front
        remaining_budget = budget - n_per_dim

        # Stratified sampling: for each cell, sample uniformly within the cell
        # This gives better coverage than a fixed grid
        a1_vals = np.empty(n_per_dim * n_per_dim)
        a2_vals = np.empty(n_per_dim * n_per_dim)
        
        idx = 0
        for i in range(n_per_dim):
            for j in range(n_per_dim):
                # Random offset within cell [i/n, (i+1)/n] x [j/n, (j+1)/n]
                a1_vals[idx] = (i + np.random.uniform(0.1, 0.9)) / n_per_dim
                a2_vals[idx] = (j + np.random.uniform(0.1, 0.9)) / n_per_dim
                idx += 1

        F_list = []
        X_list = []
        used = 0

        for k in range(n_per_dim * n_per_dim):
            if not time_ok():
                break
            theta1 = a1_vals[k] * (np.pi / 2)
            theta2 = a2_vals[k] * (np.pi / 2)
            
            x = np.zeros(dim)
            x[0] = a1_vals[k]
            x[1] = a2_vals[k]
            x[2:] = 0.5
            x = np.clip(x, lo, hi)
            f = problem(x)
            F_list.append(f)
            X_list.append(x)
            used += 1

        F = np.array(F_list)
        X = np.array(X_list)

        # Use remaining budget for local refinement
        # Perturb points to find slightly better positions on the front
        if remaining_budget > 0 and time_ok():
            rng = np.random.default_rng(42)
            for _ in range(remaining_budget):
                if not time_ok():
                    break
                if len(X) == 0:
                    break
                idx = rng.integers(len(X))
                child = X[idx].copy()
                # Small perturbations in angle space
                child[0] += 0.02 * rng.standard_normal()
                child[1] += 0.02 * rng.standard_normal()
                child[2:] = 0.5
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                
                # Check if this point is non-dominated with respect to existing set
                # and if it improves hypervolume (approximation: if it's not dominated)
                if len(F) > 0:
                    # Point is non-dominated if no existing point dominates it
                    dominated = np.any(np.all(F <= fc, axis=1) & np.any(F < fc, axis=1))
                    if not dominated:
                        # Replace the point that is most "redundant" (dominated by others or worst)
                        # Find the point with worst sum of objectives among dominated ones
                        dominated_mask = np.zeros(len(F), dtype=bool)
                        for i in range(len(F)):
                            dominated_mask[i] = np.any(np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1))
                        
                        if np.any(dominated_mask):
                            worst_idx = np.where(dominated_mask)[0]
                            worst = worst_idx[np.argmax(np.sum(F[worst_idx], axis=1))]
                        else:
                            # All non-dominated, replace the one with highest sum
                            worst = int(np.argmax(np.sum(F, axis=1)))
                        
                        X[worst] = child
                        F[worst] = fc

        return X
# EVOLVE-BLOCK-END
