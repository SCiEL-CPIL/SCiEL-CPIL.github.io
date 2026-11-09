import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class Optimizer:
    """Multi-objective optimizer using Sobol initialization and local L-BFGS-B refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        self.eval_count = 0
        self.points = []
        self.objectives = []
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Reference point for hypervolume calculation (used for selection)
        # The task says reference point is [3000.0, 0.06] for scoring.
        # We should use this for our internal HV calculation to align with scoring.
        ref_point = np.array([3000.0, 0.06])

        def count_eval(x):
            self.eval_count += 1
            if self.eval_count > self.budget:
                raise Exception("Budget exceeded")
            f = problem(x)
            self.points.append(np.array(x))
            self.objectives.append(np.array(f))
            return f

        # Phase 1: Global Exploration with Sobol
        n_init = min(40, self.budget)
        if n_init > 0:
            sobol = Sobol(d=self.dim, scramble=True)
            X_sobol = sobol.random(n_init)
            X_init = lo + (hi - lo) * X_sobol
            for x in X_init:
                try:
                    count_eval(x)
                except Exception:
                    break

        # Phase 2: Local Refinement
        # We will iteratively pick the non-dominated point with the highest HV contribution
        # and refine it using L-BFGS-B on a weighted sum objective.
        
        def calc_hv(F, ref):
            """Calculate hypervolume for 2D objectives."""
            # Filter points strictly better than reference point
            mask = np.all(F < ref, axis=1)
            F_valid = F[mask]
            if len(F_valid) == 0:
                return 0.0
            
            # Sort by f1 ascending
            F_valid = F_valid[np.argsort(F_valid[:, 0])]
            
            hv = 0.0
            n = len(F_valid)
            for i in range(n):
                # Width in f1 direction
                if i < n - 1:
                    f1_next = F_valid[i+1, 0]
                else:
                    f1_next = ref[0]
                
                # Height in f2 direction
                f2_val = F_valid[i, 1]
                
                # Only count if f2 is better than ref
                if f2_val < ref[1]:
                    hv += (f1_next - F_valid[i, 0]) * (ref[1] - f2_val)
            
            return hv

        def get_non_dominated(F):
            F = np.array(F)
            n = len(F)
            is_nd = np.ones(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_nd[i] = False
                        break
            return is_nd

        while self.eval_count < self.budget and (time.time() - self.start_time) < 290.0:
            if len(self.objectives) == 0:
                break
            
            F = np.array(self.objectives)
            nd_mask = get_non_dominated(F)
            nd_indices = np.where(nd_mask)[0]
            
            if len(nd_indices) == 0:
                break
            
            # Calculate HV contribution for each non-dominated point
            # HV of full set
            F_nd = F[nd_indices]
            hv_full = calc_hv(F_nd, ref_point)
            
            if hv_full <= 0:
                # If no valid points, just pick the best by sum of objectives
                scores = np.sum(F[nd_indices], axis=1)
                best_idx_rel = np.argmin(scores)
                best_idx = nd_indices[best_idx_rel]
            else:
                contributions = []
                for i in range(len(nd_indices)):
                    # Remove point i
                    mask_remove = np.ones(len(nd_indices), dtype=bool)
                    mask_remove[i] = False
                    F_reduced = F_nd[mask_remove]
                    hv_reduced = calc_hv(F_reduced, ref_point)
                    contributions.append(hv_full - hv_reduced)
                
                contributions = np.array(contributions)
                best_idx_rel = np.argmax(contributions)
                best_idx = nd_indices[best_idx_rel]
            
            x_start = self.points[best_idx]
            f_start = self.objectives[best_idx]
            
            # Determine weights for Weighted Sum
            # Use inverse of current objectives to balance
            # Add small epsilon to avoid division by zero
            w = 1.0 / (f_start + 1e-6)
            w = w / np.sum(w)
            
            # Local optimization
            maxfev_local = min(15, self.budget - self.eval_count)
            if maxfev_local <= 0:
                break
            
            def local_obj(x):
                f = count_eval(x)
                return np.dot(w, f)
            
            try:
                res = minimize(local_obj, x_start, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 50, 'maxfev': maxfev_local})
                # The best point found is res.x
                # We already counted the evaluations inside local_obj
            except Exception:
                # If minimize fails or exceeds budget, just continue
                pass
            
            if self.eval_count >= self.budget:
                break

        return self.points
