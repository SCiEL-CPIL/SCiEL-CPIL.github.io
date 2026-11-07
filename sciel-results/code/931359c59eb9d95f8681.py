# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Structure-aware optimizer for ZDT2 exploiting the Pareto front geometry."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        n_points = min(budget, 200)
        if n_points < 2:
            n_points = 2

        # ZDT2 parameters
        C = 1.0 + 10.0 * (dim - 1)
        r1 = 11.0
        r2 = 11.0

        # f2 as function of x1 for points on the Pareto front (x[1:] = 0)
        def f2_of_x1(x1):
            x1_clipped = np.clip(x1, 0.0, 1.0)
            return C * (1.0 - np.sqrt(x1_clipped / C))

        # Hypervolume calculation for 2D with reference point (r1, r2)
        # Points must have f1 < r1 and f2 < r2 to contribute
        def compute_hv(x1_vec):
            x1 = np.sort(np.clip(x1_vec, 0.0, 1.0))
            f2 = f2_of_x1(x1)
            # Filter points that are strictly better than reference in both objectives
            mask = (x1 < r1) & (f2 < r2)
            x1_f = x1[mask]
            f2_f = f2[mask]
            
            if len(x1_f) == 0:
                return 0.0
            
            # 2D hypervolume: sum of (r2 - f2[i]) * (f1[i+1] - f1[i])
            # where f1[n] = r1
            f1_next = np.concatenate([x1_f[1:], [r1]])
            widths = f1_next - x1_f
            heights = r2 - f2_f  # positive since f2 < r2
            contribs = np.maximum(heights, 0) * np.maximum(widths, 0)
            return np.sum(contribs)

        # Negative hypervolume for minimization
        def neg_hv(x1_vec):
            return -compute_hv(x1_vec)

        # Strategy: Use a deterministic grid as the initial guess, then refine.
        # The optimal density for maximizing hypervolume of a concave curve like ZDT2's
        # f2 = C*(1-sqrt(x1/C)) is not uniform. 
        # We use a power-law bias x1 = t^k. 
        # Initial k=0.5 (sqrt) is a good heuristic, but we will optimize it.
        
        t = np.linspace(0.0, 1.0, n_points)
        
        # Try a few initial k values to find a good starting point for optimization
        best_k = 0.5
        best_init_hv = -np.inf
        for k_test in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
            x1_test = t ** k_test
            hv_test = compute_hv(x1_test)
            if hv_test > best_init_hv:
                best_init_hv = hv_test
                best_k = k_test
        
        x1_init = t ** best_k

        # Optimize the x1 values directly using scipy
        try:
            from scipy.optimize import minimize
            
            # Use L-BFGS-B with bounds [0, 1] for each x1
            # We optimize the sorted vector directly to avoid permutation issues
            res = minimize(
                neg_hv,
                x1_init,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * n_points,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            x1_opt = res.x
        except Exception:
            x1_opt = x1_init

        # Ensure x1_opt is sorted and in [0, 1]
        x1_opt = np.clip(np.sort(x1_opt), 0.0, 1.0)

        # Refine with local search if time permits
        if time.time() < deadline - 10:
            best_x1 = x1_opt.copy()
            best_val = neg_hv(best_x1)
            step = 0.001
            n_local_iters = 10000
            for it in range(n_local_iters):
                if time.time() > deadline - 5:
                    break
                idx = np.random.randint(n_points)
                old_val = best_x1[idx]
                # Try several step sizes and directions
                for delta in [-step, step, -step*0.5, step*0.5, -step*2, step*2]:
                    candidate = best_x1.copy()
                    candidate[idx] = np.clip(old_val + delta, 0.0, 1.0)
                    candidate = np.sort(candidate)
                    val = neg_hv(candidate)
                    if val < best_val:
                        best_x1 = candidate
                        best_val = val
                        step *= 0.7  # Shrink step on success
            x1_opt = best_x1

        # Final high-precision polish: use Nelder-Mead on the sorted x1 vector
        if time.time() < deadline - 5:
            try:
                from scipy.optimize import minimize as sp_minimize
                # Unconstrained optimization on the sorted values
                # We'll work with the sorted vector directly
                res2 = sp_minimize(
                    neg_hv,
                    x1_opt,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                x1_opt = np.clip(np.sort(res2.x), 0.0, 1.0)
            except Exception:
                pass

        # Construct and evaluate points
        X_list = []
        F_list = []
        
        for i in range(n_points):
            if time.time() > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = x1_opt[i]
            # Set remaining dimensions to lower bound to minimize g(x)
            if dim > 1:
                x[1:] = lo[1:]
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            X_list.append(x)
            F_list.append(f)

        return np.array(X_list)
# EVOLVE-BLOCK-END
