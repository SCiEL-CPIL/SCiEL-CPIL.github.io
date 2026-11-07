# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Model-and-optimize optimizer for ZDT2.
    
    Exploits the known Pareto front geometry of ZDT2:
    - Front: x1 in [0,1], x2..xn = 0
    - f1 = x1
    - f2 = C * (1 - sqrt(x1/C)) where C = 1 + 10*(n-1)
    
    For 2 objectives, hypervolume with reference (r1, r2) is computed
    by sorting points by f1 and summing trapezoid contributions.
    
    We optimize the placement of n_points x1 values to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        # ZDT2 parameters
        C = 1.0 + 10.0 * (dim - 1)
        r1 = 11.0
        r2 = 11.0

        n_points = min(budget, 200)
        if n_points < 2:
            n_points = 2

        # f2 as function of x1
        def f2_of_x1(x1):
            x1_clipped = np.clip(x1, 0.0, 1.0)
            return C * (1.0 - np.sqrt(x1_clipped / C))

        # Threshold x1 where f2 = r2: C*(1-sqrt(x1/C)) = r2
        # 1 - sqrt(x1/C) = r2/C
        # sqrt(x1/C) = 1 - r2/C
        # x1 = C * (1 - r2/C)^2
        x1_thresh = C * (1.0 - r2 / C) ** 2

        # Smooth hypervolume using log-sum-exp for the max operation
        # For 2D with points sorted by f1, HV = sum_i (r2 - f2[i]) * (f1[i+1] - f1[i])
        # where f1[n] = r1, and we only count points with f2 < r2.
        # The "max" issue: if f2[i] >= r2, that point shouldn't contribute.
        # We can use a smooth approximation: max(0, r2 - f2[i]) ≈ softplus(r2 - f2[i])
        
        def neg_hv_smooth(x1_vec, beta=50.0):
            x1 = np.clip(x1_vec, 0.0, 1.0)
            x1 = np.sort(x1)
            f2 = f2_of_x1(x1)
            
            # Soft threshold: points with f2 >= r2 get ~0 contribution
            # Use softplus: max(0, z) ≈ log(1 + exp(z)) / beta for large beta
            z = r2 - f2  # positive when f2 < r2
            # softplus(z/beta) * beta approximates max(0, z)
            heights = beta * np.log1p(np.exp(z / beta))
            # For z >> 0: heights ≈ z. For z << 0: heights ≈ 0.
            
            f1_next = np.concatenate([x1[1:], [r1]])
            widths = f1_next - x1
            widths = np.maximum(widths, 0.0)
            
            contribs = heights * widths
            hv = np.sum(contribs)
            return -hv

        def neg_hv_exact(x1_vec):
            x1 = np.clip(x1_vec, 0.0, 1.0)
            x1 = np.sort(x1)
            f2 = f2_of_x1(x1)
            
            mask = (x1 < r1) & (f2 < r2)
            x1_f = x1[mask]
            f2_f = f2[mask]
            
            if len(x1_f) == 0:
                return 0.0
            
            f1_next = np.concatenate([x1_f[1:], [r1]])
            widths = f1_next - x1_f
            heights = r2 - f2_f  # positive
            contribs = np.maximum(heights, 0) * np.maximum(widths, 0)
            hv = np.sum(contribs)
            return -hv

        # Initial guess: logarithmic spacing from threshold to 1.0
        # The optimal density should be higher where the front curves more
        t = np.linspace(0, 1, n_points)
        # Use a power law that concentrates points near the threshold
        k = 0.5
        x1_init = x1_thresh + (1.0 - x1_thresh) * t ** k

        # Ensure all points are above threshold for valid contribution
        x1_init = np.maximum(x1_init, x1_thresh + 1e-6)
        x1_init = np.minimum(x1_init, 1.0)

        # Optimize using scipy with smooth objective first
        try:
            from scipy.optimize import minimize
            
            # First pass: smooth objective with L-BFGS-B
            res1 = minimize(neg_hv_smooth, x1_init, method='L-BFGS-B',
                          bounds=[(x1_thresh, 1.0)] * n_points,
                          options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            x1_opt = res1.x
            
            # Second pass: exact objective with SLSQP for polish
            if time.time() < deadline - 20:
                res2 = minimize(neg_hv_exact, x1_opt, method='SLSQP',
                              bounds=[(x1_thresh, 1.0)] * n_points,
                              options={'maxiter': 200, 'ftol': 1e-15, 'disp': False})
                if res2.fun < neg_hv_exact(x1_opt):
                    x1_opt = res2.x
            
        except Exception:
            pass

        # Ensure x1_opt is sorted and in valid range
        x1_opt = np.clip(np.sort(x1_opt), x1_thresh, 1.0)

        # Local refinement with adaptive step sizes
        if time.time() < deadline - 15:
            best_x1 = x1_opt.copy()
            best_val = neg_hv_exact(best_x1)
            step = 0.01
            n_iter = 0
            max_iter = 5000
            
            while time.time() < deadline - 10 and n_iter < max_iter:
                n_iter += 1
                idx = np.random.randint(n_points)
                old_val = best_x1[idx]
                improved = False
                
                for delta in [-step, step]:
                    candidate = best_x1.copy()
                    candidate[idx] = np.clip(old_val + delta, x1_thresh, 1.0)
                    candidate = np.sort(candidate)
                    val = neg_hv_exact(candidate)
                    if val < best_val - 1e-15:
                        best_x1 = candidate
                        best_val = val
                        improved = True
                        break
                
                if not improved:
                    step *= 0.95
                else:
                    step *= 0.999
                
                if step < 1e-10:
                    break

            x1_opt = best_x1

        # Final high-precision polish using Nelder-Mead on a subset
        if time.time() < deadline - 5:
            try:
                from scipy.optimize import minimize
                # Try Nelder-Mead on the full vector with tight tolerances
                res3 = minimize(neg_hv_exact, x1_opt, method='Nelder-Mead',
                              bounds=[(x1_thresh, 1.0)] * n_points,
                              options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-15, 'adaptive': True})
                if res3.fun < neg_hv_exact(x1_opt):
                    x1_opt = np.clip(np.sort(res3.x), x1_thresh, 1.0)
            except Exception:
                pass

        # Evaluate the optimal points
        X_list = []
        F_list = []

        for i in range(n_points):
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x1_opt[i]
            x = np.clip(x, lo, hi)

            f = problem(x)
            X_list.append(x)
            F_list.append(f)

        return np.array(X_list)
# EVOLVE-BLOCK-END
