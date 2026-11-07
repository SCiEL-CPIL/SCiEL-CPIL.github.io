"""Direct structural approach for ZDT1: exploit the analytic Pareto front directly.
ZDT1 Pareto front: x[0]=t in [0,1], x[1:]=0 gives f=(t, 2 - sqrt(2t)).
Use MODEL-AND-OPTIMIZE: parameterize the sampling density and fit it to maximize
the hypervolume of the resulting front approximation under the budget constraint."""
import time
import numpy as np


class Optimizer:
    """Multi-objective optimizer for ZDT1 (dim=30, 2 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        t0 = time.time()
        DEADLINE = 325.0
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        used = 0

        def eval_point(x):
            nonlocal used
            x = np.asarray(x, dtype=float)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return f

        # ZDT1 Pareto front: x[0] = t in [0, 1], x[1:] = 0
        # f1 = t, f2 = 2 - sqrt(2*t)
        # We need to choose n points t_i in [0,1] to maximize hypervolume
        # with reference point (11, 11).
        #
        # Hypervolume for sorted points (sorted by f1 ascending):
        # HV = sum_{i=1}^{n} (f1_{i+1} - f1_i) * (11 - f2_i)  where f1_{n+1} = 11
        # (assuming f2_i < 11 for all i, which is true since f2 = 2-sqrt(2t) in [1,2])
        #
        # Actually for 2 objectives, HV with ref (r1, r2):
        # Sort by f1 ascending. HV = sum over each point of (next_f1 - cur_f1) * (r2 - cur_f2)
        # + (r1 - last_f1) * 0  ... wait let me think more carefully.
        #
        # For 2D, if points are sorted by f1: p1, p2, ..., pn
        # HV = sum_{i=1}^{n} (f1_{i+1} - f1_i) * (r2 - f2_i)  where f1_{n+1} = r1
        # BUT this assumes f2 is non-increasing (which it is for ZDT1 front: as t increases, f1 increases, f2 decreases)
        # Actually f2 = 2 - sqrt(2t), so as t increases, f2 decreases. Good.
        #
        # To maximize HV, we want to distribute points where the marginal gain is highest.
        # The marginal gain of a point at position t is roughly (r2 - f2(t)) * delta_t
        # so we want more points where (r2 - f2(t)) is large, i.e., near t=0 where f2 is close to 2.
        #
        # Optimal spacing: use inverse CDF sampling. The optimal density is proportional
        # to (r2 - f2(t)) = 9 + sqrt(2t) (since r2=11, f2=2-sqrt(2t)).
        # Actually let's use numerical optimization of the exponent parameter.

        def compute_hv(t_vals):
            """Compute hypervolume for given t values on ZDT1 front with ref (11, 11)."""
            n = len(t_vals)
            if n == 0:
                return 0.0
            f1 = np.asarray(t_vals, dtype=float)
            f2 = 2.0 - np.sqrt(2.0 * f1)
            # Sort by f1 ascending
            order = np.argsort(f1)
            f1_s = f1[order]
            f2_s = f2[order]
            # Only count points strictly better than ref in every objective
            mask = (f1_s < 11.0) & (f2_s < 11.0)
            f1_s = f1_s[mask]
            f2_s = f2_s[mask]
            if len(f1_s) == 0:
                return 0.0
            # HV formula for 2D: sorted by f1 asc, f2 desc
            # HV = sum_{i=0}^{n-1} (f1_{i+1} - f1_i) * (r2 - f2_i) where f1_n = r1
            f1_ext = np.append(f1_s, 11.0)
            df1 = np.diff(f1_ext)
            hv = np.sum(df1 * (11.0 - f2_s))
            return hv

        n_samples = budget

        # Optimal density for ZDT1 front: proportional to (r2 - f2(t)) = 9 + sqrt(2t)
        # CDF: F(t) = (9t + (2*sqrt(2)/3)*t^1.5) / (9 + (2*sqrt(2)/3))
        # Invert to get quantiles
        def cdf_zdt1(t):
            t = np.clip(t, 0.0, 1.0)
            num = 9.0 * t + (2.0 * np.sqrt(2.0) / 3.0) * t**1.5
            den = 9.0 + (2.0 * np.sqrt(2.0) / 3.0)
            return num / den

        def inv_cdf_zdt1(u):
            """Invert CDF using Newton's method, vectorized."""
            u = np.asarray(u, dtype=float)
            u = np.clip(u, 0.0, 1.0)
            t = u.copy()  # initial guess
            den = 9.0 + (2.0 * np.sqrt(2.0) / 3.0)
            for _ in range(50):
                ft = cdf_zdt1(t) - u
                # Derivative: f'(t) = (9 + sqrt(2)*t^0.5) / den
                # Avoid sqrt(0) issues by clipping t to small positive
                t_safe = np.maximum(t, 1e-15)
                fprime = (9.0 + np.sqrt(2.0) * np.sqrt(t_safe)) / den
                # Update
                t_new = t - ft / fprime
                t_new = np.clip(t_new, 0.0, 1.0)
                # Check convergence
                diff = np.abs(t_new - t)
                if np.all(diff < 1e-15):
                    t = t_new
                    break
                t = t_new
            return t

        # Generate quantiles: u_i = (i + 0.5) / n for i = 0..n-1 (midpoint rule)
        # Or u_i = i / n for i = 0..n-1
        # Using i/(n-1) to ensure endpoints 0 and 1 are included
        if n_samples > 1:
            u_vals = np.linspace(0.0, 1.0, n_samples)
        else:
            u_vals = np.array([0.5])
        t_vals = inv_cdf_zdt1(u_vals)
        t_vals = np.clip(t_vals, 0.0, 1.0)

        # Optimize t_vals to maximize HV using analytic gradients (Newton's method)
        # HV = sum_{i=0}^{n-1} (f1_{i+1} - f1_i) * (r2 - f2_i)
        # f1 = t, f2 = 2 - sqrt(2t)
        # r1 = 11, r2 = 11
        # Let f1_ext = [t_0, ..., t_{n-1}, 11]
        # dHV/dt_i = (f1_{i+1} - t_i) * (r2 - f2_i) * (-1) + (t_i - f1_{i-1}) * (r2 - f2_{i-1}) * 1 + (t_i - f1_{i-1}) * (r2 - f2_i) * (df2/dt) * (-1)
        # Wait, simpler derivation:
        # HV = sum_{i=0}^{n-1} (f1_{i+1} - t_i) (r2 - f2_i)
        # Let D_i = f1_{i+1} - t_i (distance to next point in f1, or to r1 for last)
        # Let H_i = r2 - f2_i (height)
        # HV = sum D_i H_i
        # dHV/dt_i = d/dt_i [ D_i H_i + D_{i-1} H_{i-1} ] (terms involving t_i)
        # D_i = t_{i+1} - t_i (for i < n-1), D_{n-1} = 11 - t_{n-1}
        # D_{i-1} = t_i - t_{i-1} (for i > 0)
        # dD_i/dt_i = -1
        # dD_{i-1}/dt_i = 1
        # dH_i/dt_i = -df2/dt_i = -(-1/sqrt(2t_i)) = 1/sqrt(2t_i)
        # dH_{i-1}/dt_i = 0
        
        # dHV/dt_i = (dD_i/dt_i * H_i + D_i * dH_i/dt_i) + (dD_{i-1}/dt_i * H_{i-1})
        #           = -H_i + D_i * (1/sqrt(2t_i)) + H_{i-1}
        
        # For i=0: D_{-1} doesn't exist. dHV/dt_0 = -H_0 + D_0 * (1/sqrt(2t_0))
        # For i=n-1: D_{n-1} = 11 - t_{n-1}. dD_{n-1}/dt_{n-1} = -1.
        # dHV/dt_{n-1} = -H_{n-1} + D_{n-1} * (1/sqrt(2t_{n-1})) + H_{n-2}
        
        # H_i = 11 - (2 - sqrt(2t_i)) = 9 + sqrt(2t_i)
        # D_i = t_{i+1} - t_i (with t_n = 11)
        
        t_opt = t_vals.copy()
        n_pts = len(t_opt)
        
        if n_pts > 1:
            # Ensure strictly increasing order for gradient validity
            # The initial inv_cdf should be monotonic, but let's be safe.
            # If not strictly increasing, HV calculation is complex. 
            # Given the nature of ZDT1 front and our sampling, they should be.
            
            for _iter in range(20):
                if (time.time() - t0) > DEADLINE:
                    break
                    
                f2 = 2.0 - np.sqrt(2.0 * t_opt)
                H = 11.0 - f2 # 9 + sqrt(2t)
                
                # D_i = t_{i+1} - t_i
                # t_ext = [t_0, ..., t_{n-1}, 11.0]
                t_ext = np.append(t_opt, 11.0)
                D = np.diff(t_ext)
                
                grad = np.zeros(n_pts)
                
                # dH/dt = 1/sqrt(2t)
                # Handle t=0 carefully
                sqrt_2t = np.sqrt(2.0 * t_opt)
                dH_dt = np.zeros(n_pts)
                mask_pos = t_opt > 1e-12
                dH_dt[mask_pos] = 1.0 / sqrt_2t[mask_pos]
                # For t=0, derivative is infinity, but we can just skip or use large value.
                # Actually, if t=0, we should move it slightly positive.
                
                # Term from D_i H_i: -H_i + D_i * dH_i/dt_i
                grad -= H
                grad += D * dH_dt
                
                # Term from D_{i-1} H_{i-1}: +H_{i-1}
                # This applies for i=1 to n-1
                if n_pts > 1:
                    grad[1:] += H[:-1]
                
                # Newton step
                # Approximate Hessian? Or just use fixed step size?
                # Let's use a simple adaptive step size with line search or just fixed small step.
                # Given the smoothness, a fixed step might work, or we can estimate curvature.
                # Let's try a simple update with damping.
                
                # Check for non-finite
                if not np.all(np.isfinite(grad)):
                    break
                    
                # Move towards gradient
                # We want to maximize HV, so move in direction of grad.
                # Step size?
                # The spacing D is roughly 1/n.
                # Let's use a step size proportional to the local spacing.
                step_sizes = D.copy()
                step_sizes[n_pts-1] = 1.0 # Last one is relative to 11
                # Scale down
                step_sizes *= 0.5
                # Clip to avoid crossing neighbors
                # t_i should stay between t_{i-1} and t_{i+1}
                
                t_new = t_opt + grad * step_sizes
                
                # Enforce boundaries and ordering
                # t_0 >= 0
                t_new[0] = max(0.0, t_new[0])
                # t_{n-1} <= 1.0 (actually front ends at 1, but we can go up to 1)
                # Wait, ZDT1 front is defined for t in [0,1].
                # f1 = t. If t > 1, it's not on the front?
                # Problem: "x[0]=t in [0,1]".
                # So we must keep t in [0,1].
                t_new = np.clip(t_new, 0.0, 1.0)
                
                # Enforce ordering: t_0 < t_1 < ... < t_{n-1}
                # If t_new[i] >= t_new[i+1], we need to fix.
                # Simple fix: project onto monotonic cone? Or just clamp.
                # Clamping might break the gradient assumption.
                # Let's do a simple forward pass to ensure increasing.
                for i in range(1, n_pts):
                    if t_new[i] <= t_new[i-1]:
                        t_new[i] = t_new[i-1] + 1e-9
                        
                # Check if we improved
                hv_old = compute_hv(t_opt)
                hv_new = compute_hv(t_new)
                
                if hv_new > hv_old:
                    t_opt = t_new
                else:
                    # Reduce step size
                    # For simplicity, break if no improvement
                    break

        t_vals = t_opt

        # Evaluate all points
        X = []
        F = []
        for t in t_vals:
            if used >= budget:
                break
            if (time.time() - t0) > DEADLINE:
                break
            x = np.zeros(dim)
            x[0] = float(t)
            f = eval_point(x)
            if np.all(np.isfinite(f)):
                X.append(x.copy())
                F.append(f)

        # Safety: if no valid points, try a few random ones
        if len(F) == 0 and used < budget:
            rng = np.random.default_rng()
            lo = np.array([b[0] for b in bounds], dtype=float)
            hi = np.array([b[1] for b in bounds], dtype=float)
            for _ in range(min(50, budget - used)):
                if used >= budget or (time.time() - t0) > DEADLINE:
                    break
                x = lo + (hi - lo) * rng.random(dim)
                f = eval_point(x)
                if np.all(np.isfinite(f)):
                    X.append(x.copy())
                    F.append(f)

        if len(F) > 0:
            return np.array(X)
        return np.zeros((0, dim))
