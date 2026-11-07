import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(12345)
        call_count = 0
        best_x = None
        best_f = np.inf
        self._best_x = None
        self._best_f = np.inf
        all_evaluated = []  # list of (x, f) for multi-start

        def safe_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget or time.time() > deadline - 2:
                raise ValueError("Budget or time exceeded")
            call_count += 1
            x_arr = np.asarray(x, dtype=float)
            val = float(func(x_arr))
            all_evaluated.append((np.array(x_arr, dtype=float), val))
            if np.isfinite(val) and val < best_f:
                best_f = val
                best_x = np.array(x_arr, dtype=float)
                self._best_x = best_x.copy()
                self._best_f = best_f
            return val

        # Phase 1: Structured sampling to locate the Rosenbrock valley
        # Rosenbrock valley: x_i ≈ x_{i-1}^2, minimum at all 1s (before transform)
        # Sample points along the valley structure and random points
        
        # Valley-structured points: for Rosenbrock, the valley follows x_i = x_{i-1}^2
        # Generate points that follow this pattern with varying starting values
        valley_pts = []
        for start_val in [-2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3]:
            pt = np.zeros(self.dim)
            pt[0] = start_val
            for i in range(1, self.dim):
                pt[i] = pt[i-1] ** 2
            # Also add small perturbations
            for scale in [0.0, 0.1, 0.5]:
                p = pt + rng.normal(0, scale, self.dim)
                valley_pts.append(p)
        
        # Random points
        n_random = 40
        random_pts = rng.uniform(-4, 4, (n_random, self.dim))
        
        # Points near the expected minimum region (all ones area)
        near_one_pts = []
        for _ in range(20):
            p = np.ones(self.dim) + rng.normal(0, 1.0, self.dim)
            near_one_pts.append(p)
        
        # Axis-aligned points
        axis_pts = []
        for i in range(self.dim):
            for v in [-3, -1, 0, 1, 2, 3]:
                p = np.zeros(self.dim)
                p[i] = v
                axis_pts.append(p)
        
        all_pts = np.vstack([valley_pts, random_pts, near_one_pts, axis_pts])
        
        for p in all_pts:
            if call_count >= self.budget or time.time() > deadline - 10:
                break
            try:
                safe_func(p)
            except ValueError:
                break
            except Exception:
                continue

        if best_x is None:
            return np.zeros(self.dim), 0.0

        # Phase 2: Multi-start L-BFGS-B from top diverse points
        remaining = self.budget - call_count
        if remaining > 10 and time.time() < deadline - 10:
            # Sort evaluated points by function value, pick diverse top candidates
            sorted_evals = sorted(all_evaluated, key=lambda t: t[1])
            # Pick top 5 diverse points (minimum distance between them)
            candidates = [sorted_evals[0][0]]
            for _, x in sorted_evals[1:]:
                if len(candidates) >= 5:
                    break
                # Check diversity
                diverse = True
                for c in candidates:
                    if np.linalg.norm(x - c) < 1.0:
                        diverse = False
                        break
                if diverse:
                    candidates.append(x)
            
            for start_pt in candidates:
                remaining = self.budget - call_count
                if remaining < 10 or time.time() > deadline - 5:
                    break
                try:
                    res = minimize(
                        safe_func,
                        start_pt,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={'maxiter': 300, 'maxfun': remaining, 'maxls': 50, 
                                 'gtol': 1e-12, 'ftol': 1e-14}
                    )
                    if np.isfinite(res.fun) and res.fun < best_f:
                        best_f = float(res.fun)
                        best_x = np.array(res.x, dtype=float)
                        self._best_x = best_x.copy()
                        self._best_f = best_f
                        all_evaluated.append((best_x.copy(), best_f))
                except Exception:
                    continue

        # Phase 3: Valley-following line search
        # Rosenbrock has a narrow valley; use custom line search along valley direction
        remaining = self.budget - call_count
        if remaining > 20 and time.time() < deadline - 5:
            # Estimate valley direction from gradient-free approach
            # Try moving along different directions from best point
            directions = []
            # Coordinate directions
            for i in range(self.dim):
                for sign in [-1, 1]:
                    d = np.zeros(self.dim)
                    d[i] = sign
                    directions.append(d)
            # Diagonal directions
            for _ in range(10):
                d = rng.normal(0, 1, self.dim)
                d /= (np.linalg.norm(d) + 1e-10)
                directions.append(d)
            
            # Try line search along each direction
            for d in directions:
                remaining = self.budget - call_count
                if remaining < 5 or time.time() > deadline - 3:
                    break
                # Golden section search along direction
                try:
                    # First, find a bracket
                    t_lo, t_hi = -1.0, 1.0
                    f_lo = float(func(np.clip(best_x + t_lo * d, -5, 5)))
                    f_hi = float(func(np.clip(best_x + t_hi * d, -5, 5)))
                    call_count += 2
                    
                    # Expand if needed
                    if f_hi < f_lo:
                        t_lo, t_hi = t_hi, t_hi * 2
                        f_lo = f_hi
                        f_hi = float(func(np.clip(best_x + t_hi * d, -5, 5)))
                        call_count += 1
                    elif f_lo < f_hi:
                        t_hi, t_lo = t_lo, t_lo * 2
                        f_hi = f_lo
                        f_lo = float(func(np.clip(best_x + t_lo * d, -5, 5)))
                        call_count += 1
                    
                    # Golden section search
                    gr = (np.sqrt(5) - 1) / 2
                    for _ in range(8):
                        if call_count >= self.budget or time.time() > deadline - 3:
                            break
                        t1 = t_lo + gr * (t_hi - t_lo)
                        t2 = t_hi - gr * (t_hi - t_lo)
                        f1 = float(func(np.clip(best_x + t1 * d, -5, 5)))
                        f2 = float(func(np.clip(best_x + t2 * d, -5, 5)))
                        call_count += 2
                        if f1 < f2:
                            t_hi, f_hi = t2, f2
                        else:
                            t_lo, f_lo = t1, f1
                    
                    # Evaluate at best point
                    t_best = (t_lo + t_hi) / 2
                    x_new = np.clip(best_x + t_best * d, -5, 5)
                    f_new = float(func(x_new))
                    call_count += 1
                    if np.isfinite(f_new) and f_new < best_f:
                        best_f = f_new
                        best_x = x_new
                        self._best_x = best_x.copy()
                        self._best_f = best_f
                        all_evaluated.append((best_x.copy(), best_f))
                except Exception:
                    continue

        # Phase 4: Nelder-Mead polish
        remaining = self.budget - call_count
        if remaining > 20 and time.time() < deadline - 5:
            try:
                res_nm = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                if np.isfinite(res_nm.fun) and res_nm.fun < best_f:
                    best_f = float(res_nm.fun)
                    best_x = np.array(res_nm.x, dtype=float)
                    self._best_x = best_x.copy()
                    self._best_f = best_f
            except Exception:
                pass

        # Phase 5: Final high-precision L-BFGS-B polish
        remaining = self.budget - call_count
        if remaining > 10 and time.time() < deadline - 3:
            try:
                res_final = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 1000, 'maxfun': remaining, 'maxls': 100, 
                             'gtol': 1e-15, 'ftol': 1e-16, 'disp': False}
                )
                if np.isfinite(res_final.fun) and res_final.fun < best_f:
                    best_f = float(res_final.fun)
                    best_x = np.array(res_final.x, dtype=float)
                    self._best_x = best_x.copy()
                    self._best_f = best_f
            except Exception:
                pass

        return best_x, best_f
