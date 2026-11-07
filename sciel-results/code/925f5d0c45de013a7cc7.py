"""Combined optimizer for BBOB f23 (Katsuura): Multi-start Nelder-Mead + Adaptive (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive ES refinement for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        B = self.budget
        start_time = time.time()
        deadline = start_time + 270.0

        best_x = None
        best_f = np.inf
        calls = 0

        def safe_call(x):
            nonlocal calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return x, fx

        def time_left():
            return deadline - time.time()

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start Nelder-Mead
        # Katsuura is highly multimodal; use multiple starting points spread across the box
        n_starts = min(30, B // 20)
        
        # Generate spread-out starting points using a lattice + jitter
        n_per_dim = max(2, int(round(n_starts ** (1.0 / d))))
        grid_coords = np.linspace(-4.5, 4.5, n_per_dim)
        
        starts = []
        for idx in range(n_starts):
            if idx < n_per_dim ** d:
                coords = np.unravel_index(idx, [n_per_dim] * d)
                x0 = np.array([grid_coords[c] for c in coords], dtype=np.float64)
            else:
                x0 = rng.uniform(-4.5, 4.5, d)
            x0 = x0 + rng.normal(0, 0.15, d)
            x0 = np.clip(x0, -5.0, 5.0)
            starts.append(x0)

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        initial_simplex_size = 0.5

        # Track local minima from NM for ES refinement
        local_minima = []

        for start in starts:
            if calls >= B - 100 or time_left() < 40:
                break

            x0, f0 = safe_call(start)
            
            # Build initial simplex
            simplex = [x0.copy()]
            simplex_f = [f0]
            for i in range(d):
                xi = x0.copy()
                xi[i] += initial_simplex_size
                xi = np.clip(xi, -5.0, 5.0)
                xi, fi = safe_call(xi)
                simplex.append(xi)
                simplex_f.append(fi)

            # Nelder-Mead iterations
            max_nm_iters = 80
            for it in range(max_nm_iters):
                if calls >= B - 100 or time_left() < 40:
                    break

                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                if abs(simplex_f[0] - simplex_f[-1]) < 1e-12:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                xr, fr = safe_call(xr)

                if fr < simplex_f[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    xe, fe = safe_call(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    if fr < simplex_f[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5.0, 5.0)
                    xc, fc = safe_call(xc)

                    if fc < min(fr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            xi, fi = safe_call(simplex[i])
                            simplex[i] = xi
                            simplex_f[i] = fi

                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_f:
                    best_f = simplex_f[min_idx]
                    best_x = simplex[min_idx].copy()

            # Record local minimum from this NM run
            min_idx = np.argmin(simplex_f)
            local_minima.append((simplex_f[min_idx], simplex[min_idx].copy()))

        # Phase 2: Adaptive (1+1)-ES refinement on top candidates
        # Sort local minima and refine the best ones
        local_minima.sort(key=lambda t: t[0])
        top_k = min(8, len(local_minima))
        
        reserve_budget = 50
        available_for_es = B - calls - reserve_budget
        
        if available_for_es > 0:
            calls_per_candidate = max(15, available_for_es // max(1, top_k))
            
            for k in range(top_k):
                if calls >= B - 30 or time_left() < 20:
                    break
                
                fx_init, x_init = local_minima[k]
                
                # Adaptive (1+1)-ES with success-rate-based step adaptation
                x = x_init.copy()
                fx = fx_init
                sigma = 0.2  # Start with moderate step size
                
                window_size = 10
                successes = []
                
                max_steps = calls_per_candidate
                
                for _ in range(max_steps):
                    if calls >= B - 30 or time_left() < 20:
                        break
                    
                    y = x + sigma * rng.standard_normal(d)
                    y = np.clip(y, -5.0, 5.0)
                    y, fy = safe_call(y)
                    
                    if fy < fx:
                        successes.append(1)
                        x = y
                        fx = fy
                        sigma *= 1.05  # Increase step on success
                    else:
                        successes.append(0)
                        sigma *= 0.95  # Decrease step on failure
                    
                    if len(successes) > window_size:
                        successes.pop(0)
                    
                    # Adapt sigma based on success rate
                    if len(successes) == window_size:
                        sr = sum(successes) / window_size
                        if sr > 0.5:
                            sigma *= 1.02
                        elif sr < 0.2:
                            sigma *= 0.98
                    
                    sigma = np.clip(sigma, 1e-6, 2.0)

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 3: Final high-precision polish with very small adaptive steps
        if best_x is not None and calls < B - 10 and time_left() > 10:
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            
            while calls < B - 5 and time_left() > 5:
                y = np.clip(x + sigma * rng.standard_normal(d), -5.0, 5.0)
                y, fy = safe_call(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.005
                else:
                    sigma *= 0.95
                    if sigma < 1e-8:
                        break

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
