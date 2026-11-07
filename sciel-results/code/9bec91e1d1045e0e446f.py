"""Rewritten optimizer: Multi-start Powell with (1+1)-ES polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start Powell's method with (1+1)-ES polish and a hard wall-clock deadline."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 295.0  # 5 s safety margin before the 300 s eval limit

        def elapsed_ok():
            return time.time() < deadline

        rng = np.random.default_rng(12345)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        evaluated = 0

        # ---------- Phase 1: coarse random sampling to locate the basin ----------
        # Use 15 starts to ensure we find the basin in [-4, 4]^5
        n_starts = 15
        candidates = rng.uniform(-4.0, 4.0, size=(n_starts, dim))
        # Include origin and a few structured points
        extra = np.array([
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0
        ])
        candidates = np.vstack([candidates, extra])
        
        start_xs = []
        start_fs = []
        for i in range(len(candidates)):
            if not elapsed_ok() or evaluated >= budget:
                break
            x = clip(candidates[i])
            f = func(x)
            evaluated += 1
            start_xs.append(x)
            start_fs.append(f)
            if f < best_f:
                best_f, best_x = f, x.copy()

        # ---------- Phase 2: Custom Nelder-Mead from the best start ----------
        remaining = budget - evaluated
        if remaining > 0 and best_x is not None and elapsed_ok():
            x = best_x.copy()
            # Initialize simplex
            simplex = [x.copy()]
            for i in range(dim):
                new_pt = x.copy()
                # Step size for initial simplex
                step = 0.1 if abs(x[i]) < 0.1 else 0.1 * x[i]
                new_pt[i] += step
                new_pt = clip(new_pt)
                simplex.append(new_pt)
            
            # Evaluate initial simplex
            simplex_f = []
            for pt in simplex:
                if evaluated >= budget or not elapsed_ok():
                    break
                f_val = func(pt)
                evaluated += 1
                simplex_f.append(f_val)
                if f_val < best_f:
                    best_f, best_x = f_val, pt.copy()
            
            # Nelder-Mead iterations
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            max_iter = remaining // (dim + 1) # Rough estimate of max feasible iterations
            
            for _ in range(max_iter):
                if evaluated >= budget or not elapsed_ok():
                    break
                # Sort simplex by function value
                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = clip(centroid + alpha * (centroid - simplex[-1]))
                fr = func(xr)
                evaluated += 1
                if fr < best_f:
                    best_f, best_x = fr, xr.copy()
                    
                if simplex_f[0] <= fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                elif fr < simplex_f[0]:
                    # Expansion
                    xe = clip(centroid + gamma * (centroid - simplex[-1]))
                    fe = func(xe)
                    evaluated += 1
                    if fe < best_f:
                        best_f, best_x = fe, xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                else:
                    # Contraction
                    if fr < simplex_f[-1]:
                        # Outside contraction
                        xc = clip(centroid + rho * (simplex[-1] - centroid))
                    else:
                        # Inside contraction
                        xc = clip(centroid - rho * (simplex[-1] - centroid))
                    fc = func(xc)
                    evaluated += 1
                    if fc < best_f:
                        best_f, best_x = fc, xc.copy()
                    
                    if fc < min(fr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            if evaluated >= budget or not elapsed_ok():
                                break
                            simplex_f[i] = func(simplex[i])
                            evaluated += 1
                            if simplex_f[i] < best_f:
                                best_f, best_x = simplex_f[i], simplex[i].copy()

        # ---------- Phase 3: final fine polish with (1+1)-ES ----------
        remaining = budget - evaluated
        if remaining > 0 and best_x is not None and elapsed_ok():
            x = best_x.copy()
            fx = best_f
            # Start with a small sigma for polish
            sigma = 0.01
            successes = 0
            window = 0
            WIN = 5
            for _ in range(remaining):
                if not elapsed_ok() or evaluated >= budget:
                    break
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = func(y)
                evaluated += 1
                window += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                if window >= WIN:
                    if successes / window > 1.0 / 5.0:
                        sigma *= 1.3
                    else:
                        sigma *= 0.7
                    sigma = float(np.clip(sigma, 1e-6, 1.0))
                    successes = 0
                    window = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
