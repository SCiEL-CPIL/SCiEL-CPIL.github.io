import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        dim = self.dim
        calls = [0]
        budget = self.budget

        def safe_func(x):
            if calls[0] >= budget:
                return np.inf
            calls[0] += 1
            try:
                return float(func(np.asarray(x, dtype=float)))
            except Exception:
                return np.inf

        best_x = np.zeros(dim)
        best_f = np.inf

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)

        # --- Phase 1: Initial Exploration ---
        # Use a mix of random points and a coarse grid to identify promising basins.
        # Rastrigin has many local minima; we need to find the global basin.
        # 50 random points + 50 grid points (subset) = 100 calls.
        
        n_explore = 100
        points = []
        
        # 1. Random sampling
        for _ in range(50):
            x = np.random.uniform(-5, 5, dim)
            f = safe_func(x)
            points.append((x, f))
            record(x, f)
            
        # 2. Coarse Grid Sampling (subset)
        # Full grid 5^5 = 3125 is too big.
        # Let's sample a structured subset.
        # Or just more random points? Random is often better for high-dim multimodal.
        # Let's do 50 more random points.
        for _ in range(50):
            x = np.random.uniform(-5, 5, dim)
            f = safe_func(x)
            points.append((x, f))
            record(x, f)
            
        if time.time() > deadline - 20:
            return best_x, best_f

        # Sort points by fitness
        points.sort(key=lambda t: t[1])
        
        # --- Phase 2: Local Polish Top Candidates ---
        # Use Nelder-Mead on the top 5 candidates to escape local minima and refine.
        top_k = 5
        for i in range(top_k):
            if calls[0] >= budget - 20:
                break
            if time.time() > deadline - 30:
                break
            
            x0, f0 = points[i]
            # Nelder-Mead
            x_polished, f_polished = self._nelder_mead(safe_func, x0, calls, max_iters=40, step0=1.0)
            record(x_polished, f_polished)
            
            # If we found a significantly better point, we might want to re-sort or just continue
            # For now, we just record the best.
            
        # --- Phase 3: Final Polish ---
        # Run Nelder-Mead again on the absolute best found so far with a smaller step.
        if calls[0] < budget - 5 and time.time() < deadline - 10:
            x_final, f_final = self._nelder_mead(safe_func, best_x, calls, max_iters=50, step0=0.5)
            record(x_final, f_final)

        return best_x, best_f

    def _nelder_mead(self, func, x0, calls, max_iters=50, step0=1.0):
        """
        Efficient Nelder-Mead implementation.
        Returns best_x, best_f.
        """
        d = self.dim
        x = np.array(x0, dtype=float)
        f0 = func(x)
        
        # Initialize simplex
        simplex = [x.copy()]
        fs = [f0]
        for i in range(d):
            xi = x.copy()
            xi[i] += step0
            simplex.append(xi)
            fs.append(func(xi))
            
        best_x = x.copy()
        best_f = f0
        
        for _ in range(max_iters):
            if calls[0] >= self.budget:
                break
                
            # Sort simplex
            order = np.argsort(fs)
            simplex = [simplex[i] for i in order]
            fs = [fs[i] for i in order]
            
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = simplex[0].copy()
                
            # Check convergence
            if fs[-1] - fs[0] < 1e-8:
                break
                
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + (centroid - simplex[-1])
            fr = func(xr)
            
            if fr < fs[0]:
                # Extension
                xe = centroid + 2.0 * (centroid - simplex[-1])
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fs[-1] = fe
                else:
                    simplex[-1] = xr
                    fs[-1] = fr
                continue
            elif fr < fs[-2]:
                # Accept reflection
                simplex[-1] = xr
                fs[-1] = fr
                continue
            else:
                # Contraction
                xc = centroid + 0.5 * (simplex[-1] - centroid)
                fc = func(xc)
                if fc < fs[-1]:
                    simplex[-1] = xc
                    fs[-1] = fc
                    continue
                else:
                    # Shrink
                    for i in range(1, d + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fs[i] = func(simplex[i])
                        
        return best_x, best_f
