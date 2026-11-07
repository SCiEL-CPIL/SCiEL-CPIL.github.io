"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid: coarse global grid -> multi-start CMA-ES ensemble -> L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit
        rng = np.random.default_rng(0)

        # --- Phase 0: coarse global grid to locate the basin ---
        # Discus has a narrow anisotropic valley; a coarse grid over the box
        # gives a cheap global picture of where the low region is.
        grid_pts = 3  # 3^5 = 243 evaluations
        grid_coords = np.linspace(-5.0, 5.0, grid_pts)
        grid = np.array(np.meshgrid(*([grid_coords] * dim)), dtype=float).reshape(-1, dim)
        # jitter to avoid exact grid points
        grid = grid + 0.01 * rng.standard_normal(grid.shape)
        grid = np.clip(grid, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        used = 0
        for p in grid:
            if used >= budget:
                break
            f = float(func(p))
            used += 1
            if f < best_f:
                best_f = f
                best_x = p.copy()
            if time.time() > deadline:
                break

        # --- Phase 1: multi-start CMA-ES ensemble from top seeds ---
        # Keep the top seeds from the grid as starting points for local CMA-ES.
        # We'll do a small population-based CMA-ES with restarts.

        def cmaes(x0, fx0, iters, seed):
            """Simple CMA-ES on a single start. Returns best (x, f)."""
            local_rng = np.random.default_rng(seed)
            n = dim
            x = x0.copy()
            f = fx0
            sigma = 1.5
            # CMA-ES state
            popsize = 4 + int(np.floor(3 * np.log(n)))
            # covariance matrix
            C = np.eye(n)
            # evolution paths
            ps = np.zeros(n)
            pc = np.zeros(n)
            # step sizes for paths
            cs = (popsize + 2) / (n + 4)
            ds = 1 + 2 * max(0, np.sqrt((popsize - 1) / (n + 1)) - 1) + cs
            cm = 1 / (n + np.sqrt(n))
            # eigen decomposition cached
            def update_eigen():
                nonlocal eigvals, eigvecs
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                return eigvals, eigvecs
            eigvals, eigvecs = update_eigen()
            for it in range(iters):
                if time.time() > deadline:
                    break
                D = np.sqrt(eigvals)
                for _ in range(popsize):
                    if time.time() > deadline:
                        break
                    # sample from N(0, C)
                    z = local_rng.standard_normal(n)
                    # B * D * z
                    step = (eigvecs * D) @ z
                    y = np.clip(x + sigma * step, -5.0, 5.0)
                    fy = float(func(y))
                    if fy < f:
                        f = fy
                        x = y.copy()
                # update mean with ranking (simple: use best popsize/2)
                # For simplicity, do a 1/5 style on the population mean of best half
                # Recompute eigen after a few iterations
                if it % 2 == 1:
                    eigvals, eigvecs = update_eigen()
                    # adapt sigma via simple success rule on this iteration
                # sigma adaptation (heuristic)
                sigma *= 0.97
                sigma = max(sigma, 1e-4)
                if f < 1e-6:
                    break
            return x, f

        # Use a few seeds: best grid point + a few random points near it
        seeds = [best_x.copy()]
        for _ in range(4):
            s = best_x + 2.0 * rng.standard_normal(dim)
            s = np.clip(s, -5.0, 5.0)
            seeds.append(s)

        # Allocate remaining budget among seeds
        remaining = budget - used
        if remaining > 0:
            per_seed = max(10, remaining // len(seeds))
            for i, s in enumerate(seeds):
                if time.time() > deadline:
                    break
                if used >= budget:
                    break
                fs = float(func(s))
                used += 1
                if used >= budget:
                    break
                iters = min(per_seed, budget - used)
                if iters < 5:
                    break
                xs, fs = cmaes(s, fs, iters, seed=100 + i)
                used += iters  # approximate; actual counted inside via func calls
                if fs < best_f:
                    best_f = fs
                    best_x = xs.copy()

        # --- Phase 2: L-BFGS-B polish from best point ---
        if time.time() < deadline and used < budget:
            try:
                res = minimize(func, best_x, method='L-BFGS-B',
                               bounds=[(-5.0, 5.0)] * dim,
                               options={'maxiter': 200, 'maxfun': budget - used})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
