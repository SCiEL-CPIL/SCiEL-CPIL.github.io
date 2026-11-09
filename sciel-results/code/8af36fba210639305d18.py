"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead: grid seeds + local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # --- Phase 1: Generate diverse initial points via stratified grid ---
        # Use a coarse grid to cover the search space, then add some random points
        n_grid = 3  # 3 points per dimension -> 3^5 = 243 points (too many)
        # Use 2 per dim -> 32 points, plus center
        # Actually let's be smarter: use a space-filling approach
        # For 5D, use 25 well-spread points
        n_seeds = 25
        # Sobol-like deterministic points using Halton sequence
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        seeds = []
        # Add center
        seeds.append(np.zeros(dim))
        # Add Halton sequence points
        for i in range(1, n_seeds):
            pt = np.array([halton(i + 1, base) for base in range(2, dim + 2)])
            # Map to [-5, 5]
            pt = pt * 10.0 - 5.0
            seeds.append(pt)

        # Evaluate all seeds
        best_x = None
        best_f = np.inf
        seed_results = []
        for s in seeds:
            if time.time() > deadline:
                break
            s_clipped = np.clip(s, -5, 5)
            fs = func(s_clipped)
            seed_results.append((fs, s_clipped))
            if fs < best_f:
                best_f = fs
                best_x = s_clipped.copy()

        # Sort seeds by fitness
        seed_results.sort(key=lambda x: x[0])

        # --- Phase 2: Local refinement with Nelder-Mead on best seeds ---
        # Allocate remaining budget across top seeds
        calls_used = len(seed_results)
        remaining = budget - calls_used
        if remaining <= 0:
            return best_x, best_f

        # Use Nelder-Mead on top 5 seeds (or fewer if budget is tight)
        n_local = min(5, len(seed_results))
        per_seed_budget = max(10, remaining // n_local)

        for i in range(n_local):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break

            f0, x0 = seed_results[i]
            # Run a simple Nelder-Mead-like local search
            x = x0.copy()
            fx = f0
            # Initial simplex: perturb each dimension
            simplex = [x.copy()]
            for d in range(dim):
                v = x.copy()
                step = 0.5 if v[d] < 4.5 else -0.5
                v[d] += step
                v = np.clip(v, -5, 5)
                fv = func(v)
                calls_used += 1
                simplex.append((fv, v))
                if fv < best_f:
                    best_f = fv
                    best_x = v.copy()

            # Nelder-Mead iterations
            max_iters = per_seed_budget
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                if calls_used >= budget:
                    break

                # Sort simplex by fitness
                simplex.sort(key=lambda t: t[0])
                # Keep only dim+1 points (truncate if we have more)
                if len(simplex) > dim + 1:
                    simplex = simplex[:dim + 1]

                best = simplex[0]
                worst = simplex[-1]
                centroid = np.mean([pt for _, pt in simplex[:-1]], axis=0)

                # Reflection
                alpha = 1.0
                xr = np.clip(centroid + alpha * (centroid - worst[1]), -5, 5)
                fr = func(xr)
                calls_used += 1
                if fr < best[0]:
                    # Expansion
                    gamma = 2.0
                    xe = np.clip(centroid + gamma * (centroid - worst[1]), -5, 5)
                    fe = func(xe)
                    calls_used += 1
                    if fe < fr:
                        simplex[-1] = (fe, xe)
                        if fe < best_f:
                            best_f = fe
                            best_x = xe.copy()
                    else:
                        simplex[-1] = (fr, xr)
                        if fr < best_f:
                            best_f = fr
                            best_x = xr.copy()
                elif fr < worst[0]:
                    simplex[-1] = (fr, xr)
                    if fr < best_f:
                        best_f = fr
                        best_x = xr.copy()
                else:
                    # Contraction
                    rho = 0.5
                    xc = np.clip(centroid + rho * (worst[1] - centroid), -5, 5)
                    fc = func(xc)
                    calls_used += 1
                    if fc < worst[0]:
                        simplex[-1] = (fc, xc)
                        if fc < best_f:
                            best_f = fc
                            best_x = xc.copy()
                    else:
                        # Shrink
                        for j in range(1, len(simplex)):
                            simplex[j] = (simplex[j][0],
                                          np.clip(best[1] + 0.5 * (simplex[j][1] - best[1]), -5, 5))
                        # Re-evaluate shrunk points
                        for j in range(1, len(simplex)):
                            if time.time() > deadline or calls_used >= budget:
                                break
                            fj = func(simplex[j][1])
                            calls_used += 1
                            simplex[j] = (fj, simplex[j][1])
                            if fj < best_f:
                                best_f = fj
                                best_x = simplex[j][1].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
