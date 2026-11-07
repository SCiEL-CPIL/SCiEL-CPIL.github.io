"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Sobol + Nelder-Mead optimizer for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        # Sobol-like quasi-random points via Halton sequence
        def halton(index, base, dim):
            """Generate a single Halton sequence point."""
            result = np.zeros(dim)
            for d in range(dim):
                f = 1.0
                i = index
                result[d] = 0.0
                while i > 0:
                    f /= base
                    i, rem = divmod(i, base)
                    result[d] += f * rem
            return result

        # Generate candidate points using Halton sequence
        candidates = []
        num_halton = 150  # number of global exploration points
        
        # Start with origin and a few strategic points
        candidates.append(np.zeros(dim))
        # Add corner-ish points
        for corner in range(2 ** min(dim, 4)):
            pt = np.zeros(dim)
            for d in range(dim):
                if corner & (1 << d):
                    pt[d] = 4.0
                else:
                    pt[d] = -4.0
            candidates.append(pt)
        
        # Halton points
        for i in range(1, num_halton + 1):
            pt = halton(i, 2, dim) * 10.0 - 5.0  # map [0,1]^d to [-5,5]^d
            pt = np.clip(pt, -5, 5)
            candidates.append(pt)

        # Evaluate all candidates
        results = []
        for pt in candidates:
            if time.time() > deadline:
                break
            val = func(pt)
            results.append((val, pt))

        # Sort by function value
        results.sort(key=lambda x: x[0])
        
        # Take top candidates for local refinement
        num_refine = min(8, len(results))
        best_val = results[0][0]
        best_x = results[0][1].copy()

        # Nelder-Mead refinement on top candidates
        def nelder_mead(func, x0, dim, max_iter=50, initial_step=0.5):
            """Simple Nelder-Mead implementation."""
            # Initialize simplex
            simplex = [x0.copy()]
            f_simplex = [func(x0)]
            
            for d in range(dim):
                x_new = x0.copy()
                x_new[d] += initial_step
                x_new = np.clip(x_new, -5, 5)
                simplex.append(x_new)
                f_simplex.append(func(x_new))

            # Sort simplex
            order = np.argsort(f_simplex)
            simplex = [simplex[i] for i in order]
            f_simplex = [f_simplex[i] for i in order]

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for _ in range(max_iter):
                if time.time() > deadline:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)

                if f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    if f_c < f_simplex[-1]:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])

                # Re-sort
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]

            return simplex[0], f_simplex[0]

        # Refine top candidates
        for val, pt in results[:num_refine]:
            if time.time() > deadline:
                break
            try:
                refined_x, refined_val = nelder_mead(func, pt, dim, max_iter=40, initial_step=0.3)
                if refined_val < best_val:
                    best_val = refined_val
                    best_x = refined_x
            except Exception:
                pass

        # If we still have budget, do a few more local refinements around the best point
        if time.time() < deadline - 10:
            rng = np.random.default_rng(42)
            for _ in range(5):
                if time.time() > deadline:
                    break
                # Perturb best point
                x_pert = best_x + rng.normal(0, 0.2, dim)
                x_pert = np.clip(x_pert, -5, 5)
                val_pert = func(x_pert)
                if val_pert < best_val:
                    best_val = val_pert
                    best_x = x_pert
                # Quick local search around perturbed point
                try:
                    rx, rv = nelder_mead(func, x_pert, dim, max_iter=20, initial_step=0.1)
                    if rv < best_val:
                        best_val = rv
                        best_x = rx
                except Exception:
                    pass

        return best_x, best_val
# EVOLVE-BLOCK-END
