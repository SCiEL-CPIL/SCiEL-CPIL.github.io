"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Deterministic coarse-to-fine search using Halton sequence with adaptive local refinement."""

    @staticmethod
    def _halton_sequence_fast(n, dim):
        """Faster Halton sequence generation using vectorized operations."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        base = primes[:dim]
        points = np.zeros((n, dim))
        for d in range(dim):
            b = base[d]
            idx = np.arange(1, n + 1)
            f = 1.0 / b
            frac = np.zeros(n)
            i_val = idx.copy()
            while np.any(i_val > 0):
                frac += f * (i_val % b)
                i_val //= b
                f /= b
            points[:, d] = frac
        return points

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.low = -5.0
        self.high = 5.0

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        low, high = self.low, self.high

        # Phase 1: Coarse Halton search - use ~40% of budget
        n_coarse = max(20, min(120, budget // 3))
        halton = self._halton_sequence_fast(n_coarse, dim)
        x_coarse = low + (high - low) * halton

        best_x = None
        best_f = np.inf
        calls_made = 0
        evals = []  # store (fx, x) for top-k selection

        for i in range(n_coarse):
            x = x_coarse[i]
            fx = func(x)
            calls_made += 1
            evals.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort to get top-k points for multi-start local search
        evals.sort(key=lambda t: t[0])
        k_restarts = min(3, len(evals), max(1, (budget - calls_made) // 30))

        remaining = budget - calls_made

        # Phase 2: Local refinement using Nelder-Mead-like simplex method
        for k in range(k_restarts):
            if calls_made >= budget:
                break
            start_x = evals[k][1].copy()
            start_f = evals[k][0]

            # Simplex initialization
            alpha = 0.5
            simplex = [start_x.copy()]
            for d in range(dim):
                vertex = start_x.copy()
                vertex[d] += alpha
                vertex = np.clip(vertex, low, high)
                simplex.append(vertex)

            # Evaluate initial simplex
            simplex_f = [start_f]
            for v in simplex[1:]:
                if calls_made >= budget:
                    break
                fv = func(v)
                calls_made += 1
                simplex_f.append(fv)

            # Nelder-Mead iterations
            max_iter = max(5, (budget - calls_made) // (dim + 1))
            for it in range(max_iter):
                if calls_made >= budget:
                    break

                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i].copy() for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                gamma = 1.0
                xr = centroid + gamma * (centroid - simplex[-1])
                xr = np.clip(xr, low, high)
                if calls_made >= budget:
                    break
                fr = func(xr)
                calls_made += 1

                if fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    xe = np.clip(xe, low, high)
                    if calls_made >= budget:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                        break
                    fe = func(xe)
                    calls_made += 1
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
                    # Contraction
                    beta = 0.5
                    xc = centroid + beta * (simplex[-1] - centroid)
                    xc = np.clip(xc, low, high)
                    if calls_made >= budget:
                        break
                    fc = func(xc)
                    calls_made += 1
                    if fc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], low, high)
                            if calls_made >= budget:
                                break
                            fi = func(simplex[i])
                            calls_made += 1
                            simplex_f[i] = fi
                        if calls_made >= budget:
                            break

                # Update best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_f:
                    best_f = simplex_f[min_idx]
                    best_x = simplex[min_idx].copy()

        # Phase 3: If budget remains, do fine Halton around best
        remaining = budget - calls_made
        if remaining > 15:
            center = best_x.copy()
            # Adaptive radius based on dim
            radius = 1.5
            n_fine = min(remaining - 5, 40)
            if n_fine > 0:
                halton_fine = self._halton_sequence_fast(n_fine, dim)
                x_fine = center + radius * (2 * halton_fine - 1)
                x_fine = np.clip(x_fine, low, high)

                for i in range(n_fine):
                    if calls_made >= budget:
                        break
                    x = x_fine[i]
                    fx = func(x)
                    calls_made += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

                # Final local polish
                if calls_made < budget - 5:
                    x = best_x.copy()
                    fx = best_f
                    step = 0.3
                    while calls_made < budget and step > 0.001:
                        improved = False
                        for d in range(dim):
                            for direction in [-1, 1]:
                                y = x.copy()
                                y[d] += direction * step
                                y = np.clip(y, low, high)
                                if calls_made >= budget:
                                    break
                                fy = func(y)
                                calls_made += 1
                                if fy < fx:
                                    x = y
                                    fx = fy
                                    improved = True
                            if calls_made >= budget:
                                break
                        if not improved:
                            step *= 0.5
                        else:
                            step *= 0.85
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
