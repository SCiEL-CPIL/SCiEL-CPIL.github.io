"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Cuckoo Search with Levy flights followed by Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0
        max_calls = self.budget

        def call_func(x):
            nonlocal calls, best_x, best_f
            if calls >= max_calls:
                return best_f
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        def levy_flight(beta=1.5):
            sigma_l = (
                (np.gamma(1 + beta) * np.sin(np.pi * beta / 2))
                / (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
            ) ** (1 / beta)
            u = rng.normal(0, 1, self.dim)
            s = rng.standard_cauchy(self.dim)
            return sigma_l * u / np.abs(s) ** (1 / beta)

        # Phase 1: Cuckoo Search global exploration
        n_nests = 15
        pa = 0.3  # probability of replacing nests
        cs_budget = int(max_calls * 0.6)

        # Initialize nests with random points
        nests = [rng.uniform(-4.5, 4.5, self.dim) for _ in range(n_nests)]
        nest_f = []
        for nest in nests:
            f = call_func(nest)
            nest_f.append(f)

        # Track best
        best_idx = np.argmin(nest_f)

        max_cs_iters = 50
        for it in range(max_cs_iters):
            if calls >= cs_budget or time.time() > deadline:
                break

            for j in range(n_nests):
                if calls >= cs_budget or time.time() > deadline:
                    break

                # Generate new solution via Levy flight
                step = levy_flight()
                new_x = nests[j] + 0.01 * step
                new_x = np.clip(new_x, -5.0, 5.0)
                new_f = call_func(new_x)

                # Replace if better
                if new_f < nest_f[j]:
                    nests[j] = new_x
                    nest_f[j] = new_f
                    if new_f < best_f:
                        best_f = new_f
                        best_idx = j

            # Replace worst nests with probability pa
            worst_idx = np.argmax(nest_f)
            if rng.random() < pa:
                new_x = rng.uniform(-4.5, 4.5, self.dim)
                new_f = call_func(new_x)
                nests[worst_idx] = new_x
                nest_f[worst_idx] = new_f
                if new_f < best_f:
                    best_f = new_f
                    best_idx = worst_idx

        # Phase 2: Nelder-Mead refinement on best candidate
        nm_budget = max_calls - calls
        if nm_budget > 10 and best_x is not None:
            x0 = best_x.copy()
            f0 = best_f

            simplex = [x0.copy()]
            fvals = [f0]

            # Create simplex
            for i in range(self.dim):
                xi = x0.copy()
                if abs(xi[i]) > 1e-6:
                    xi[i] += 0.05 * abs(xi[i])
                else:
                    xi[i] += 0.3
                xi = np.clip(xi, -5.0, 5.0)
                fi = call_func(xi)
                simplex.append(xi)
                fvals.append(fi)

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            max_nm_iters = min(200, nm_budget)
            for _ in range(max_nm_iters):
                if calls >= max_calls:
                    break
                if time.time() > deadline:
                    break

                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                if abs(fvals[-1] - fvals[0]) < 1e-12:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = call_func(xr)

                if fr < fvals[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = call_func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    if fr < fvals[-1]:
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = call_func(xc)
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = call_func(simplex[i])

            local_best_idx = np.argmin(fvals)
            if fvals[local_best_idx] < best_f:
                best_f = fvals[local_best_idx]
                best_x = simplex[local_best_idx].copy()

        if best_x is None:
            best_x = np.zeros(self.dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
