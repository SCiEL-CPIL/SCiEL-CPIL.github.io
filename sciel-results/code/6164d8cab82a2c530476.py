"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-restart and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(12345)

        def time_left():
            return deadline - time.time()

        best_x = None
        best_f = np.inf
        calls = 0

        # Phase 1: Initial exploration with diverse starting points
        starts = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            -np.ones(dim) * 2.0,
            np.ones(dim) * 3.5,
            -np.ones(dim) * 3.5,
            np.array([3.0, -3.0, 3.0, -3.0, 3.0]),
            np.array([-3.0, 3.0, -3.0, 3.0, -3.0]),
        ]
        for _ in range(10):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        for s in starts:
            if calls >= 18 or time_left() < 10:
                break
            x0 = np.clip(s, -5, 5)
            f0 = func(x0)
            calls += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        x = best_x.copy()
        fx = best_f

        # Phase 2: CMA-ES with restarts
        def run_cmaes(x_init, fx_init, calls_init, max_calls, max_time):
            nonlocal calls, best_f, best_x, fx, x

            lam = 14
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)

            c_sigma = (lam + 2) / (dim + lam + 5)
            c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + lam)
            c_mu = min(1 - c_1, 2 * ((lam / 2 + 1) / (dim + 2) - 1) / ((dim + 2) ** 2 + lam))
            chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            sigma = 1.5
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)

            local_best_f = fx_init
            local_best_x = x_init.copy()
            stagnant = 0
            stagnation_limit = 80

            while calls < calls_init + max_calls and time_left() > max_time:
                # Generate population
                offspring = []
                for i in range(lam):
                    if calls >= calls_init + max_calls or time_left() < max_time:
                        break
                    z = rng.standard_normal(dim)
                    y = x_init + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls += 1
                    offspring.append((y, fy))

                if len(offspring) < mu:
                    break

                offspring.sort(key=lambda t: t[1])

                # Update local best
                if offspring[0][1] < local_best_f:
                    local_best_f = offspring[0][1]
                    local_best_x = offspring[0][0].copy()
                    stagnant = 0
                else:
                    stagnant += 1

                # Update global best
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()

                # Mean of selected
                x_new = sum(weights[i] * offspring[i][0] for i in range(mu))

                # Delta
                diff = (x_new - x_init) / (sigma + 1e-16)

                # Evolution path for sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff / dim) * diff

                # Evolution path for covariance
                pc = (1 - c_c) * pc + c_c * diff

                # Covariance matrix update
                H = np.zeros((dim, dim))
                for i in range(mu):
                    diff_i = (offspring[i][0] - x_init) / (sigma + 1e-16)
                    H += weights[i] * np.outer(diff_i, diff_i)

                C = (1 + c_1 + c_mu) * C - c_1 * np.outer(pc, pc) - c_mu * H

                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                D = np.sqrt(eigvals)
                B = eigvecs

                # Step size adaptation
                sigma *= np.exp(c_sigma / chi * (np.linalg.norm(ps) / chi - 1))
                sigma = max(sigma, 1e-6)

                x_init = x_new

                # Check for stagnation - restart if no progress
                if stagnant >= stagnation_limit:
                    # Restart from local best with perturbation
                    noise = rng.normal(0, 0.5, dim)
                    x_init = np.clip(local_best_x + noise, -5, 5)
                    sigma = 1.0
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)
                    C = np.eye(dim)
                    B = np.eye(dim)
                    D = np.ones(dim)
                    stagnant = 0

            # Update global best after CMA-ES
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

            return local_best_f, local_best_x

        # Run CMA-ES for the first 300 calls
        remaining_calls = budget - calls
        cmaes_calls = min(300, remaining_calls)
        if cmaes_calls > 10 and time_left() > 20:
            fx, x = run_cmaes(x, fx, calls, cmaes_calls, 20)

        # Phase 3: Nelder-Mead local polish
        if calls < budget and time_left() > 10:
            # Build simplex around current best
            simplex = [x.copy()]
            step = 0.2
            for d in range(dim):
                pt = x.copy()
                pt[d] += step
                pt = np.clip(pt, -5, 5)
                simplex.append(pt)

            fvals = [best_f]
            for pt in simplex[1:]:
                if calls >= budget or time_left() < 5:
                    break
                fv = func(pt)
                calls += 1
                fvals.append(fv)
                if fv < best_f:
                    best_f = fv
                    best_x = pt.copy()

            # Nelder-Mead iterations
            for _ in range(100):
                if calls >= budget or time_left() < 3:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                if calls >= budget or time_left() < 3:
                    break
                fr = func(xr)
                calls += 1

                if fr < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    if calls >= budget or time_left() < 3:
                        break
                    fe = func(xe)
                    calls += 1
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
                    # Contraction
                    rho = 0.5
                    xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    if calls >= budget or time_left() < 3:
                        break
                    fc = func(xc)
                    calls += 1
                    if fc < max(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if calls >= budget or time_left() < 3:
                                break
                            fi = func(simplex[i])
                            calls += 1
                            fvals[i] = fi

                # Update best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
