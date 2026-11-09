"""Hybrid CMA-ES + local refinement for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with adaptive restarts and local Nelder-Mead/coordinate refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 290.0  # Safety margin before 330s kill
        n = self.dim
        calls_made = 0
        best_x = np.zeros(n)
        best_f = np.inf

        def check_time():
            return (time.time() - start_time) < time_limit

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if not check_time():
                return np.inf
            calls_made += 1
            try:
                fx = func(x)
            except Exception:
                return np.inf
            fx = float(fx)
            if not np.isfinite(fx):
                fx = np.inf
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float).copy()
            return fx

        def nelder_mead(x_start, f_start, max_steps=40, initial_delta=0.5):
            nonlocal calls_made, best_x, best_f
            dim = self.dim
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            simplex = [x_start.copy()]
            for i in range(dim):
                xi = x_start.copy()
                xi[i] += initial_delta
                xi = np.clip(xi, -5, 5)
                if np.allclose(xi, x_start):
                    xi[i] = x_start[i] + initial_delta * 0.1
                    xi = np.clip(xi, -5, 5)
                simplex.append(xi)

            simplex_f = [f_start]
            for i in range(1, dim + 1):
                if not check_time() or calls_made >= self.budget:
                    return
                fx = safe_func(simplex[i])
                simplex_f.append(fx)

            for step in range(max_steps):
                if not check_time() or calls_made >= self.budget:
                    break

                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = safe_func(xr)
                if not check_time() or calls_made >= self.budget:
                    break

                if fxr < simplex_f[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = safe_func(xe)
                    if not check_time() or calls_made >= self.budget:
                        break
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_f[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fxr
                elif fxr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = safe_func(xc)
                    if not check_time() or calls_made >= self.budget:
                        break
                    if fxc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if not check_time() or calls_made >= self.budget:
                                return
                            simplex_f[i] = safe_func(simplex[i])

                if np.max(simplex_f) - np.min(simplex_f) < 1e-14:
                    break

        def coo_local_search(x0, f0, max_iter=30):
            """Coordinate descent local search."""
            nonlocal calls_made, best_x, best_f
            x = x0.copy()
            f = f0
            step = 0.3
            for it in range(max_iter):
                if not check_time() or calls_made >= self.budget:
                    break
                improved = False
                for i in range(self.dim):
                    if not check_time() or calls_made >= self.budget:
                        break
                    for direction in [1, -1]:
                        x_trial = x.copy()
                        x_trial[i] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = safe_func(x_trial)
                        if f_trial < f:
                            x = x_trial
                            f = f_trial
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return x, f

        def cma_es(x0, sigma0, max_calls, max_gen=150):
            """
            Full CMA-ES with covariance matrix adaptation.
            Returns (best_x, best_f, calls_used, stagnation_flag).
            """
            nonlocal calls_made, best_x, best_f
            dim = n
            lambda_ = min(4 + int(3 * np.log(dim)), 15)
            mu = lambda_ // 2

            w = np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = (2 - cc + 2 * mueff / (dim - 1)) / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

            xmean = np.array(x0, dtype=float)
            sigma = sigma0
            C = np.eye(dim)
            C_eigval = np.ones(dim)
            C_eigvec = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))
            hs = (1.0 + (1.5 + 1.0 / dim) * cc) / (1.0 - cc)

            best_x_local = xmean.copy()
            best_f_local = np.inf
            calls_used = 0
            no_improve = 0
            max_no_improve = 30

            gen = 0
            while calls_used < max_calls and gen < max_gen and check_time() and calls_made < self.budget:
                gen += 1
                # Generate population
                offspring = []
                for i in range(lambda_):
                    if calls_made + calls_used >= self.budget or not check_time():
                        break
                    z = np.random.randn(dim)
                    # Transform z using eigendecomposition
                    z_trans = C_eigvec @ (np.sqrt(C_eigval) * z)
                    x_trial = xmean + sigma * z_trans
                    x_trial = np.clip(x_trial, -5, 5)
                    offspring.append(x_trial)

                if len(offspring) < mu:
                    break

                # Evaluate
                fvals = []
                for x_trial in offspring:
                    if calls_made + calls_used >= self.budget or not check_time():
                        break
                    fx = safe_func(x_trial)
                    calls_used += 1
                    fvals.append(fx)
                    if fx < best_f_local:
                        best_f_local = fx
                        best_x_local = x_trial.copy()
                        no_improve = 0
                    else:
                        no_improve += 1

                if len(fvals) < mu:
                    break

                # Sort
                idx = np.argsort(fvals)
                selected = [offspring[idx[i]] for i in range(mu)]
                selected_f = [fvals[idx[i]] for i in range(mu)]

                # Update mean
                xnew = np.sum([w[i] * selected[i] for i in range(mu)], axis=0)

                # Update covariance matrix
                diff = np.array([selected[i] - xmean for i in range(mu)])
                delta_x = (xnew - xmean) / sigma
                pc = (1 - cc) * pc + cc * delta_x
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * pc / (np.linalg.norm(pc) + 1e-300)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (gen + 1))) / chi_n < (1.4 + 2 / (dim + 1)) * hs

                # Rank-one update
                C = (1 + c1 * (2 - c1) + (hsig - 1) * c1 * cc) * C + c1 * np.outer(pc, pc)
                # Rank-mu update
                C += cmu * np.sum([w[i] * np.outer(diff[i], diff[i]) for i in range(mu)], axis=0)

                # Ensure positive definiteness
                C = (C + C.T) / 2
                try:
                    eigval, eigvec = np.linalg.eigh(C)
                    eigval = np.maximum(eigval, 1e-10)
                    C_eigval = eigval
                    C_eigvec = eigvec
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_eigval = np.ones(dim)
                    C_eigvec = np.eye(dim)

                # Update sigma
                sigma *= np.exp(cs / (dim + 3) * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 50)

                xmean = xnew

                # Check for stagnation
                if no_improve > max_no_improve:
                    break

            return best_x_local, best_f_local, calls_used, no_improve > max_no_improve

        # Phase 1: Initial exploration with diverse starting points
        rng = np.random.default_rng(42)
        starts = [
            np.zeros(n),
            rng.uniform(-4, 4, size=n),
            rng.uniform(-4, 4, size=n),
            rng.uniform(-3, 3, size=n),
            rng.uniform(-2, 2, size=n),
            rng.uniform(-4, 4, size=n),
            rng.uniform(-4, 4, size=n),
        ]

        # Evaluate starts
        for x in starts:
            if not check_time() or calls_made >= self.budget:
                break
            safe_func(x)

        # Phase 2: CMA-ES from best starting point
        remaining = self.budget - calls_made
        if remaining > 50 and check_time():
            cma_x, cma_f, cma_calls, cma_stag = cma_es(best_x.copy(), 2.0, min(remaining, 300))
            if cma_f < best_f:
                best_f = cma_f
                best_x = cma_x.copy()

        # Phase 3: Local refinement with Nelder-Mead
        remaining = self.budget - calls_made
        if remaining > 20 and check_time() and best_f < np.inf:
            nelder_mead(best_x.copy(), best_f, max_steps=min(30, remaining // 2), initial_delta=0.3)

        # Phase 4: Coordinate descent polish
        remaining = self.budget - calls_made
        if remaining > 10 and check_time() and best_f < np.inf:
            coo_local_search(best_x.copy(), best_f, max_iter=min(20, remaining // 2))

        # Phase 5: If budget remains, try another CMA-ES from perturbed best
        remaining = self.budget - calls_made
        if remaining > 40 and check_time():
            for i in range(2):
                if not check_time() or calls_made >= self.budget:
                    break
                x_start = best_x + rng.normal(0, 0.3, size=n)
                x_start = np.clip(x_start, -5, 5)
                cma_x, cma_f, cma_calls, cma_stag = cma_es(x_start, 1.0, min(remaining, 100))
                if cma_f < best_f:
                    best_f = cma_f
                    best_x = cma_x.copy()
                remaining = self.budget - calls_made
                if remaining <= 20:
                    break

        # Phase 6: Final fine polish
        remaining = self.budget - calls_made
        if remaining > 10 and check_time() and best_f < np.inf:
            nelder_mead(best_x.copy(), best_f, max_steps=min(20, remaining // 2), initial_delta=0.1)
            coo_local_search(best_x.copy(), best_f, max_iter=min(10, remaining // 2))

        return best_x, best_f

# EVOLVE-BLOCK-END
