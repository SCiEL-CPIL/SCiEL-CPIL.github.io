"""Portfolio optimizer for BBOB f19 (Griewank-Rosenbrock)."""
import numpy as np
import time


class Optimizer:
    """Multi-strategy portfolio optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0

        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return x, fx

        def check_deadline():
            return time.time() > deadline

        def check_budget():
            return calls_made >= budget

        # Phase 1: Broad random sampling to find good starting regions
        rng = np.random.default_rng(42)
        n_initial = min(20, budget // 10)
        for i in range(n_initial):
            if check_budget() or check_deadline():
                break
            x0 = rng.uniform(-4, 4, dim)
            safe_call(x0)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = 0.0

        # Phase 2: Nelder-Mead simplex method from best point
        def nelder_mead(x0, f0, n_steps, rng_nm):
            nonlocal best_x, best_f, calls_made
            if check_budget() or check_deadline():
                return

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            simplex_size = 0.5

            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [f0]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += simplex_size
                xi = np.clip(xi, -5, 5)
                _, fi = safe_call(xi)
                simplex.append(xi)
                fvals.append(fi)
                if check_budget() or check_deadline():
                    return

            for step in range(n_steps):
                if check_budget() or check_deadline():
                    break

                # Sort simplex by fitness
                order = np.argsort(fvals)
                simplex = [simplex[i].copy() for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                _, fr = safe_call(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    _, fe = safe_call(xe)
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
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    _, fc = safe_call(xc)

                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            _, fi = safe_call(simplex[i])
                            fvals[i] = fi
                            if check_budget() or check_deadline():
                                return

        # Run Nelder-Mead from best point
        nm_budget = min(100, budget - calls_made)
        if nm_budget > 10:
            nelder_mead(best_x.copy(), best_f, nm_budget, np.random.default_rng(123))

        # Phase 3: CMA-ES from current best
        def cma_es(x0, sigma0, n_evals, rng_cma):
            nonlocal best_x, best_f, calls_made
            if check_budget() or check_deadline():
                return

            x = np.array(x0, dtype=np.float64)
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            cs = np.zeros(dim)

            mu = dim // 2 + 1
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            csigma = (1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cc) / (dim + 5)
            damps = 1 + 2 * max(0, np.sqrt(mu_eff / (dim + 1)) - 1) + cc

            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                B = np.eye(dim)

            for i in range(n_evals):
                if calls_made + mu > budget:
                    break
                if check_deadline():
                    break

                ys = []
                for k in range(mu):
                    z = rng_cma.standard_normal(dim)
                    y = x + sigma * (B @ z)
                    y = np.clip(y, -5, 5)
                    ys.append(y)

                fvals = []
                for y in ys:
                    _, fy = safe_call(y)
                    fvals.append(fy)
                    if check_deadline():
                        break

                if len(fvals) < mu:
                    break

                order = np.argsort(fvals)
                ys = [ys[i] for i in order]
                fvals = [fvals[i] for i in order]

                new_mean = sum(w[k] * ys[k] for k in range(mu))

                pc = (1 - cc) * pc + cc * (new_mean - x) / (sigma * np.linalg.norm(B))
                cs = (1 - csigma) * cs + np.sign(cc * (1 - cc)) * np.linalg.norm(pc)

                sigma *= np.exp(csigma / damps * (np.linalg.norm(pc) / np.sqrt(1 - (1 - cc) ** (2 * (i + 1))) - 1))

                delta_s = (1 + 2 * max(0, cs - 1)) * csigma
                outer = np.outer(pc, pc)
                C = (1 + c1 * delta_s - c1 - cmu) * C
                C += c1 * (outer - C)
                for k in range(mu):
                    yk = (ys[k] - x) / sigma
                    C += cmu * w[k] * np.outer(yk, yk)

                x = new_mean

                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    B = np.eye(dim)

        cma_budget = min(150, budget - calls_made)
        if cma_budget > 20:
            cma_es(best_x.copy(), 0.5, cma_budget, np.random.default_rng(456))

        # Phase 4: Local refinement with (1+1)-ES
        remaining = budget - calls_made
        if remaining > 10 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            rng2 = np.random.default_rng(789)

            for _ in range(remaining):
                if check_budget() or check_deadline():
                    break

                y = np.clip(x + sigma * rng2.standard_normal(dim), -5, 5)
                _, fy = safe_call(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.85

        return best_x, best_f
