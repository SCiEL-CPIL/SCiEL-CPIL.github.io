# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 285.0

        def _ok():
            return time.time() < deadline and evaluated < budget

        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        evaluated = 0

        def _eval(x):
            nonlocal evaluated, best_x, best_f
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def _nelder_mead(x0, alloc, seed_off):
            nonlocal evaluated, best_x, best_f
            if not _ok():
                return

            r = np.random.default_rng(42 + seed_off)
            x = np.array(x0, dtype=float)
            fx = _eval(x) if best_x is None or not np.allclose(x, best_x) else best_f

            # Build initial simplex with randomized, ridge-aware initialization
            # Use larger initial step to explore the landscape
            simplex = [x.copy()]
            for i in range(dim):
                xi = x.copy()
                # Randomize the step direction and magnitude for better ridge alignment
                step = 0.5 * r.standard_normal(dim)
                xi[i] += step[i]
                # Also add a component along a random direction to help with rotated ridges
                extra = 0.3 * r.standard_normal(dim)
                xi += extra
                xi = np.clip(xi, -5.0, 5.0)
                simplex.append(xi)

            # Evaluate simplex
            for i, xi in enumerate(simplex):
                if not _ok():
                    break
                if i == 0:
                    continue  # already evaluated
                simplex[i] = (xi, _eval(xi))
            simplex[0] = (simplex[0], fx)

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            while _ok():
                if evaluated >= budget:
                    break

                # Sort by function value
                simplex.sort(key=lambda t: t[1])

                # Check convergence
                fvals = np.array([s[1] for s in simplex])
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break

                xw = simplex[-1][0]
                xc = np.mean([s[0] for s in simplex[:-1]], axis=0)

                # Reflection
                xr = xc + alpha * (xc - xw)
                xr = np.clip(xr, -5.0, 5.0)
                fr = _eval(xr)

                if fr < simplex[0][1]:
                    # Expansion
                    xe = xc + gamma * (xc - xw)
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = _eval(xe)
                    if fe < fr:
                        simplex[-1] = (xe, fe)
                    else:
                        simplex[-1] = (xr, fr)
                elif fr < simplex[-2][1]:
                    simplex[-1] = (xr, fr)
                else:
                    # Contraction
                    if fr < simplex[-1][1]:
                        # Outside contraction
                        xc2 = xc + rho * (xc - xw)
                        xc2 = np.clip(xc2, -5.0, 5.0)
                        fc = _eval(xc2)
                        if fc <= fr:
                            simplex[-1] = (xc2, fc)
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i][0] = simplex[0][0] + sigma * (simplex[i][0] - simplex[0][0])
                                simplex[i][0] = np.clip(simplex[i][0], -5.0, 5.0)
                                simplex[i][1] = _eval(simplex[i][0])
                    else:
                        # Inside contraction
                        xc2 = xc - rho * (xc - xw)
                        xc2 = np.clip(xc2, -5.0, 5.0)
                        fc = _eval(xc2)
                        if fc < simplex[-1][1]:
                            simplex[-1] = (xc2, fc)
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i][0] = simplex[0][0] + sigma * (simplex[i][0] - simplex[0][0])
                                simplex[i][0] = np.clip(simplex[i][0], -5.0, 5.0)
                                simplex[i][1] = _eval(simplex[i][0])

        def _cma_es(x0, alloc, seed_off):
            nonlocal evaluated, best_x, best_f
            if not _ok():
                return

            r = np.random.default_rng(42 + seed_off)
            pop = max(4, int(np.ceil(4 + 3 * np.log(dim))))
            lam = pop
            sigma = 1.0
            mean = np.array(x0, dtype=float)
            C = np.eye(dim)

            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mue = 1.0 / np.sum(w ** 2)

            cs = (mue + 2) / (dim + mue + 5)
            cc = (4 + mue / dim) / (dim + 4 + 2 * mue / dim)
            c1 = 2 / ((dim + 1.3) ** 2 + mue)
            cmu = min(1.0 - c1, 2 * (mue - 2 + 2 / mue) / ((dim + 2) ** 2 + mue))

            ps = np.zeros(dim)
            pc = np.zeros(dim)

            # Initial population
            xs = []
            for _ in range(min(lam, alloc)):
                if not _ok():
                    break
                z = r.standard_normal(dim)
                x_new = np.clip(mean + sigma * z, -5.0, 5.0)
                f = _eval(x_new)
                xs.append((x_new, f))

            if len(xs) < mu:
                return

            # Main loop
            while _ok():
                if evaluated + lam > budget:
                    break

                xs = []
                L = None
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    pass

                for _ in range(lam):
                    if not _ok():
                        break
                    z = r.standard_normal(dim)
                    if L is not None:
                        x_new = np.clip(mean + sigma * (L @ z), -5.0, 5.0)
                    else:
                        x_new = np.clip(mean + sigma * z, -5.0, 5.0)
                    f = _eval(x_new)
                    xs.append((x_new, f))

                if len(xs) < mu:
                    break

                xs.sort(key=lambda t: t[1])

                old_mean = mean.copy()
                mean = np.zeros(dim)
                for i in range(mu):
                    mean += w[i] * xs[i][0]

                # Sigma update
                C_sinv = None
                try:
                    C_sinv = np.linalg.inv(np.linalg.cholesky(C))
                except np.linalg.LinAlgError:
                    pass

                d = (mean - old_mean) / (sigma * np.sqrt(mue))
                if C_sinv is not None:
                    d = C_sinv @ d
                ps = (1 - cs) * ps + cs * d
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2)
                chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
                if hsig < (1 + 2 / (dim + 1)) * chi:
                    sigma *= np.exp(cs / dim * (hsig - chi))
                sigma = max(sigma, 1e-10)

                # Covariance update
                dc = (mean - old_mean) / sigma
                pc = (1 - cc) * pc + cc * dc
                hcv = np.linalg.norm(pc) / np.sqrt(1 - (1 - cc) ** 2)
                hc = hcv / ((1 + 2 / (dim + 1)) * chi)

                rank1 = np.outer(pc, pc)
                if hc < (1 + 2 / (dim + 1)) * chi:
                    rank1 += (1 - (1 - cc) ** 2) * (cc / (1 - cc)) * np.eye(dim)

                rankmu = np.zeros((dim, dim))
                for i in range(mu):
                    y = (xs[i][0] - old_mean) / sigma
                    rankmu += w[i] * np.outer(y, y)

                C = (1 - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
                C = (C + C.T) / 2

                # Ensure PD
                try:
                    ev, evec = np.linalg.eigh(C)
                    ev = np.maximum(ev, 1e-12)
                    C = evec @ np.diag(ev) @ evec.T
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    sigma = max(sigma * 0.5, 0.01)

        def _plus_one_es(x0, alloc, seed_off):
            nonlocal evaluated, best_x, best_f
            if not _ok():
                return

            r = np.random.default_rng(77 + seed_off)
            x = np.array(x0, dtype=float)
            fx = best_f
            sigma = 0.2

            while _ok():
                if evaluated >= budget:
                    break
                z = r.standard_normal(dim)
                x_try = np.clip(x + sigma * z, -5.0, 5.0)
                f_try = _eval(x_try)
                if f_try < fx:
                    fx = f_try
                    x = x_try
                    sigma = min(sigma * 1.15, 3.0)
                else:
                    sigma = max(sigma * 0.95, 1e-6)

        # === Strategy ===
        # Phase 1: Multi-start Nelder-Mead with randomized simplex initialization
        # This aligns with rotated ridges faster than axis-aligned methods
        n_starts = 4
        alloc1 = budget // 5

        for i in range(n_starts):
            if not _ok():
                break
            # Randomized starting points, biased towards center
            start = rng.uniform(-3.0, 3.0, size=dim)
            _nelder_mead(start, alloc1, seed_off=i * 100)

        # Phase 2: CMA-ES refinement from best point found
        if best_x is not None and _ok():
            alloc2 = budget // 3
            _cma_es(best_x, alloc2, seed_off=500)

        # Phase 3: (1+1)-ES polish for final refinement
        if best_x is not None and _ok():
            remaining = budget - evaluated
            _plus_one_es(best_x, remaining, seed_off=200)

        return best_x, best_f
# EVOLVE-BLOCK-END
