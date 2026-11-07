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

        def _adaptive_rotated_search(x0, alloc, seed_off):
            nonlocal evaluated, best_x, best_f
            if not _ok():
                return

            r = np.random.default_rng(77 + seed_off)
            x = np.array(x0, dtype=float)
            fx = best_f

            # Build initial rotation from a small set of evaluations
            # to estimate the principal directions of the landscape
            # around the current best point.
            probe_pts = []
            probe_vals = []
            for _ in range(min(10, alloc)):
                if not _ok():
                    break
                z = r.standard_normal(dim)
                x_p = np.clip(x + 0.5 * z, -5.0, 5.0)
                f_p = _eval(x_p)
                probe_pts.append(x_p)
                probe_vals.append(f_p)

            if len(probe_pts) >= dim:
                # Center the probe points
                pts_arr = np.array(probe_pts)
                center = pts_arr.mean(axis=0)
                centered = pts_arr - center
                # Compute covariance
                cov = np.cov(centered.T)
                if cov.shape == (dim, dim):
                    try:
                        eigvals, eigvecs = np.linalg.eigh(cov)
                        # Sort by eigenvalue descending
                        order = np.argsort(eigvals)[::-1]
                        eigvecs = eigvecs[:, order]
                        # Use the top two principal directions for rotation
                        # Build rotation matrix with top 2 eigenvectors and complete with identity
                        R = np.eye(dim)
                        R[:, 0] = eigvecs[:, 0]
                        R[:, 1] = eigvecs[:, 1]
                        # Ensure orthonormality for remaining columns
                        for k in range(2, dim):
                            # Find a vector orthogonal to R[:, :k]
                            v = np.random.randn(dim)
                            for j in range(k):
                                v -= np.dot(v, R[:, j]) * R[:, j]
                            nv = np.linalg.norm(v)
                            if nv < 1e-10:
                                v = np.eye(dim)[k]
                                for j in range(k):
                                    v -= np.dot(v, R[:, j]) * R[:, j]
                                nv = np.linalg.norm(v)
                                if nv < 1e-10:
                                    continue
                                v /= nv
                            else:
                                v /= nv
                            R[:, k] = v
                    except np.linalg.LinAlgError:
                        R = np.eye(dim)
                else:
                    R = np.eye(dim)
            else:
                R = np.eye(dim)

            # Now perform adaptive coordinate descent in the rotated frame
            # Work in rotated coordinates: y = R^T @ x
            y = R.T @ x
            fy = fx

            sigma = 0.3
            step_size = 0.8
            no_improve = 0
            max_no_improve = 20

            while _ok() and no_improve < max_no_improve:
                if evaluated >= budget:
                    break
                improved = False
                for i in range(dim):
                    if not _ok():
                        break
                    step = 0.5
                    for _s in range(15):
                        if not _ok():
                            break
                        for sign in (1, -1):
                            y_try = y.copy()
                            y_try[i] += sign * step
                            x_try = R @ y_try
                            x_try = np.clip(x_try, -5.0, 5.0)
                            f_try = _eval(x_try)
                            if f_try < fy:
                                fy = f_try
                                y = y_try
                                x = x_try
                                step *= 1.3
                                improved = True
                                no_improve = 0
                                break
                        else:
                            break
                    if not _ok():
                        break
                if not improved:
                    no_improve += 1
                    # If stuck, try a larger random step in rotated space
                    if _ok():
                        z = r.standard_normal(dim)
                        y_try = y + sigma * z
                        x_try = R @ y_try
                        x_try = np.clip(x_try, -5.0, 5.0)
                        f_try = _eval(x_try)
                        if f_try < fy:
                            fy = f_try
                            y = y_try
                            x = x_try
                            sigma = min(sigma * 1.5, 5.0)
                            no_improve = 0
                        else:
                            sigma = max(sigma * 0.8, 1e-6)

        def _plus_one_es(x0, alloc, seed_off):
            nonlocal evaluated, best_x, best_f
            if not _ok():
                return

            r = np.random.default_rng(77 + seed_off)
            x = np.array(x0, dtype=float)
            fx = best_f
            sigma = 0.3
            step_size = 0.8

            while _ok():
                if evaluated >= budget:
                    break
                z = r.standard_normal(dim)
                x_try = np.clip(x + sigma * z, -5.0, 5.0)
                f_try = _eval(x_try)
                if f_try < fx:
                    fx = f_try
                    x = x_try
                    sigma = min(sigma * 1.2, 5.0)
                else:
                    sigma = max(sigma * step_size, 1e-6)

        # === Strategy ===
        # Phase 1: Broad exploration with CMA-ES (2 restarts to save budget)
        n_starts = 2
        alloc1 = budget // 3

        for i in range(n_starts):
            if not _ok():
                break
            start = rng.uniform(-4.0, 4.0, size=dim)
            _cma_es(start, alloc1, seed_off=i * 100)

        # Phase 2: Adaptive rotated coordinate search from best
        if best_x is not None and _ok():
            alloc2 = budget // 3
            _adaptive_rotated_search(best_x, alloc2, seed_off=0)

        # Phase 3: (1+1)-ES polish
        if best_x is not None and _ok():
            remaining = budget - evaluated
            _plus_one_es(best_x, remaining, seed_off=200)

        return best_x, best_f
# EVOLVE-BLOCK-END
