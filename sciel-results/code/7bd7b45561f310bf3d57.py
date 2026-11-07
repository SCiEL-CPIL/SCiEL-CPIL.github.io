"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES + Rastrigin-snap local refinement for rotated Rastrigin f15.

    AOCC rewards early progress, so we seed with parallel samples, then run
    CMA-ES (handles the hidden rotation automatically), then polish with a
    Rastrigin-aware 1D scan that snaps coordinates toward 2*pi multiples.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(7)
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            call_count += 1
            fx = float(func(np.asarray(x, dtype=float)))
            if fx < best_f:
                best_f = fx
                best_x = np.asarray(x, dtype=float).copy()
            return fx

        # ---------- Rastrigin-aware 1D refinement ----------
        def refine_1d(x, axis, fx, n_scan=14):
            nonlocal call_count, best_x, best_f
            if call_count >= budget or time.time() > deadline:
                return x, fx
            x0 = float(x[axis])
            lo = max(-5.0, x0 - 1.2)
            hi = min(5.0, x0 + 1.2)
            cands = np.linspace(lo, hi, n_scan)
            k_lo = int(np.floor(lo / (2.0 * np.pi)))
            k_hi = int(np.ceil(hi / (2.0 * np.pi)))
            for k in range(k_lo, k_hi + 1):
                cands = np.append(cands, 2.0 * np.pi * k)
            cands = np.unique(np.clip(cands, -5.0, 5.0))
            cands = np.unique(np.append(cands, x0))
            best_c = x0
            best_cf = fx
            for c in cands:
                if call_count >= budget or time.time() > deadline:
                    break
                c = float(c)
                if abs(c - x0) < 1e-9:
                    continue
                y = x.copy()
                y[axis] = c
                fy = eval_func(y)
                if fy < best_cf:
                    best_cf = fy
                    best_c = c
            if best_c != x0:
                x[axis] = best_c
                fx = best_cf
            return x, fx

        def local_refine(x, fx, max_round=3):
            nonlocal call_count, best_x, best_f
            if call_count >= budget or time.time() > deadline:
                return x, fx
            x = x.copy()
            for _ in range(max_round):
                if call_count >= budget or time.time() < deadline:
                    break
                improved = False
                for axis in range(dim):
                    if call_count >= budget or time.time() < deadline:
                        break
                    x, fx = refine_1d(x, axis, fx, n_scan=14)
                    improved = True
                if not improved:
                    break
            return x, fx

        # ---------- Phase 0: parallel initial samples (fast early progress) ----------
        init_pts = []
        for _ in range(40):
            init_pts.append(rng.uniform(-5, 5, dim))
        init_pts.append(np.zeros(dim))
        for _ in range(8):
            p = np.clip(2.0 * np.pi * rng.integers(-1, 2, dim), -5, 5)
            init_pts.append(p)
        init_pts = np.array(init_pts)
        init_f = np.array([eval_func(p) for p in init_pts])
        if call_count >= budget:
            return best_x, best_f

        best_idx = int(np.argmin(init_f))
        x_mean = init_pts[best_idx].copy()

        # ---------- Phase 1: CMA-ES ----------
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2
        sigma = 1.5
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        c_sigma = (mu + 2) / (dim + mu + 5)
        d_sigma = 1 + 2 * max(0.0, np.sqrt((mu - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * ((mu + 2) / (dim + 2) - 1))
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        w = (np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)

        gen = 0
        no_improve = 0
        best_f_prev = best_f

        while call_count < budget and time.time() < deadline:
            gen += 1
            if gen % 5 == 1 or gen == 1:
                eigval, eigvec = np.linalg.eigh(C)
                eigval = np.clip(eigval, 1e-12, None)
                C_inv_sqrt = eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T

            pop = []
            pop_f = []
            for i in range(lam):
                if call_count >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = C_inv_sqrt @ z
                x = np.clip(x_mean + sigma * y, -5.0, 5.0)
                fx = eval_func(x)
                pop.append(x)
                pop_f.append(fx)
            if len(pop) < 2:
                break
            pop = np.array(pop)
            pop_f = np.array(pop_f)
            order = np.argsort(pop_f)
            pop = pop[order]
            pop_f = pop_f[order]

            x_w = pop[:mu]
            y_w = (x_w - x_mean) / sigma
            x_mean = np.clip(x_mean + sigma * np.sum(w[:, None] * y_w, axis=0), -5.0, 5.0)

            y_sigma = C_inv_sqrt @ (sigma * np.sum(w[:, None] * y_w, axis=0))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n
            sigma = float(np.clip(sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1)), 1e-4, 20.0))

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * np.sum(w[:, None] * y_w, axis=0)

            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - d_sigma * c_sigma * (1 - c_sigma)) * np.outer(p_sigma, p_sigma))
            for i in range(mu):
                C += c_mu * w[i] * np.outer(y_w[i], y_w[i])
            C = 0.5 * (C + C.T)

            if best_f < best_f_prev - 1e-8:
                no_improve = 0
                best_f_prev = best_f
            else:
                no_improve += 1

            if gen % 4 == 0 and call_count < budget and time.time() < deadline and best_x is not None:
                x_r, f_r = local_refine(best_x.copy(), best_f, max_round=2)
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r
                    x_mean = np.clip(0.5 * x_mean + 0.5 * x_r, -5.0, 5.0)

            if no_improve >= 12 and gen % 6 == 0 and call_count < budget:
                for _ in range(3):
                    if call_count >= budget or time.time() < deadline:
                        break
                    eval_func(rng.uniform(-5, 5, dim))
                sigma = max(sigma, 1.0)
                C = 0.5 * C + 0.5 * np.eye(dim)
                p_c = np.zeros(dim)
                p_sigma = np.zeros(dim)
                no_improve = 0

        # ---------- Phase 2: intensive local refinement ----------
        if best_x is not None and call_count < budget and time.time() < deadline:
            x_final = best_x.copy()
            f_final = best_f
            for _ in range(4):
                if call_count >= budget or time.time() < deadline:
                    break
                x_final, f_final = local_refine(x_final, f_final, max_round=3)
                if f_final >= best_f:
                    break
                best_f = f_final
                best_x = x_final

        return best_x, best_f
# EVOLVE-BLOCK-END
