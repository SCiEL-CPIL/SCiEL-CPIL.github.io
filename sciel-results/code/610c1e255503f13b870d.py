"""Uncertainty inequality: tune Hermite-Gaussian coefficients to reduce the bound on C4."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    t0 = time.time()
    DEADLINE = 200.0          # hard wall-clock guard (evaluator kills at 210s)
    SAFETY = 18.0

    def _time_left():
        return DEADLINE - (time.time() - t0)

    # ---------- Hermite polynomials (physicists', He_4k) ----------
    H = {0: 1.0, 4: 16.0, 8: 256.0, 12: 4096.0}

    # ---------- evaluation of g, g' and the uncertainty bound ----------
    def _gval(x, c0, c4, c8, c12):
        e = np.exp(-0.5 * x * x)
        return (c0 * H[0] + c4 * H[4] * x**4 + c8 * H[8] * x**8
                + c12 * H[12] * x**12) * e

    def _gprim(x, c0, c4, c8, c12):
        e = np.exp(-0.5 * x * x)
        p = (c4 * H[4] * 4 * x**3 + c8 * H[8] * 8 * x**7
             + c12 * H[12] * 12 * x**11)
        return (p - x * (c0 * H[0] + c4 * H[4] * x**4 + c8 * H[8] * x**8
                         + c12 * H[12] * x**12)) * e

    def _c12(c0, c4, c8):
        # g(0)=0  =>  c0 + c12*H[12] = 0
        return -c0 / H[12]

    def _bound(c0, c4, c8, n=40000):
        c12 = _c12(c0, c4, c8)
        x = np.linspace(-12.0, 12.0, n)
        g = _gval(x, c0, c4, c8, c12)
        gp = _gprim(x, c0, c4, c8, c12)
        l1 = np.trapz(np.abs(g), x)
        l1p = np.trapz(np.abs(gp), x)
        if not (np.isfinite(l1) and np.isfinite(l1p)) or l1 <= 0.0:
            return np.inf
        return l1 * l1p

    def _obj(c):
        return _bound(c[0], c[1], c[2])

    # ---------- multi-start CMA-ES (self-contained, no external deps) ----------
    def cma_es(f, x0, sigma0, pop, max_iter, seed, time_fn):
        rng = np.random.default_rng(seed)
        x = np.array(x0, dtype=float)
        sigma = sigma0
        n = x.size
        lam = pop
        mu = max(1, lam // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w * w)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n * n))
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        best = x.copy()
        best_f = f(x)
        for it in range(max_iter):
            if time_fn() <= 0:
                break
            Y = []
            for i in range(lam):
                z = rng.standard_normal(n)
                Y.append(x + sigma * (B @ (D * z)))
            Y = np.array(Y)
            Fs = np.array([f(y) for y in Y])
            order = np.argsort(Fs)
            Ys = Y[order]
            Fs = Fs[order]
            ymean = np.sum(w[:, None] * Ys[:mu], axis=0)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (B @ (D * ymean)) / sigma
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) < (1.4 + 2 / (n + 1)) * chi_n:
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * ymean
            D = D * np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chi_n - 1))
            if np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (2 * (it + 1))) < (1.4 + 2 / (n + 1)) * chi_n:
                B = B + c_1 / D * np.outer(pc, pc) + c_mu / D * np.sum(w[:mu][:, None] * np.outer(Ys[:mu] - x, Ys[:mu] - x), axis=0)
            else:
                B = B + (c_1 + c_mu) / D * np.outer(pc, pc)
            B, D = _eigh_update(B, D)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            x = x + ymean
            if Fs[0] < best_f:
                best_f = Fs[0]
                best = Ys[0].copy()
        return best, best_f

    def _eigh_update(B, D):
        try:
            B = 0.5 * (B + B.T)
            evals, evecs = np.linalg.eigh(B)
            evals = np.clip(evals, 1e-12, None)
            return evecs @ (np.diag(1.0 / np.sqrt(evals)) @ (evecs.T @ B))
        except np.linalg.LinAlgError:
            return np.eye(B.shape[0]), np.ones(B.shape[0])

    # ---------- search ----------
    base = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_x = base.copy()
    best_f = _obj(best_x)

    def _try(x0, sigma0, pop, max_iter, seed):
        nonlocal best_x, best_f
        if _time_left() < SAFETY:
            return
        try:
            x, fv = cma_es(_obj, x0, sigma0, pop, max_iter, seed, _time_left)
        except Exception:
            return
        if np.all(np.isfinite(x)) and fv < best_f:
            best_f = fv
            best_x = x

    # diverse multi-start initializations
    rng = np.random.default_rng(0)
    starts = [base]
    for k in range(20):
        starts.append(base + rng.standard_normal(3) * 0.5)

    for i, s in enumerate(starts):
        if _time_left() < SAFETY + 5:
            break
        _try(s, 0.3, 12, 120, seed=1000 + i)

    # final local polish around the global best
    if _time_left() > SAFETY:
        _try(best_x, 0.05, 10, 100, seed=99999)

    return np.array([best_x[0], best_x[1], best_x[2]])
# EVOLVE-BLOCK-END
