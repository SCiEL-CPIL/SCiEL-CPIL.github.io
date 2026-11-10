"""Surrogate-assisted multi-objective Bayesian optimizer with EHVI acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import erf
from scipy.stats import norm
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        dim, n_obj, budget = self.dim, self.n_obj, self.budget
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        ref = np.array([3000.0, 0.06], dtype=float)
        deadline = time.time() + 280.0
        rng = np.random.default_rng(0)

        def time_ok():
            return time.time() < deadline

        def norm_cdf(x):
            return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

        def norm_pdf(x):
            return norm.pdf(x)

        def hypervolume_2d(points, refpt):
            pts = np.array([p for p in points if all(p[i] < refpt[i] for i in range(n_obj))], dtype=float)
            if len(pts) == 0:
                return 0.0
            pts = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            prev = refpt[1]
            for p in pts:
                if p[1] < prev:
                    hv += (refpt[0] - p[0]) * (prev - p[1])
                    prev = p[1]
                if p[0] >= refpt[0]:
                    break
            return hv

        # Sobol quasi-random initialization
        n_init = 32
        init = np.empty((n_init, dim), dtype=float)
        for i in range(n_init):
            frac = (i + 0.5) / n_init
            for d in range(dim):
                # van der Corput in base 2
                v, f = 0.0, 1.0
                x = frac
                while x > 0:
                    x *= 2.0
                    digit = int(x) % 2
                    f *= 0.5
                    v += digit * f
                init[i, d] = v
        X = lo + span * init
        F = np.array([np.asarray(problem(x), dtype=float) for x in X], dtype=float)
        n_eval = n_init

        # Fit GP: normalize inputs to [0,1]; per-output standardization
        def fit_gp(Xs, Fs):
            Xn = (Xs - lo) / span
            mu = Fs.mean(axis=0)
            sd = Fs.std(axis=0)
            sd[sd < 1e-12] = 1.0
            Yn = (Fs - mu) / sd
            n = len(Xn)
            L = 1.0
            beta = 1.0
            sigma_n = 1e-2
            for _ in range(60):
                sq = np.sum((Xn[:, :, None] - Xn[:, None, :]) ** 2, axis=2)
                K = (L ** 2) * np.exp(-0.5 * sq) + sigma_n ** 2 * np.eye(n)
                K[np.diag_indices(n)] += 1e-8
                try:
                    Lmat = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    L += 0.1
                    continue
                alpha = np.linalg.solve(Lmat.T, np.linalg.solve(Lmat, Yn))
                trK = np.trace(np.linalg.inv(K))
                logdet = 2.0 * np.sum(np.log(np.diag(Lmat)))
                ell = 0.5 * (n * np.log(2 * np.pi) + logdet + np.sum(Yn * alpha))
                dL = 0.5 * (n / (L ** 2) - np.sum(sq * alpha[:, :, None] * alpha[None, :, :]))
                dB = 0.5 * (n / beta - np.sum(alpha ** 2))
                dS = 0.5 * (n / (sigma_n ** 2) - np.sum(np.diag(np.linalg.inv(K))))
                grad = np.array([dL, dB, dS])
                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-4:
                    break
                step = 0.1 / max(1.0, gnorm)
                L = max(0.05, L - step * dL)
                beta = max(1e-2, beta - step * dB)
                sigma_n = max(1e-4, sigma_n - step * dS)
            sq = np.sum((Xn[:, :, None] - Xn[:, None, :]) ** 2, axis=2)
            K = (L ** 2) * np.exp(-0.5 * sq) + sigma_n ** 2 * np.eye(n)
            K[np.diag_indices(n)] += 1e-8
            Lmat = np.linalg.cholesky(K)
            alpha = np.linalg.solve(Lmat.T, np.linalg.solve(Lmat, Yn))
            return Xn, mu, sd, L, beta, sigma_n, Lmat, alpha

        def gp_predict(Xn, gp):
            Xn0, mu, sd, L, beta, sigma_n, Lmat, alpha = gp
            sq = np.sum((Xn0[:, None, :] - Xn[None, :, :]) ** 2, axis=2)
            Ks = (L ** 2) * np.exp(-0.5 * sq)
            v = np.linalg.solve(Lmat, Ks.T)
            mean_n = np.sum(alpha[:, None, :] * v, axis=0)
            var_n = 1.0 / beta - np.sum(v ** 2, axis=0)
            var_n = np.maximum(var_n, 1e-8)
            mean = mean_n * sd[None, :] + mu[None, :]
            var = (var_n * sd ** 2)[None, :]
            return mean, var

        def ehvi_at(x):
            Xn = np.clip((x - lo) / span, 0.0, 1.0)
            mean, var = gp_predict(Xn, gp)
            mu = mean[0]
            s = np.sqrt(var[0])
            if s[0] < 1e-6 or s[1] < 1e-6:
                return 0.0
            z0 = (ref[0] - mu[0]) / s[0]
            z1 = (ref[1] - mu[1]) / s[1]
            I0 = s[0] * (norm_pdf(z0) + z0 * (1.0 - norm_cdf(z0)))
            I1 = s[1] * (norm_pdf(z1) + z1 * (1.0 - norm_cdf(z1)))
            I01 = 0.0
            for k in range(11):
                xk = -3.0 + k * 0.6
                wk = 1.0 if k not in (0, 10) else 0.5
                Xn2 = np.array([np.clip((x - lo) / span, 0.0, 1.0)])
                m2, v2 = gp_predict(Xn2, gp)
                mu2 = m2[0]
                s2 = np.sqrt(v2[0])
                z2 = (xk * s[1] + mu[1] - mu[0]) / s[0]
                I01 += wk * s[0] * (norm_pdf(z2) + z2 * (1.0 - norm_cdf(z2)))
            I01 *= 0.6
            return (ref[0] - mu[0]) * I1 + (ref[1] - mu[1]) * I0 - I01

        def ehvi_neg(x):
            return -ehvi_at(x)

        while n_eval < budget:
            if not time_ok():
                break
            gp = fit_gp(X, F)
            best_x, best_v = None, -np.inf
            n_start = 6
            for si in range(n_start):
                if not time_ok():
                    break
                if si == 0:
                    x0 = (lo + hi) / 2.0
                else:
                    x0 = lo + span * rng.random(dim)
                res = minimize(ehvi_neg, x0, method='L-BFGS-B',
                               bounds=list(zip(lo, hi)),
                               options={'maxiter': 40, 'ftol': 1e-8})
                v = -res.fun
                if v > best_v:
                    best_v = v
                    best_x = res.x
            if best_x is None:
                best_x = lo + span * rng.random(dim)
            best_x = np.clip(best_x, lo, hi)
            fc = np.asarray(problem(best_x), dtype=float)
            X = np.vstack([X, best_x])
            F = np.vstack([F, fc])
            n_eval += 1
        return X
# EVOLVE-BLOCK-END
