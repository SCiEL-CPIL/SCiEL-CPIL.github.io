# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm


class Optimizer:
    """Surrogate (RBF) EI for global mapping, then (1+1)-ES local refinement.

    Minimizes func on [-5, 5]^dim. Black-box continuous, BBOB f11 (Discus).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # deadline safety margin: stop calling func well before 300s wall-clock
        self.deadline = time.time() + 280.0

    def __call__(self, func):
        rng = np.random.default_rng(0)
        n = self.dim
        B = self.budget
        calls = 0

        def call(x):
            nonlocal calls
            x = np.asarray(x, dtype=float).reshape(n)
            x = np.clip(x, -5.0, 5.0)
            val = float(func(x))
            calls += 1
            return x, val

        # ---- Phase 1: global surrogate mapping with EI ----
        # RBF surrogate parameters
        length_scale = 1.6
        noise = 1e-6

        X = np.empty((0, n))
        Y = np.empty((0,))

        def rbf_predict(Xs, Xq):
            # returns mean, var (var via diagonal of K)
            if len(Xs) == 0:
                return np.zeros(len(Xq)), np.ones(len(Xq)) * 1e3
            D2 = ((Xs[:, None, :] - Xq[None, :, :]) ** 2).sum(-1)
            K = np.exp(-D2 / (2.0 * length_scale ** 2))
            Kyy = K + noise * np.eye(len(Xs))
            try:
                alpha = np.linalg.solve(Kyy, Y)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(Kyy, Y, rcond=None)[0]
            mean = K @ alpha
            var = 1.0 - np.sum(K * K.T @ alpha, axis=1)
            var = np.maximum(var, 1e-9)
            return mean, var

        def ei_point(Xs, Ys, Xq):
            # expected improvement for minimization, with a small exploration floor
            mu, var = rbf_predict(Xs, Xq)
            sigma = np.sqrt(var)
            best = Ys.min()
            # improvement = best - f(x); EI = E[max(best - f, 0)]
            z = (best - mu) / sigma
            ei = (best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            # add a small exploration floor to avoid over-exploitation early
            ei = ei + 0.02 * sigma
            return ei, mu

        # initial sample: center + a few random + a few on a coarse grid
        init = []
        init.append(np.zeros(n))
        for _ in range(6):
            init.append(rng.uniform(-5, 5, n))
        # a few axis-aligned probe points to start mapping the valley
        for _ in range(4):
            p = rng.uniform(-5, 5, n)
            init.append(p)

        phase1_budget = min(B, max(40, int(0.40 * B)))

        for _ in range(phase1_budget):
            if time.time() > self.deadline:
                break
            # pick next point by EI over a candidate set
            cand = []
            # random candidates
            for _ in range(80):
                cand.append(rng.uniform(-5, 5, n))
            # perturbation around current best
            if len(Y) > 0:
                bx = X[int(np.argmin(Y))]
                for _ in range(40):
                    cand.append(np.clip(bx + 1.0 * rng.standard_normal(n), -5, 5))
            # coarse grid-ish candidates
            for _ in range(20):
                p = rng.uniform(-5, 5, n)
                p[np.argmax(np.abs(p))] = rng.choice([-4, -2, 0, 2, 4])
                cand.append(p)

            cand = np.array(cand)
            if len(X) == 0:
                # first point: just center
                xq = np.zeros(n)
            else:
                ei, mu = ei_point(X, Y, cand)
                xq = cand[int(np.argmax(ei))]

            x, fx = call(xq)
            X = np.vstack([X, x]) if len(X) else x[None, :]
            Y = np.append(Y, fx)

            if time.time() > self.deadline:
                break

        # ---- Phase 2: local (1+1)-ES refinement from best point ----
        if len(Y) == 0:
            x, fx = call(np.zeros(n))
        else:
            bi = int(np.argmin(Y))
            x = X[bi].copy()
            fx = Y[bi]

        sigma = 1.0
        # adaptive step: track successes in a window
        window = []
        W = 15

        while calls < B:
            if time.time() > self.deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = call(y)
            if fy < fx:
                x, fx = y, fy
                window.append(1)
                sigma *= 1.3
            else:
                window.append(0)
                sigma *= 0.85
            if len(window) > W:
                window.pop(0)
            # keep sigma in a sensible range
            sigma = float(np.clip(sigma, 1e-4, 5.0))

        return x, fx
# EVOLVE-BLOCK-END
