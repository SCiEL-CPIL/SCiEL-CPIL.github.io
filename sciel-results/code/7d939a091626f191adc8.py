"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global-Local Hybrid: RBF surrogate + Expected Improvement, then CMA-ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 315  # safety margin before 330s kill
        n = self.dim
        B = self.budget

        rng = np.random.default_rng(0)

        best_x = None
        best_f = np.inf
        calls = 0

        def update_best(x, f):
            nonlocal best_x, best_f, calls
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_ok():
            return time.time() - start_time < time_limit

        # ---- RBF surrogate machinery ----
        def make_kernel(X, Y, eps):
            # X: (m,n) training, Y: (q,n) query
            D = (X[:, None, :] - Y[None, :, :]) ** 2
            D = D.sum(-1)
            return np.exp(-eps * D)

        def rbf_predict(X, Y, centers, alpha, eps, sigma2):
            K = make_kernel(centers, Y, eps)
            return K @ alpha, K @ sigma2

        # ---- Phase 1: Global exploration (LHS + a few adaptive EI) ----
        n_lhs = 35
        X = []
        Y = []
        if B > n_lhs:
            perm = [rng.permutation(n_lhs) for _ in range(n)]
            for i in range(n_lhs):
                x = np.array([(perm[j][i] + rng.random()) / n_lhs * 10.0 - 5.0 for j in range(n)])
                if calls >= B or not time_ok():
                    break
                fx = func(x)
                calls += 1
                X.append(x)
                Y.append(fx)
                update_best(x, fx)

        # ---- Phase 2: EI-driven global/local search ----
        # A small number of EI acquisitions to home in on the basin, then hand off.
        ei_rounds = 10
        eps = 0.35
        gamma = 1e-5
        for r in range(ei_rounds):
            if calls >= B or not time_ok():
                break
            m = len(X)
            if m < 3:
                break
            Xa = np.array(X)
            Ya = np.array(Y)
            # Fit RBF: solve (K + I*gamma) alpha = Y
            K = make_kernel(Xa, Xa, eps)
            K = K + gamma * np.eye(m)
            try:
                alpha = np.linalg.solve(K, Ya)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K, Ya, rcond=None)[0]
            # Fit noise variance (simple: estimate from residuals)
            pred = make_kernel(Xa, Xa, eps) @ alpha
            resid = Ya - pred
            sigma2 = np.clip(np.mean(resid**2) + 1e-8, 1e-8, 1e2)

            # Candidate set: random points + points near current best
            cand = []
            n_rand = 60
            for _ in range(n_rand):
                cand.append(rng.uniform(-5, 5, size=n))
            # local candidates around best
            for _ in range(30):
                cand.append(best_x + rng.normal(0, 0.6, size=n))
            cand = np.array(cand)
            cand = np.clip(cand, -5, 5)

            mu, var = rbf_predict(Xa, cand, Xa, alpha, eps, sigma2)
            var = np.maximum(var, 1e-10)
            sd = np.sqrt(var)
            # Expected improvement (minimization)
            imp = best_f - mu
            z = imp / sd
            from math import erf, sqrt
            # Phi(z) via erf
            Phi = 0.5 * (1.0 + np.vectorize(lambda t: erf(t / sqrt(2.0)))(z))
            pdf = np.vectorize(lambda t: np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi))(z)
            EI = imp * Phi + sd * pdf
            idx = int(np.argmax(EI))
            xnew = cand[idx]
            fx = func(xnew)
            calls += 1
            X.append(xnew)
            Y.append(fx)
            update_best(xnew, fx)

        # ---- Phase 3: CMA-ES local polish at the best point ----
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls += 1

        if calls < B and time_ok():
            x = best_x.copy()
            sigma = 0.6
            lam = 15
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            C = np.eye(n)
            D, B_eig = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-10))
            pc = np.zeros(n)
            ps = np.zeros(n)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

            def reflect(y):
                y = y.copy()
                for i in range(n):
                    if y[i] < -5:
                        y[i] = -5 + (-5 - y[i]) * 0.5
                    elif y[i] > 5:
                        y[i] = 5 - (y[i] - 5) * 0.5
                return np.clip(y, -5, 5)

            gen = 0
            while calls < B and time_ok():
                offspring = []
                fvals = []
                for i in range(lam):
                    if calls >= B or not time_ok():
                        break
                    z = rng.standard_normal(n)
                    y = reflect(x + sigma * (B_eig @ (D * z)))
                    offspring.append(y)
                    fy = func(y)
                    fvals.append(fy)
                    calls += 1
                    update_best(y, fy)
                if len(fvals) < lam:
                    break
                order = np.argsort(fvals)
                offspring = [offspring[i] for i in order]
                fvals = [fvals[i] for i in order]
                x_new = np.sum([weights[i] * offspring[i] for i in range(mu)], axis=0)
                y_inv = np.linalg.solve(C, (x_new - x) / sigma)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_inv
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1)))
                        < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1)))
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_inv
                delta_hsig = (1 - hsig) * cc * (2 - cc)
                rank_one = np.outer(pc, pc)
                rank_mu_terms = []
                for i in range(mu):
                    zi = np.linalg.solve(C, (offspring[i] - x) / sigma)
                    rank_mu_terms.append(weights[i] * np.outer(zi, zi))
                rank_mu = np.sum(rank_mu_terms, axis=0)
                C = (1 + c1 * delta_hsig + cmu - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
                C = (C + C.T) / 2
                ev = np.linalg.eigvalsh(C)
                if np.min(ev) < 1e-10:
                    C += (1e-10 - np.min(ev)) * np.eye(n)
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(2 * n) - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)
                x = x_new
                gen += 1
                D, B_eig = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-10))

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
