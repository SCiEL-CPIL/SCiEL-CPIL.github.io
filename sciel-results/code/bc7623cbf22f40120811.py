"""Bayesian-optimization-style optimizer with GP surrogate and EI acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """GP-based Bayesian optimizer with Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s limit

        # State
        X = np.zeros((0, dim))
        Y = np.zeros(0)
        best_x = None
        best_f = np.inf

        def record(x, f):
            nonlocal X, Y, best_x, best_f
            X = np.vstack([X, x.reshape(1, -1)]) if X.size > 0 else x.reshape(1, -1)
            Y = np.append(Y, f)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # GP kernel: Matern 5/2 with ARD
        def matern52(X1, X2, lengthscales, variance):
            D = np.zeros((X1.shape[0], X2.shape[0]))
            for i in range(dim):
                D += ((X1[:, i:i+1] - X2[:, i:i+1].T) / lengthscales[i]) ** 2
            D = np.sqrt(D)
            return variance * (1 + np.sqrt(5) * D + 5 / 3 * D**2) * np.exp(-np.sqrt(5) * D)

        def log_likelihood(theta):
            variance = np.exp(theta[0])
            lengthscales = np.exp(theta[1:1+dim])
            noise = np.exp(theta[1+dim])
            K = matern52(X, X, lengthscales, variance) + noise * np.eye(X.shape[0])
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return -1e10
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
            ll = -0.5 * Y @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * X.shape[0] * np.log(2 * np.pi)
            return ll

        def predict(Xs, theta):
            variance = np.exp(theta[0])
            lengthscales = np.exp(theta[1:1+dim])
            noise = np.exp(theta[1+dim])
            K = matern52(X, X, lengthscales, variance) + noise * np.eye(X.shape[0])
            Ks = matern52(X, Xs, lengthscales, variance)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(K + 1e-6 * np.eye(K.shape[0]))
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
            mu = Ks.T @ alpha
            v = np.linalg.solve(L, Ks.T)
            var = variance - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-12)
            return mu, np.sqrt(var)

        # Initial points: LHS + center
        rng = np.random.default_rng(42)
        n_init = 12
        # Latin hypercube
        X_init = np.zeros((n_init, dim))
        for i in range(dim):
            X_init[:, i] = rng.permutation(n_init) / n_init
        X_init = X_init * 10.0 - 5.0
        # Add center
        X_init = np.vstack([X_init, np.zeros(dim)])
        
        for x_init in X_init:
            if time.time() > deadline:
                break
            f = func(x_init)
            record(x_init, f)

        # Fit initial GP
        theta = None
        if X.shape[0] >= 2:
            theta0 = np.zeros(1 + dim + 1)
            theta0[0] = np.log(np.var(Y) + 1e-8)
            theta0[1:1+dim] = np.log(1.0)  # lengthscales = 1
            theta0[1+dim] = np.log(1e-4)  # noise
            try:
                res = minimize(log_likelihood, theta0, method='L-BFGS-B',
                               bounds=[(-10, 10)] * (1 + dim + 1),
                               options={'maxiter': 50})
                if res.fun > -1e9:
                    theta = res.x
            except:
                theta = theta0
        if theta is None:
            theta = np.zeros(1 + dim + 1)
            theta[0] = np.log(np.var(Y) + 1e-8)
            theta[1+dim] = np.log(1e-4)

        # Main loop
        calls_made = X.shape[0]
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Re-fit GP periodically
            if calls_made % 10 == 0 or calls_made == X.shape[0]:
                if X.shape[0] >= 2:
                    theta0 = theta if theta is not None else np.zeros(1 + dim + 1)
                    theta0[0] = np.log(np.var(Y) + 1e-8)
                    theta0[1+dim] = np.log(1e-4)
                    try:
                        res = minimize(log_likelihood, theta0, method='L-BFGS-B',
                                       bounds=[(-10, 10)] * (1 + dim + 1),
                                       options={'maxiter': 30})
                        if res.fun > -1e9:
                            theta = res.x
                    except:
                        pass

            # Generate candidate points for EI
            n_candidates = 200
            # Mix: random + around best
            cand = np.zeros((n_candidates, dim))
            n_rand = n_candidates // 2
            cand[:n_rand] = rng.uniform(-5, 5, (n_rand, dim))
            # Around current best
            if best_x is not None:
                spread = max(0.5, 2.0 * (1 - calls_made / budget))
                cand[n_rand:] = best_x + rng.normal(0, spread, (n_candidates - n_rand, dim))
                cand[n_rand:] = np.clip(cand[n_rand:], -5, 5)

            # Predict and compute EI
            mu, sigma = predict(cand, theta)
            if best_f == np.inf:
                ei = np.zeros(n_candidates)
            else:
                Z = (best_f - mu) / (sigma + 1e-12)
                ei = (best_f - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)

            # Select best EI point
            idx = np.argmax(ei)
            x_new = cand[idx]
            
            f_new = func(x_new)
            record(x_new, f_new)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
