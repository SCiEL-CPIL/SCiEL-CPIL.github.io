"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with adaptive sigma and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cmaes(self, func, x0, sigma0, max_evals, rng, deadline, best_x, best_f):
        dim = self.dim
        x = x0.copy()
        f = func(x)
        best_x = x.copy()
        best_f = f
        evals = 1
        sigma = sigma0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cc = (4 + dim / 3.0) / ((dim + 4) * (dim / 3.0 + 4))
        cs = (2 + 2 / (dim + 1.8)) / (dim + cs + 3)
        damps = 1 + 2 * max(0, np.sqrt((dim - 1) / (dim + 3)) - 1) + cs
        cmu = min(0.5, (2 + 2.0 / (dim + 1.8)) / ((dim + 2) * (dim + 2 + 2.0 / (dim + 1.8))))
        lam = max(4, int(np.ceil(4 + 3 * np.log(dim))))
        weights = np.log(np.arange(1, lam + 1))
        weights /= np.sum(weights)
        mu = int(np.floor(lam / 2))
        mueff = np.sum(weights[:mu]) ** 2
        mueff = min(mueff, 0.5 * lam)
        nu = 2 + 2.0 / (dim + 1.8)

        def _update_C(C, C_inv, C_sqrt, C_inv_sqrt, w, y):
            # Rank-mu update
            C_new = C + cmu * np.sum([w[i] * np.outer(y[i], y[i]) for i in range(mu)], axis=0)
            # Rank-one update
            C_new = (1 + cs * (1 - np.exp(-damps)) - 1) * C_new + cs * np.outer(ps, ps)
            # Actually let's do it properly
            C_new = C + cmu * np.sum([w[i] * np.outer(y[i], y[i]) for i in range(mu)], axis=0)
            C_new = C_new + cs * (1 - np.exp(-damps)) * np.outer(ps, ps)
            # Wait, the standard CMA-ES update is:
            # C = (1 - cs) * C + cs * ps * ps^T + cmu * sum(w_i * y_i * y_i^T)
            # But we need to be careful with the scaling
            C_new = (1 - cs) * C + cs * np.outer(ps, ps) + cmu * np.sum(
                [w[i] * np.outer(y[i], y[i]) for i in range(mu)], axis=0
            )
            # Compute new C_inv, C_sqrt, C_inv_sqrt
            # Use SVD for stability
            U, S, Vt = np.linalg.svd(C_new)
            C_sqrt = U @ np.diag(np.sqrt(S)) @ Vt
            C_inv = Vt.T @ np.diag(1.0 / S) @ U.T
            C_inv_sqrt = Vt.T @ np.diag(1.0 / np.sqrt(S)) @ U.T
            return C_new, C_inv, C_sqrt, C_inv_sqrt

        while evals < max_evals:
            if time.time() > deadline:
                break
            # Generate offspring
            ys = np.zeros((lam, dim))
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = C_sqrt @ z
                ys[i] = y
            # Sort by fitness
            fx = np.array([func(x + sigma * y) for y in ys])
            evals += lam
            order = np.argsort(fx)
            ys_sorted = ys[order]
            fx_sorted = fx[order]
            # Update best
            if fx_sorted[0] < best_f:
                best_f = fx_sorted[0]
                best_x = (x + sigma * ys_sorted[0]).copy()
            # Update mean
            x_new = x + sigma * np.sum(weights[:mu] * ys_sorted[:mu], axis=0)
            # Update path
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ (x_new - x)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (evals))) < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - cs) ** evals)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * C_inv_sqrt @ (x_new - x)
            # Update covariance
            if hsig:
                ps_norm = np.linalg.norm(ps)
                if ps_norm > 0:
                    ps = ps / ps_norm * np.sqrt(1 - (1 - cs) ** (evals))
            C, C_inv, C_sqrt, C_inv_sqrt = _update_C(C, C_inv, C_sqrt, C_inv_sqrt, weights, ys_sorted)
            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (evals)) - np.sqrt(dim)))
            x = x_new
            # Clip to box
            x = np.clip(x, -5, 5)
            # Evaluate new mean
            fx = func(x)
            evals += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            # Restart if sigma is too small or too large
            if sigma < 1e-10 or sigma > 1e10:
                break
        return best_x, best_f, evals

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        remaining = self.budget
        # Multiple restarts with different initial points
        n_restarts = 3
        evals_per_restart = max(50, (self.budget - n_restarts) // n_restarts)
        for r in range(n_restarts):
            if remaining <= 0:
                break
            if time.time() > deadline:
                break
            if r == 0:
                x0 = rng.uniform(-5, 5, self.dim)
                sigma0 = 2.0
            else:
                # Start near the best found so far, with some perturbation
                x0 = best_x + rng.standard_normal(self.dim) * 1.0
                x0 = np.clip(x0, -5, 5)
                sigma0 = 1.0
            max_evals = min(evals_per_restart, remaining)
            bx, bf, used = self._cmaes(func, x0, sigma0, max_evals, rng, deadline, best_x, best_f)
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
            remaining -= used
            if time.time() > deadline:
                break
        if best_x is None:
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            best_x = x
            best_f = fx
        return best_x, best_f
# EVOLVE-BLOCK-END
