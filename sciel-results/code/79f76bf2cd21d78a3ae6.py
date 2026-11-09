"""Optimizer for BBOB f21 (Gallagher 101 peaks) using CMA-ES."""
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with deadline guard for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, max_evals, deadline, rng):
        """Run CMA-ES. Returns (best_x, best_f, evals_used)."""
        n = self.dim
        # Population size: lambda = 4 + floor(3 * ln(n))
        lam = 4 + int(3 * np.log(n))
        # Weights: log weights
        w = np.log(lam / 2) - np.arange(1, lam + 1)
        w = np.exp(w)
        w /= w.sum()
        mu = int(lam / 2)
        w = w[:mu]
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cc = (4.0 + mueff / (n + 4.0)) / ((n + 1.3) + 4.0 + mueff / (n + 4.0))
        cs = (mueff + 2.0) / (n + mueff + 5.0)
        c1 = (2.0 / ((n + 1.8) ** 2 + mueff)) * (3.0 - n) / (n + 3.0)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
        sigma = sigma0
        x = np.array(x0, dtype=float)
        ps = np.zeros(n)
        hs = np.zeros(n)
        pc = np.zeros(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        D = np.ones(n)
        B = np.eye(n)
        # Cholesky of C
        L = np.linalg.cholesky(C)
        invsqrtC = np.linalg.inv(L.T)
        # Initial evaluation
        best_x = x.copy()
        best_f = func(x)
        evals = 1
        start_time = time.time()

        while evals < max_evals:
            # Check deadline
            if time.time() > deadline:
                break
            # Generate population
            X = np.zeros((lam, n))
            for i in range(lam):
                z = rng.standard_normal(n)
                X[i] = x + sigma * (invsqrtC @ z)
            # Evaluate
            F = np.zeros(lam)
            for i in range(lam):
                F[i] = func(np.clip(X[i], -5, 5))
                evals += 1
                if evals >= max_evals:
                    break
            if evals >= max_evals:
                break
            # Sort
            idx = np.argsort(F)[:mu]
            Xw = X[idx]
            Fw = F[idx]
            # Update best
            if Fw[0] < best_f:
                best_f = Fw[0]
                best_x = np.clip(Xw[0], -5, 5)
            # Weighted mean
            xnew = np.sum(w[:, None] * Xw, axis=0)
            # Update evolution paths
            dcor = np.sqrt((1 - np.exp(-2.0 / cc)) * (1 + np.exp(-2.0 * (lam - mu) / cc)))
            pc = (1.0 - cc) * pc + cc / (sigma * np.sqrt(2.0 * mueff)) * (invsqrtC @ (xnew - x))
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) / sigma * (xnew - x)
            # Update covariance
            if np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (evals + 1) / (n + 1))) < (1.4 + 2.0 / (n + 1)) * np.sqrt((n + 1) / (n - 1)):
                delta_hs = 1.0
            else:
                delta_hs = 0.0
            hs = (1.0 - cs) * hs + delta_hs * np.sqrt(cs * (2.0 - cs) * mueff) * (pc / sigma)
            rank1 = np.outer(hs, hs)
            rankmu = np.sum(w[:, None] * np.outer(invsqrtC @ (Xw - x), invsqrtC @ (Xw - x)), axis=0)
            C = (1.0 + cc * (delta_hs - 1.0) * np.outer(pc, pc) + c1 * rank1 + cmu * rankmu) * (1.0 - cc * (delta_hs - 1.0))
            # Actually, standard update:
            C = (1.0 + cc * (delta_hs - 1.0) * np.outer(pc, pc) + c1 * rank1 + cmu * rankmu)
            # Symmetrize
            C = 0.5 * (C + C.T)
            # Eigen decomposition for sqrt inverse
            try:
                D, B = np.linalg.eigh(C)
                D = np.clip(D, 1e-20, None)
                # Update invsqrtC
                invsqrtC = (B / np.sqrt(D)) @ B.T
                # Update sigma
                sigma *= np.exp((cs / (5.0 * (n + 2))) * (np.linalg.norm(ps) / np.sqrt(2.0 * n) * (1.0 + 2.0 / max(n - 1, 1)) - 1.0))
                sigma = np.clip(sigma, 1e-10, 1e5)
                x = xnew
            except np.linalg.LinAlgError:
                # Fallback: reset
                C = np.eye(n)
                invsqrtC = np.eye(n)
                x = xnew

        return best_x, best_f, evals

    def __call__(self, func):
        rng = np.random.default_rng(42)
        deadline = time.time() + 300.0  # 300s margin before 330s kill
        n = self.dim
        budget = self.budget

        # Strategy: Multi-restart CMA-ES
        # Split budget: first 60% for initial CMA-ES from center,
        # then restart from best point with smaller sigma if time/budget remains.
        # For f21, starting from center is reasonable; peaks are scattered.
        # Use a single CMA-ES with adaptive sigma, which is robust.

        x0 = np.zeros(n)
        sigma0 = 2.0  # Initial step size for [-5,5]^5

        best_x, best_f, evals_used = self._cma_es(
            func, x0, sigma0, budget, deadline, rng
        )

        # If we have remaining budget and time, do a second restart
        # from the best point with a smaller sigma to fine-tune
        remaining = budget - evals_used
        if remaining > 20 and time.time() < deadline - 5.0:
            # Restart with smaller sigma
            sigma2 = 0.5
            x2 = best_x.copy()
            bx2, bf2, eu2 = self._cma_es(
                func, x2, sigma2, remaining, deadline, rng
            )
            if bf2 < best_f:
                best_x, best_f = bx2, bf2

        return best_x, best_f
