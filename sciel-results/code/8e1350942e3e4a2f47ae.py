"""Rewritten optimizer using CMA-ES with restarts for BBOB-style functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for 5-dimensional BBOB functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best overall
        best_x = rng.uniform(-5, 5, dim)
        best_f = func(best_x)
        total_calls = 1

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * ln(n))
        lam = max(4 + int(3 * np.log(dim)), 4)
        # Initial sigma: cover a reasonable fraction of the box
        sigma_init = 2.0

        # Restart strategy: if stagnation detected, restart with new center
        stagnation_window = 15 * lam  # no improvement in this many evaluations
        no_improve_count = 0
        restart_count = 0
        max_restarts = 5

        while total_calls < budget and time.time() < deadline:
            # Check if we should restart
            if no_improve_count >= stagnation_window and restart_count < max_restarts:
                restart_count += 1
                # Restart: new center near best found, larger sigma
                center = best_x + rng.normal(0, 1.5, dim)
                center = np.clip(center, -5, 5)
                sigma = max(sigma * 2.0, 1.0)
                no_improve_count = 0
            elif no_improve_count >= stagnation_window:
                # No more restarts allowed; keep going with smaller sigma
                sigma *= 0.5
                no_improve_count = 0

            # CMA-ES iteration
            # Generate population
            mu = lam // 2  # number of parents
            # Weights: linear weights
            w = np.arange(1, mu + 1, dtype=float)
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)

            # Strategy parameters
            cs = (mueff + 2) / (dim + mueff + 5)
            ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
            c1 = (2 / ((dim + 1.3) ** 2 + mueff)) * (0.5 + 2 / (dim + mueff))
            cmu = min(1 - c1, (2 * (mueff - 2 + 1 / mueff)) / ((dim + 2) ** 2 + mueff))
            chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Initialize state
            if not hasattr(self, '_cma_state') or self._cma_state.get('restart') != restart_count:
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                B = np.eye(dim)
                D = np.ones(dim)
                self._cma_state = {
                    'x': center.copy(),
                    'sigma': sigma,
                    'pc': pc,
                    'ps': ps,
                    'B': B,
                    'D': D,
                    'restart': restart_count
                }

            state = self._cma_state
            xmean = state['x']
            sigma = state['sigma']
            pc = state['pc']
            ps = state['ps']
            B = state['B']
            D = state['D']

            # Generate population
            N = rng.standard_normal((lam, dim))
            # Transform: xmean + sigma * B @ D * N
            pop = xmean + sigma * (B @ (D[:, None] * N.T)).T
            pop = np.clip(pop, -5, 5)

            # Evaluate population
            fvals = np.array([func(x) for x in pop])
            total_calls += lam

            # Sort by fitness
            idx = np.argsort(fvals)
            pop = pop[idx]
            fvals = fvals[idx]

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pop[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += lam

            # Weighted mean of parents
            w_norm = w / w.sum()
            xnew = np.sum(w_norm[:, None] * pop[:mu], axis=0)

            # Update evolution path ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / (sigma) * (B @ (D * (xnew - xmean)))

            # Check Hoeffding test for sigma adaptation
            if np.linalg.norm(ps) / chi < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - cs) ** (2 * (budget - total_calls))):
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi - 1))

            # Update evolution path pc
            hsig = (np.linalg.norm(ps) / chi) < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - cs) ** 2)
            hsig = float(hsig)
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mueff) * (xnew - xmean) / sigma

            # Update covariance matrix
            C = (B @ np.diag(D ** 2)) @ B.T
            C = (1 + c1 * hsig * (2 - hsig) - c1) * C
            rank1 = np.outer(pc, pc)
            C += c1 * rank1
            rankmu = np.sum(w_norm[:, None, None] * np.outer(pop[:mu] - xmean, pop[:mu] - xmean), axis=0) / sigma ** 2
            C += cmu * rankmu

            # Eigen decomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                D = np.sqrt(eigvals)
                B = eigvecs
            except np.linalg.LinAlgError:
                pass

            # Update state
            state['x'] = xnew
            state['sigma'] = sigma
            state['pc'] = pc
            state['ps'] = ps
            state['B'] = B
            state['D'] = D

            # Clip mean to box
            state['x'] = np.clip(state['x'], -5, 5)

            # Check time
            if time.time() > deadline - 5:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
