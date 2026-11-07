"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with coordinate-descent polish for Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Start at center (0,...,0) which is often closer to hidden min in [-4,4]^5
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))  # ~12 for dim=5
        # Step size
        sigma = 1.0
        # Covariance matrix
        C = np.eye(dim)
        # Evolution path for sigma
        ps = np.zeros(dim)
        # Evolution path for C
        pc = np.zeros(dim)
        # Learning rates
        cs = (1 + 2 * max(0, np.log((lam - 2) / (dim + 1)))) / (dim + 3)
        c1 = 2 / ((dim + 1.3) ** 2 + lam)
        cmu = min(1 - c1, 2 * ((lam - 2 + 1 / dim) / (dim + 2) ** 2 - 1))
        # Damping
        damps = 1 + 2 * max(0, np.sqrt((lam / (dim + 1) - 1) * cs - 1)) + cs
        dampc = 1 + 2 * max(0, np.sqrt((lam / (dim + 1) - 1) * c1 - 1)) + c1

        # Cholesky of C for sampling
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(dim)

        call_count = 1  # we already called once

        while call_count < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            offspring = np.zeros((lam, dim))
            for i in range(lam):
                z = rng.standard_normal(dim)
                offspring[i] = x + sigma * (L @ z)

            # Evaluate offspring
            fvals = np.zeros(lam)
            for i in range(lam):
                if call_count >= budget:
                    fvals[i] = fvals[i - 1] if i > 0 else fx
                    break
                fvals[i] = func(offspring[i])
                call_count += 1

            # Sort by fitness
            idx = np.argsort(fvals)
            offspring = offspring[idx]
            fvals = fvals[idx]

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = offspring[0].copy()

            # Weighted mean
            weights = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam + 1))
            weights /= weights.sum()
            mueff = int(np.sum(weights[:int(np.floor(lam / 2))]) ** 2)

            # New mean
            x_new = np.sum(offspring[:int(np.floor(lam / 2))] * weights[:int(np.floor(lam / 2))].reshape(-1, 1), axis=0)

            # Update evolution paths
            y = (x_new - x) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff / damps) * y
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (call_count - 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - cs) ** 2):
                pc = (1 - c1) * pc + c1 * (y - ps)

            # Update covariance
            delta_hs = (1 + 2 * max(0, np.sqrt((lam - 2) / (dim + 1) - 1))) * (1 - (1 - c1) ** 2)
            C = (1 + c1 * (mueff - 2 + cmu) - c1) * C \
                + c1 * (np.outer(pc, pc) + delta_hs * np.eye(dim)) \
                + cmu / damps * np.sum([
                    np.outer(offspring[i] - x, (offspring[i] - x)) / sigma ** 2
                    for i in range(int(np.floor(lam / 2)))
                ] * weights[:int(np.floor(lam / 2))], axis=0)

            # Symmetrize
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(mueff) - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            x = x_new

            # Update Cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize
                C += 1e-6 * np.eye(dim)
                L = np.linalg.cholesky(C)

            # Coordinate descent polish every ~50 calls
            if call_count % 50 == 0 and call_count < budget - 5:
                # Do a few rounds of coordinate descent
                for _ in range(3):
                    if time.time() > deadline:
                        break
                    for d in range(dim):
                        if call_count >= budget:
                            break
                        # Try small steps in each coordinate
                        for step in [0.1, -0.1, 0.01, -0.01]:
                            if call_count >= budget:
                                break
                            trial = best_x.copy()
                            trial[d] += step
                            trial = np.clip(trial, -5, 5)
                            ft = func(trial)
                            call_count += 1
                            if ft < best_f:
                                best_f = ft
                                best_x = trial

        # Final coordinate descent polish with remaining budget
        remaining = budget - call_count
        if remaining > 0 and time.time() < deadline:
            for _ in range(min(remaining, 20)):
                if time.time() > deadline:
                    break
                for d in range(dim):
                    if call_count >= budget:
                        break
                    for step in [0.05, -0.05, 0.01, -0.01, 0.001, -0.001]:
                        if call_count >= budget:
                            break
                        trial = best_x.copy()
                        trial[d] += step
                        trial = np.clip(trial, -5, 5)
                        ft = func(trial)
                        call_count += 1
                        if ft < best_f:
                            best_f = ft
                            best_x = trial

        return best_x, best_f
# EVOLVE-BLOCK-END
