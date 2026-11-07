"""Combined optimizer: Sobol exploration + CMA-ES exploitation + coordinate descent."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _sobol_next(dim, idx):
    """Generate next Sobol-like point using a simple deterministic sequence."""
    # Use a simple low-discrepancy sequence based on van der Corput
    points = []
    for d in range(dim):
        # Van der Corput sequence with base 2, offset by dimension
        n = idx + 1
        result = 0.0
        denom = 1.0
        base = 2 + d  # different base per dimension for better coverage
        while n > 0:
            denom *= base
            result += (n % base) / denom
            n //= base
        points.append(result)
    return np.array(points)


class Optimizer:
    """
    Hybrid optimizer: Sobol sequence exploration, CMA-ES exploitation,
    and coordinate descent refinement for ill-conditioned functions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        if budget < 20:
            rng = np.random.default_rng(0)
            best_x = rng.uniform(-5, 5, n)
            best_f = func(best_x)
            for _ in range(budget - 1):
                x = rng.uniform(-5, 5, n)
                f = func(x)
                if f < best_f:
                    best_f, best_x = f, x
            return best_x, best_f

        rng = np.random.default_rng(42)
        evals = 0
        best_x = None
        best_f = float('inf')

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Sobol sequence exploration (first ~25% of budget)
        phase1_budget = max(10, budget // 4)
        for i in range(phase1_budget):
            if evals >= budget or time.time() > deadline:
                break
            # Map Sobol point to [-5, 5]^n
            p = _sobol_next(n, i)
            x = -5.0 + 10.0 * p
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            evals += 1
            record(x, f)

        # Phase 2: CMA-ES from best Sobol point
        if evals < budget and time.time() < deadline:
            m = best_x.copy()
            sigma = 1.5
            C = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros(n)

            # CMA-ES parameters
            lam = 4 + int(3 * np.log(n))
            if lam < 4:
                lam = 4
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mueff = 1.0 / np.sum(w**2)

            c_sigma = (mueff + 2) / (n + mueff + 5)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma

            # Track last improvement for stagnation detection
            last_improve_eval = evals
            stagnation_threshold = lam * 3  # 3 generations without improvement

            while evals < budget and time.time() < deadline:
                # Compute C^(-1/2)
                try:
                    D, B = np.linalg.eigh(C)
                    D = np.maximum(D, 1e-10)
                    C_inv_sqrt = (B / np.sqrt(D)) @ B.T
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(n)

                # Sample offspring
                Y = np.zeros((lam, n))
                F = np.empty(lam)
                for i in range(lam):
                    if evals >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    Y[i] = m + sigma * (C_inv_sqrt @ z)
                    Y[i] = np.clip(Y[i], -5.0, 5.0)
                    F[i] = func(Y[i])
                    evals += 1
                    record(Y[i], F[i])

                if evals >= budget or time.time() > deadline:
                    break

                # Sort
                order = np.argsort(F)
                Y_best = Y[order[:mu]]
                F_best = F[order[:mu]]

                # Check for improvement
                if F_best[0] < best_f - 1e-10:
                    last_improve_eval = evals

                # Detect stagnation: restart with smaller sigma if stuck
                if evals - last_improve_eval > stagnation_threshold:
                    sigma *= 0.5
                    last_improve_eval = evals
                    if sigma < 0.01:
                        break

                # Update mean
                m_old = m.copy()
                m = w @ Y_best.T
                m = np.clip(m, -5.0, 5.0)

                # Update path statistics
                m_diff = m - m_old
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ m_diff)

                hc = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * max(1, evals // lam))) <
                      (1.4 + c_c / n) * np.sqrt((n + 1) / (n - 1)))
                pc = (1 - c_c) * pc + hc * np.sqrt(c_c * (2 - c_c) * mueff) * (C_inv_sqrt @ m_diff)

                # Update covariance matrix
                rank_one = pc @ pc.T + (1 - hc) * c_sigma * (2 - c_sigma) * C
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    y_i = C_inv_sqrt @ (Y_best[i] - m_old)
                    rank_mu += w[i] * np.outer(y_i, y_i)

                C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(2 * n) - 1))
                sigma = np.clip(sigma, 0.001, 10.0)

        # Phase 3: Coordinate descent refinement
        if evals < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            step = 0.5

            for coord in range(n * 3):  # 3 passes over all coordinates
                if evals >= budget or time.time() > deadline:
                    break
                d = coord % n
                # Try positive step
                x_trial = x.copy()
                x_trial[d] += step
                x_trial = np.clip(x_trial, -5.0, 5.0)
                if evals < budget and time.time() < deadline:
                    f_trial = func(x_trial)
                    evals += 1
                    if f_trial < fx:
                        x, fx = x_trial, f_trial
                        record(x, fx)
                        step *= 1.5
                    else:
                        step *= 0.5

                # Try negative step
                x_trial = x.copy()
                x_trial[d] -= step
                x_trial = np.clip(x_trial, -5.0, 5.0)
                if evals < budget and time.time() < deadline:
                    f_trial = func(x_trial)
                    evals += 1
                    if f_trial < fx:
                        x, fx = x_trial, f_trial
                        record(x, fx)
                        step *= 1.5
                    else:
                        step *= 0.5

                if step < 0.001:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
