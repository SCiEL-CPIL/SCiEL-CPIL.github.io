import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        best_x = np.zeros(dim)
        best_f = float('inf')
        n_calls = 0

        def timed_call(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = func(x)
            n_calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return float(val)

        # 1. Random Initialization
        n_init = 20
        init_points = []
        for _ in range(n_init):
            if n_calls >= budget or time.time() > deadline:
                break
            x = np.random.uniform(-4.5, 4.5, dim)
            f = timed_call(x)
            init_points.append((f, x.copy()))

        if n_calls >= budget:
            return best_x, best_f

        init_points.sort(key=lambda x: x[0])
        x_start = init_points[0][1].copy()

        def run_cma_es(x_start, sigma_start, max_iter):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget or time.time() > deadline:
                return x_start, best_f

            # CMA-ES parameters tuned for dim=5
            lam = 15
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1. / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            chi_lambda = np.sqrt(dim * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2)))

            # State
            m = x_start.copy()
            sigma = sigma_start
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            best_f_local = best_f
            best_x_local = x_start.copy()
            no_improvement = 0
            stagnation_limit = 15

            for it in range(max_iter):
                if n_calls >= budget or time.time() > deadline:
                    break

                # Sample lambda candidates
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-18, None)
                sqrt_eigvals = np.sqrt(eigvals)

                Z = np.random.randn(lam, dim)
                X = m + sigma * (Z * sqrt_eigvals) @ eigvecs.T

                # Evaluate
                F = np.empty(lam)
                for i in range(lam):
                    if n_calls >= budget or time.time() > deadline:
                        break
                    F[i] = timed_call(X[i])

                if n_calls >= budget or time.time() > deadline:
                    break

                # Sort
                order = np.argsort(F)
                F_sorted = F[order]
                X_sorted = X[order]

                # Update best
                if F_sorted[0] < best_f_local:
                    best_f_local = F_sorted[0]
                    best_x_local = X_sorted[0].copy()
                    no_improvement = 0
                else:
                    no_improvement += 1

                if no_improvement > stagnation_limit:
                    break

                # Update Mean
                m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)

                # Update Sigma (CSA)
                v = (m_new - m) / sigma
                v_rot = eigvecs.T @ v
                v_scaled = v_rot / sqrt_eigvals
                v_inv_half = eigvecs @ v_scaled

                p_sigma = (1 - c_sigma) * p_sigma + \
                          np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v_inv_half

                # h_sigma
                h_sigma_thresh = (1 + 1.4 / mu_eff) * np.sqrt((1 - c_c) * (2 - c_c))
                h_sigma = 1.0 if (np.linalg.norm(p_sigma) / chi_lambda) < h_sigma_thresh else 0.0

                # Update C
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * v_inv_half

                Y = (X_sorted[:mu] - m) / sigma
                M2 = (Y.T * weights) @ Y

                C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * M2
                C = (C + C.T) / 2

                # Update sigma
                sigma = sigma * np.exp(c_sigma / chi_lambda * (np.linalg.norm(p_sigma) / chi_lambda - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update m
                m = m_new

            return best_x_local, best_f_local

        def coordinate_descent(x_start, max_calls):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget or time.time() > deadline:
                return x_start, best_f

            x = x_start.copy()
            f = best_f
            steps = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]

            for step in steps:
                if n_calls >= budget or time.time() > deadline:
                    break
                improved = True
                while improved and (n_calls < budget and time.time() < deadline):
                    improved = False
                    for i in range(dim):
                        if n_calls >= budget or time.time() > deadline:
                            break
                        for direction in [1, -1]:
                            if n_calls >= budget or time.time() > deadline:
                                break
                            new_x = x.copy()
                            new_x[i] += direction * step
                            new_x = np.clip(new_x, -5.0, 5.0)
                            if np.allclose(new_x, x):
                                continue
                            f_new = timed_call(new_x)
                            if f_new < f:
                                x = new_x
                                f = f_new
                                improved = True
            return x, f

        # Execution Plan:
        # 1. Random Initialization (done)
        # 2. CMA-ES with moderate sigma
        # 3. Coordinate Descent polish
        # 4. If budget remains, another CMA-ES run with larger sigma (restart)
        # 5. Final Coordinate Descent

        x_best, f_best = run_cma_es(x_start, 1.0, 100)

        if n_calls < budget and time.time() < deadline:
            x_best, f_best = coordinate_descent(x_best, budget - n_calls)

        if n_calls < budget and time.time() < deadline:
            # Restart with larger sigma to explore more
            x_best, f_best = run_cma_es(x_best, 2.0, 50)

        if n_calls < budget and time.time() < deadline:
            x_best, f_best = coordinate_descent(x_best, budget - n_calls)

        return x_best, f_best
