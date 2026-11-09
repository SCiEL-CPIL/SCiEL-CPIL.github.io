import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        n = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: CMA-ES with proper implementation ---
        lambda_ = max(4 + int(3 * np.log(n)), 8)
        mu = lambda_ // 2

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)

        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))

        # Initialize with a larger sigma to cover the box [-5,5]
        mean = np.zeros(n)
        sigma = 2.5
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        gen = 0

        # Reserve ~30 calls for final polish
        cmaes_budget_limit = total_budget - 30

        def get_C_sqrt(C_mat):
            try:
                L = np.linalg.cholesky(C_mat + 1e-12 * np.eye(n))
                return L
            except:
                eigvals, eigvecs = np.linalg.eigh(C_mat)
                eigvals = np.clip(eigvals, 1e-12, None)
                return eigvecs @ np.diag(np.sqrt(eigvals))

        while calls_made < cmaes_budget_limit:
            if time.time() > deadline:
                break

            # Generate population
            z = np.random.standard_normal((lambda_, n))
            C_sqrt = get_C_sqrt(C)
            y = np.einsum('ij,jk->ik', z, C_sqrt.T)
            x_pop = mean + sigma * y

            # Evaluate population
            f_pop = np.array([evaluate(x_pop[i]) for i in range(lambda_)])

            if calls_made >= cmaes_budget_limit:
                break

            # Sort indices by fitness
            idx = np.argsort(f_pop)

            # Update mean
            x_w = np.zeros(n)
            for i in range(mu):
                x_w += weights[i] * x_pop[idx[i]]

            # Compute C_inv_sqrt
            C_sqrt = get_C_sqrt(C)
            try:
                C_inv_sqrt = np.linalg.inv(C_sqrt)
            except:
                C_inv_sqrt = np.eye(n)

            y_w = (x_w - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mueff) * (C_inv_sqrt @ y_w)

            # h_sigma check
            norm_p_sigma = np.linalg.norm(p_sigma)
            threshold = (1.4 + 2 / (n + 1)) * np.sqrt((mueff + 1) / (mueff + 3))
            h_sigma = 1.0 if norm_p_sigma < threshold else 0.0

            p_c = (1 - c_c) * p_c + h_sigma * c_c * (x_w - mean) / sigma

            # Update Covariance C
            C_update = np.zeros((n, n))
            for i in range(mu):
                y_i = (x_pop[idx[i]] - mean) / sigma
                C_update += weights[i] * np.outer(y_i, y_i)

            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (1 - c_c) * C) + c_mu * C_update

            # Symmetrize C for numerical stability
            C = 0.5 * (C + C.T)

            # Update mean
            mean = x_w

            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (norm_p_sigma / np.sqrt(mueff) - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)

            gen += 1

            # Check convergence
            if sigma < 1e-9:
                break

        # --- Phase 2: Local Polish with adaptive step sizes ---
        if best_x is not None and calls_made < total_budget:
            # Start with a moderate step and decay
            step = 0.1
            min_step = 1e-6
            no_improve_count = 0
            max_no_improve = 15

            while calls_made < total_budget and time.time() < deadline:
                if step < min_step:
                    break

                improved = False
                for dim_idx in range(n):
                    if calls_made >= total_budget or time.time() > deadline:
                        break

                    # Try positive direction
                    x_pos = best_x.copy()
                    x_pos[dim_idx] += step
                    x_pos = np.clip(x_pos, -5, 5)
                    f_pos = evaluate(x_pos)
                    if f_pos < best_f:
                        improved = True
                        # Step in the positive direction further if still improving
                        continue

                    # Try negative direction
                    x_neg = best_x.copy()
                    x_neg[dim_idx] -= step
                    x_neg = np.clip(x_neg, -5, 5)
                    f_neg = evaluate(x_neg)
                    if f_neg < best_f:
                        improved = True

                if improved:
                    no_improve_count = 0
                    # Keep same step size if improving
                else:
                    no_improve_count += 1
                    step *= 0.5
                    if no_improve_count > max_no_improve:
                        break

        # Fallback
        if best_x is None:
            x = np.zeros(n)
            best_f = func(x)
            best_x = x

        return best_x, best_f
