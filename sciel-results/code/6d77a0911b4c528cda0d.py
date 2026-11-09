import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() < deadline

        best_x = None
        best_fx = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal best_x, best_fx, calls
            if calls >= budget:
                return best_fx
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Scan with stratified sampling (LHS)
        n_scan = min(50, budget // 4)
        n_bins = max(2, int(np.ceil(n_scan ** (1.0 / dim))))
        
        scan_points = []
        for i in range(n_scan):
            x = np.zeros(dim)
            for d in range(dim):
                bin_idx = (i + d * n_scan // n_bins) % n_bins
                frac = (bin_idx + rng.uniform()) / n_bins
                x[d] = frac * 10.0 - 5.0
            scan_points.append(x)

        candidates = []
        for x in scan_points:
            if not time_ok() or calls >= budget:
                break
            fx = evaluate(x)
            candidates.append((fx, x.copy()))

        if not candidates:
            x0 = rng.uniform(-5, 5, dim)
            fx0 = evaluate(x0)
            return best_x, best_fx

        candidates.sort(key=lambda t: t[0])

        if not time_ok() or calls >= budget:
            return best_x, best_fx

        # Phase 2: CMA-ES
        lambda_pop = max(10, 4 + int(3 * np.log(dim)))
        
        # Initialize sigma from top candidates
        if len(candidates) >= 2:
            init_sigma = np.linalg.norm(candidates[0][1] - candidates[1][1])
            init_sigma = max(0.1, min(init_sigma, 2.0))
        else:
            init_sigma = 1.0

        mean = candidates[0][1].copy()
        sigma = init_sigma
        C = np.eye(dim)
        
        mu = lambda_pop // 2
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        try:
            C_chol = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_chol = np.eye(dim)
        
        no_improvement = 0
        stall_threshold = 5 * lambda_pop
        generation = 0

        while time_ok() and calls < budget:
            if budget - calls < lambda_pop:
                break

            # Generate offspring
            offspring = []
            for i in range(lambda_pop):
                if calls >= budget or not time_ok():
                    break
                z = rng.standard_normal(dim)
                y = mean + sigma * C_chol @ z
                y = np.clip(y, -5.0, 5.0)
                offspring.append(y)

            if not offspring:
                break

            # Evaluate
            offspring_fx = []
            for y in offspring:
                if calls >= budget or not time_ok():
                    break
                fy = evaluate(y)
                offspring_fx.append((fy, y))

            if len(offspring_fx) < 2:
                break

            # Sort by fitness
            offspring_fx.sort(key=lambda t: t[0])
            
            # Track improvement
            if offspring_fx[0][0] < best_fx - 1e-12:
                no_improvement = 0
            else:
                no_improvement += 1

            # Update mean
            m_actual = min(mu, len(offspring_fx))
            new_mean = np.zeros(dim)
            for i in range(m_actual):
                new_mean += weights[i] * offspring_fx[i][1]

            # Compute C^{-1/2} for path updates
            try:
                C_sqrt = np.linalg.cholesky(C + 1e-15 * np.eye(dim))
                C_inv_sqrt = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)

            # Update path variables
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (new_mean - mean) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma

            # Update sigma
            denom = np.sqrt(1 - (1 - c_sigma) ** (2 * (generation + 1)))
            if denom < 1e-10:
                denom = 1.0
            threshold = (1.4 + 2 / (dim + 1)) * chi_N
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / denom < threshold else 0.0
            
            sigma *= np.exp(c_sigma / chi_N * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - chi_N))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update covariance matrix
            delta_h = h_sigma * (1 / c_sigma - 1)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(m_actual):
                diff = offspring_fx[i][1] - mean
                rank_mu += weights[i] * np.outer(diff, diff)
            
            C = (1 + c_1 * (delta_h - 1) + c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            C = 0.5 * (C + C.T)

            # Update Cholesky factor
            try:
                C_chol = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                C_chol = np.eye(dim)

            mean = new_mean
            generation += 1

            # Stagnation restart
            if no_improvement > stall_threshold:
                mean = best_x.copy()
                sigma = max(0.3, sigma * 0.5)
                C = np.eye(dim)
                try:
                    C_chol = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C_chol = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                no_improvement = 0
                generation = 0

        # Phase 3: (1+1)-ES polish
        if time_ok() and calls < budget:
            polish_budget = min(budget - calls, 30)
            x = best_x.copy()
            fx = best_fx
            step = 0.1
            successes = 0
            total_steps = 0
            
            for _ in range(polish_budget):
                if not time_ok() or calls >= budget:
                    break
                y = x + step * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                total_steps += 1
                
                if total_steps % 5 == 0:
                    rate = successes / 5.0
                    if rate > 0.4:
                        step *= 1.3
                    elif rate < 0.15:
                        step *= 0.7
                    successes = 0
                    if step < 1e-4:
                        break

        return best_x, best_fx
