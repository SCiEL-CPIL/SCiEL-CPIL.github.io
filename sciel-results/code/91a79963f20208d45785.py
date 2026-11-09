"""IPOP-CMA-ES with Polish for BBOB f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return None, None
            n_calls += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
                f_val = float(func(x_clipped))
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val, x_clipped
            except Exception:
                return None, None

        rng = np.random.default_rng(42)
        
        # Phase 1: Initial sampling
        n_init = 10
        for _ in range(n_init):
            if check_time() or n_calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            call_func(x)

        if best_x is None:
            x = np.zeros(dim)
            call_func(x)
            if best_x is None:
                return None

        # Phase 2: IPOP-CMA-ES
        n = dim
        lambda_pop = max(4, int(4 + 3 * np.log(n)))
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
        cs = (mueff + 2.0) / (n + mueff + 5.0)
        c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
        chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

        x_mean = best_x.copy()
        sigma = 1.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        best_f_segment = best_f
        stagnation_count = 0
        restarts = 0
        max_restarts = 20
        global_improvement_restarts = 0
        best_f_global_before_restart = best_f

        while n_calls < budget and not check_time():
            # Generate population
            pop = np.zeros((lambda_pop, n))
            fvals = np.zeros(lambda_pop)
            valid_idx = 0
            
            while valid_idx < lambda_pop and n_calls < budget and not check_time():
                z = rng.standard_normal(n)
                try:
                    L = np.linalg.cholesky(C)
                    x_trial = x_mean + sigma * (L @ z)
                except np.linalg.LinAlgError:
                    x_trial = x_mean + sigma * z
                
                f_val, x_eval = call_func(x_trial)
                if f_val is None:
                    break
                pop[valid_idx] = x_eval
                fvals[valid_idx] = f_val
                valid_idx += 1
            
            if valid_idx < lambda_pop:
                break

            order = np.argsort(fvals[:valid_idx])
            pop = pop[order]
            fvals = fvals[order]
            
            if fvals[0] < best_f_segment:
                best_f_segment = fvals[0]
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Update CMA-ES state
            x_mean_new = np.zeros(n)
            for i in range(mu):
                x_mean_new += weights[i] * pop[i]
            
            try:
                L = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(L).T
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(n)

            y_mean = (x_mean_new - x_mean) / sigma
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (C_inv_sqrt @ y_mean)

            hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (valid_idx + 1)))) < (1.4 + 2.0 / (n + 1)) * chi_n

            pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * y_mean

            C = (1.0 + c1 * hsig * (1.0 - hsig) - c1 - cmu) * C + c1 * np.outer(pc, pc)
            y_i = np.zeros((mu, n))
            for i in range(mu):
                y_i[i] = (pop[i] - x_mean) / sigma
            C += cmu * (weights @ (y_i @ y_i.T))
            C = (C + C.T) / 2.0

            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
            sigma = np.clip(sigma, 1e-15, 1e6)
            x_mean = x_mean_new

            # Restart Logic
            if sigma < 1e-4 or (stagnation_count > 10 and sigma < 0.1):
                restarts += 1
                if restarts > max_restarts:
                    break
                
                # Check if we are stuck globally
                if best_f < best_f_global_before_restart:
                    global_improvement_restarts = 0
                    best_f_global_before_restart = best_f
                else:
                    global_improvement_restarts += 1
                    if global_improvement_restarts > 5:
                        break # No global progress in 5 restarts

                # Restart
                sigma *= 2.0
                C = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                x_mean = best_x + rng.normal(0, 0.1, n)
                x_mean = np.clip(x_mean, -5, 5)
                best_f_segment = best_f
                stagnation_count = 0

        # Phase 3: Polish
        if n_calls < budget and not check_time():
            # Use remaining budget for a tight local search
            sigma_polish = 0.01
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            x_mean = best_x.copy()
            best_f_segment = best_f
            stagnation_count = 0
            
            while n_calls < budget and not check_time():
                pop = np.zeros((lambda_pop, n))
                fvals = np.zeros(lambda_pop)
                valid_idx = 0
                while valid_idx < lambda_pop and n_calls < budget and not check_time():
                    z = rng.standard_normal(n)
                    try:
                        L = np.linalg.cholesky(C)
                        x_trial = x_mean + sigma_polish * (L @ z)
                    except np.linalg.LinAlgError:
                        x_trial = x_mean + sigma_polish * z
                    
                    f_val, x_eval = call_func(x_trial)
                    if f_val is None:
                        break
                    pop[valid_idx] = x_eval
                    fvals[valid_idx] = f_val
                    valid_idx += 1
                
                if valid_idx < lambda_pop:
                    break

                order = np.argsort(fvals[:valid_idx])
                pop = pop[order]
                fvals = fvals[order]
                
                if fvals[0] < best_f_segment:
                    best_f_segment = fvals[0]
                    stagnation_count = 0
                    sigma_polish *= 0.9 # Shrink slowly
                else:
                    stagnation_count += 1
                    if stagnation_count > 5:
                        break # Stop polishing if no progress

                x_mean_new = np.zeros(n)
                for i in range(mu):
                    x_mean_new += weights[i] * pop[i]
                
                try:
                    L = np.linalg.cholesky(C)
                    C_inv_sqrt = np.linalg.inv(L).T
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(n)

                y_mean = (x_mean_new - x_mean) / sigma_polish
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (C_inv_sqrt @ y_mean)
                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (valid_idx + 1)))) < (1.4 + 2.0 / (n + 1)) * chi_n
                pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * y_mean
                C = (1.0 + c1 * hsig * (1.0 - hsig) - c1 - cmu) * C + c1 * np.outer(pc, pc)
                y_i = np.zeros((mu, n))
                for i in range(mu):
                    y_i[i] = (pop[i] - x_mean) / sigma_polish
                C += cmu * (weights @ (y_i @ y_i.T))
                C = (C + C.T) / 2.0
                sigma_polish *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma_polish = np.clip(sigma_polish, 1e-15, 1e6)
                x_mean = x_mean_new

        return best_x, best_f
# EVOLVE-BLOCK-END
