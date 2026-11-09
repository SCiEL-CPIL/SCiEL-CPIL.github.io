"""Direct optimizer exploiting Rastrigin periodicity via multi-start CMA-ES with restarts."""
import numpy as np
import time
import math


class Optimizer:
    """Direct optimizer using multi-start CMA-ES with adaptive restarts for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline and calls_used < budget

        # Phase 1: Initial random sampling to find a promising basin
        n_random = 20
        for _ in range(n_random):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)
        
        if not time_ok():
            return (best_x, best_f)

        # CMA-ES parameters
        lambda_pop = 10 + int(3 * math.log(dim))
        mu = lambda_pop // 2
        
        c_sigma = (mu + 0.5) / (dim + 2) / 5
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (mu - 2 + 1 / mu) / ((dim + 2) ** 2 + mu))
        chi_N = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        w = np.array([math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)])
        w /= w.sum()
        nu_eff = 1 / sum(w ** 2)
        
        def run_cmaes(x0, sigma0, max_evals):
            """Run CMA-ES from x0 with initial sigma0 for at most max_evals evaluations."""
            nonlocal calls_used, best_x, best_f
            
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            evolution_path_sigma = np.zeros(dim)
            evolution_path_cov = np.zeros((dim, dim))
            
            last_best = best_f
            no_improve = 0
            evals_this_run = 0
            
            while evals_this_run < max_evals and time_ok():
                # Generate population
                offspring = []
                for _ in range(lambda_pop):
                    if not time_ok() or evals_this_run >= max_evals:
                        break
                    z = rng.standard_normal(dim)
                    x_off = x_mean + sigma * (C @ z)
                    x_off = np.clip(x_off, -5, 5)
                    fx_off = evaluate(x_off)
                    evals_this_run += 1
                    offspring.append((fx_off, x_off))
                
                if len(offspring) < mu:
                    break
                
                offspring.sort(key=lambda t: t[0])
                selected = offspring[:mu]
                
                if selected[0][0] < last_best - 1e-10:
                    last_best = selected[0][0]
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve > 15:
                    break
                
                # Update mean
                x_new_mean = sum(w[i] * selected[i][1] for i in range(mu))
                
                # Update sigma evolution path
                evolution_path_sigma = (1 - c_sigma) * evolution_path_sigma + \
                                       math.sqrt(c_sigma * (2 - c_sigma) * nu_eff) * \
                                       (C @ ((x_new_mean - x_mean) / sigma))
                
                h_sigma = float(np.linalg.norm(evolution_path_sigma) / chi_N < (1.4 + 2 / (dim + 1)))
                
                # Update covariance evolution path
                evolution_path_cov = (1 - c_c) * evolution_path_cov + \
                                     h_sigma * c_c * (1 - c_c) * np.outer(x_new_mean - x_mean,
                                                                          x_new_mean - x_mean)
                
                # Update covariance matrix
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank_one = np.outer(x_new_mean - x_mean, x_new_mean - x_mean)
                rank_mu = sum(w[i] * np.outer(selected[i][1] - x_mean,
                                              selected[i][1] - x_mean) for i in range(mu))
                
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * evolution_path_cov + c_mu * rank_mu
                C = (C + C.T) / 2
                
                # Update sigma
                sigma *= math.exp(c_sigma / chi_N * (np.linalg.norm(evolution_path_sigma) / chi_N - 1))
                sigma = float(np.clip(sigma, 0.001, 10.0))
                
                x_mean = x_new_mean
                
                if sigma < 0.001:
                    break
            
            return best_f

        # Phase 2: Multi-start CMA-ES with restarts
        # Strategy: Start with large sigma to explore, restart if stuck.
        # Use a portion of the budget for exploration, rest for exploitation.
        
        remaining_budget = budget - calls_used
        # Divide budget into chunks for multiple runs
        num_runs = 5
        evals_per_run = max(10, int(remaining_budget / num_runs))
        
        for i in range(num_runs):
            if not time_ok():
                break
            
            # Alternate between best point and random restarts
            if i == 0:
                x0 = best_x.copy()
                sigma0 = 1.0  # Large sigma to jump between local minima
            elif i % 2 == 0:
                x0 = best_x.copy()
                sigma0 = 0.5
            else:
                x0 = rng.uniform(-5, 5, dim)
                sigma0 = 1.5  # Even larger for exploration from random start
            
            max_evals = min(evals_per_run, budget - calls_used)
            if max_evals > 0:
                run_cmaes(x0, sigma0, max_evals)
        
        # Phase 3: Final refinement with small sigma
        if time_ok() and budget - calls_used > 10:
            max_evals = min(50, budget - calls_used)
            run_cmaes(best_x.copy(), 0.05, max_evals)

        return (best_x, best_f)
