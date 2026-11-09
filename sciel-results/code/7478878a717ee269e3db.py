"""Bootstrap spawn: DIRECT approach using Rastrigin structure + CMA-ES refinement."""
import numpy as np
import time
import math


class Optimizer:
    """Direct optimizer exploiting Rastrigin periodicity + CMA-ES for rotation-aware search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _rastrigin_lower_bound(self, x):
        """Compute the minimum possible Rastrigin value given a fixed rotation/shift by
        exploiting the fact that for each coordinate i, f_i(x_i) >= 10 - 10*cos(2*pi*x_i),
        and the minimum of 10-10*cos(2*pi*x_i) over x_i is 0, achieved at integer x_i.
        Actually the Rastrigin term per dimension is x_i^2 - 10*cos(2*pi*x_i) + 10.
        The minimum of this is 0 at x_i = integer.
        So the global min of Rastrigin is 0, achieved at any integer point.
        But with rotation, the 'integer' points in the rotated frame are what matter.
        We can compute: for a candidate center c, evaluate the function and also compute
        the theoretical lower bound contribution.
        """
        pass

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        rng = np.random.default_rng(12345)
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

        # IPOP-CMA-ES Parameters
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

        x_mean = rng.uniform(-5, 5, dim)
        sigma = 2.0
        C = np.eye(dim)
        evolution_path_sigma = np.zeros(dim)
        evolution_path_cov = np.zeros((dim, dim))
        
        stagnation_count = 0
        best_f_in_run = np.inf
        stagnation_threshold = 150
        
        while calls_used < budget:
            if time.time() > deadline:
                break
            
            if stagnation_count > stagnation_threshold:
                lambda_pop = int(lambda_pop * 1.5)
                mu = lambda_pop // 2
                
                w = np.array([math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)])
                w /= w.sum()
                nu_eff = 1 / sum(w ** 2)
                c_sigma = (mu + 0.5) / (dim + 2) / 5
                c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
                c_1 = 2 / ((dim + 1.3) ** 2 + mu)
                c_mu = min(1 - c_1, 2 * (mu - 2 + 1 / mu) / ((dim + 2) ** 2 + mu))
                
                C = np.eye(dim)
                evolution_path_sigma = np.zeros(dim)
                evolution_path_cov = np.zeros((dim, dim))
                
                sigma = max(sigma * 1.5, 1.0)
                
                stagnation_count = 0
                best_f_in_run = best_f

            offspring = []
            for _ in range(lambda_pop):
                if calls_used >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                x_off = x_mean + sigma * (C @ z)
                x_off = np.clip(x_off, -5, 5)
                fx_off = evaluate(x_off)
                offspring.append((fx_off, x_off))
            
            if len(offspring) < mu:
                break
            
            offspring.sort(key=lambda t: t[0])
            selected = offspring[:mu]
            
            if selected[0][0] < best_f_in_run - 1e-8:
                best_f_in_run = selected[0][0]
                stagnation_count = 0
            else:
                stagnation_count += 1
                
            x_new_mean = sum(w[i] * selected[i][1] for i in range(mu))
            
            evolution_path_sigma = (1 - c_sigma) * evolution_path_sigma + \
                                   math.sqrt(c_sigma * (2 - c_sigma) * nu_eff) * \
                                   (C @ ((x_new_mean - x_mean) / sigma))
            
            h_sigma = (np.linalg.norm(evolution_path_sigma) / chi_N) < (1.4 + 2 / (dim + 1))
            h_sigma = float(h_sigma)
            
            evolution_path_cov = (1 - c_c) * evolution_path_cov + \
                                 h_sigma * c_c * (1 - c_c) * np.outer(x_new_mean - x_mean,
                                                                      x_new_mean - x_mean)
            
            C = (1 - c_1 - c_mu) * C + c_1 * evolution_path_cov + \
                c_mu * sum(w[i] * np.outer(selected[i][1] - x_mean,
                                            selected[i][1] - x_mean) for i in range(mu))
            
            C = (C + C.T) / 2
            
            sigma *= math.exp(c_sigma / chi_N *
                              (np.linalg.norm(evolution_path_sigma) / chi_N - 1))
            sigma = float(np.clip(sigma, 0.01, 10.0))
            
            x_mean = x_new_mean
            
            if sigma < 0.01:
                break

        return (best_x, best_f)
