"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid CMA-ES + (1+1)-ES optimizer for the BBOB-style task."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        # Phase 1: CMA-ES for global exploration (~60% of budget)
        # Phase 2: (1+1)-ES with coordinate descent for local refinement (~40% of budget)
        
        cma_budget = int(budget * 0.6)
        es_budget = budget - cma_budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        evals = 0
        
        def time_exceeded():
            return time.time() > deadline
        
        # CMA-ES parameters
        lambda_pop = 12
        mu = lambda_pop // 2
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C_sqrt = np.eye(dim)
        
        # CMA-ES phase
        while evals < cma_budget and not time_exceeded():
            # Generate population
            pop = np.zeros((lambda_pop, dim))
            for i in range(lambda_pop):
                if time_exceeded() or evals >= cma_budget:
                    break
                y = rng.standard_normal(dim)
                z = C_sqrt @ y
                pop[i] = mean + sigma * z
                np.clip(pop[i], -5, 5, out=pop[i])
                evals += 1
            
            if time_exceeded() or evals >= cma_budget:
                break
                
            # Evaluate fitness
            fitness = np.full(lambda_pop, np.inf)
            for i in range(lambda_pop):
                if evals > cma_budget:
                    break
                try:
                    f_val = func(pop[i])
                except Exception:
                    f_val = np.inf
                fitness[i] = f_val
                if f_val < best_f:
                    best_f = f_val
                    best_x = pop[i].copy()
            
            if time_exceeded():
                break
                
            # Sort population by fitness
            idx = np.argsort(fitness)
            pop = pop[idx]
            fitness = fitness[idx]
            
            # Update Mean
            mean_new = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)
            
            # Calculate step for CSA
            delta_mean = (mean_new - mean) / sigma
            try:
                v = np.linalg.solve(C_sqrt, delta_mean)
            except np.linalg.LinAlgError:
                v = delta_mean
            
            # Update pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * v
            
            # Update sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v
            
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            
            # Update Covariance Matrix
            H = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (pop[i] - mean) / sigma
                H += weights[i] * np.outer(y_i, y_i)
            
            C = (1 - c_1 - c_mu) * C 
            C += c_1 * (np.outer(pc, pc) + (1 - (1 - c_c)**2) * np.eye(dim))
            C += c_mu * H
            C = (C + C.T) / 2
            
            # Update Cholesky factor
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-12 * np.eye(dim)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    break
                    
            mean = mean_new
            
            if sigma < 1e-10:
                break
        
        # Phase 2: (1+1)-ES with adaptive step size and coordinate descent
        if evals < budget and not time_exceeded():
            x = best_x.copy()
            fx = best_f
            sigma_es = max(0.5, sigma * 0.5)  # Start with smaller step from CMA-ES
            window_size = 8
            recent_outcomes = []
            
            while evals < budget and not time_exceeded():
                # Adaptive sigma based on success rate
                if len(recent_outcomes) >= window_size:
                    success_rate = sum(recent_outcomes) / window_size
                    if success_rate > 0.35:
                        sigma_es *= 1.15
                    elif success_rate < 0.15:
                        sigma_es *= 0.85
                    else:
                        sigma_es *= 1.0
                    recent_outcomes = []
                
                # Generate candidate
                y = np.clip(x + sigma_es * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                evals += 1
                
                if fy < fx:
                    x, fx = y, fy
                    recent_outcomes.append(1)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    recent_outcomes.append(0)
                
                # Periodic coordinate descent for local refinement
                if evals % 30 == 0 and evals < budget - 15:
                    for d in range(dim):
                        if evals >= budget - 5 or time_exceeded():
                            break
                        # Binary search along dimension d
                        lo, hi = max(-5, x[d] - 1.0), min(5, x[d] + 1.0)
                        for _ in range(4):
                            if evals >= budget - 5 or time_exceeded():
                                break
                            mid = (lo + hi) / 2.0
                            trial = x.copy()
                            trial[d] = mid
                            trial = np.clip(trial, -5, 5)
                            ft = func(trial)
                            evals += 1
                            if ft < fx:
                                fx, x = ft, trial.copy()
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                                lo = mid
                            else:
                                hi = mid
        
        return best_x, best_f
# EVOLVE-BLOCK-END
