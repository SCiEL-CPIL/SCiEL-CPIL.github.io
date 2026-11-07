"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with adaptive restarts for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        best_x = None
        best_f = float('inf')
        eval_count = 0
        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                raise Exception("Budget exceeded")
            if time.time() > deadline:
                raise Exception("Time exceeded")
            
            x = np.asarray(x, dtype=float)
            fx = func(x)
            eval_count += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            return fx

        # Phase 1: Global Search with Sobol + strategic points
        try:
            from scipy.stats import qmc
            n_scout = 40
            sampler = qmc.Sobol(d=dim, scramble=True)
            samples = sampler.random(n=n_scout)
            samples = samples * 10.0 - 5.0
            scout_points = list(samples)
        except ImportError:
            n_scout = 40
            scout_points = np.random.uniform(-5, 5, size=(n_scout, dim)).tolist()

        # Add strategic points
        scout_points.append(np.zeros(dim))
        scout_points.append(np.ones(dim))
        scout_points.append(-np.ones(dim))

        for p in scout_points:
            if time.time() > deadline or eval_count >= budget:
                break
            try:
                fp = wrapped_func(p)
            except Exception:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
                eval_count += 1
            except Exception:
                pass

        # Phase 2: CMA-ES implementation
        remaining = budget - eval_count
        if remaining > 30:
            # Simple CMA-ES implementation
            def cma_es(x0, sigma0, max_evals):
                nonlocal eval_count, best_x, best_f, deadline, budget
                if time.time() > deadline or eval_count >= budget:
                    return x0, float('inf')
                
                dim_local = len(x0)
                # CMA-ES parameters
                mu = max(1, dim_local // 2)  # number of parents
                lambda_pop = 2 * mu  # population size
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= np.sum(weights)
                mueff = 1.0 / np.sum(weights**2)
                
                # Adaptation rates
                cc = (4 + mueff / dim_local) / (dim_local + 4 + 2 * mueff / dim_local)
                cs = (mueff + 2) / (dim_local + mueff + 5)
                c1 = 2 / ((dim_local + 1.3)**2 + mueff)
                cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim_local + 2)**2 + mueff))
                
                # Initialize
                x_mean = x0.copy()
                sigma = sigma0
                B = np.eye(dim_local)
                D = np.ones(dim_local)
                pc = np.zeros(dim_local)
                ps = np.zeros(dim_local)
                chi_n = np.sqrt(dim_local) * (1 - 1/(4*dim_local) + 1/(20*dim_local**2))
                
                # Initial population
                pop = []
                for i in range(lambda_pop):
                    z = np.random.randn(dim_local)
                    x = x_mean + sigma * (B @ (D * z))
                    pop.append(x)
                
                # Evaluate initial population
                fitness = []
                for x in pop:
                    if time.time() > deadline or eval_count >= budget:
                        return x_mean, float('inf')
                    try:
                        f = wrapped_func(x)
                        fitness.append(f)
                    except Exception:
                        return x_mean, float('inf')
                
                # Sort by fitness
                indices = np.argsort(fitness)
                x_sorted = [pop[i] for i in indices]
                fitness_sorted = [fitness[i] for i in indices]
                
                # Main loop
                while eval_count < budget and time.time() < deadline:
                    # Update mean
                    x_mean_new = sum(weights[i] * x_sorted[i] for i in range(mu))
                    
                    # Update step size control
                    if sigma > 0:
                        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (B @ (D * ((x_mean_new - x_mean) / sigma)))
                    
                    # Update covariance matrix
                    if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 1/dim_local) * chi_n:
                        pc = (1 - cc) * pc + cc / sigma * (x_mean_new - x_mean)
                        
                        delta_sig = (np.linalg.norm(ps) / chi_n - 1) * cs * 5
                        sigma *= np.exp(delta_sig / (dim_local + 2))
                        
                        rank_one = np.outer(pc, pc)
                        rank_mu = sum(weights[i] * np.outer((x_sorted[i] - x_mean) / sigma, (x_sorted[i] - x_mean) / sigma) for i in range(mu))
                        
                        C = (1 + c1 * (2 - c1) - c1 - cmu) * np.eye(dim_local) + c1 * rank_one + cmu * rank_mu
                        
                        # Update B and D from C
                        D, B = np.linalg.eigh(C)
                        D = np.sqrt(np.maximum(D, 1e-20))
                        B = B * D  # Actually B should be the eigenvectors, D the square roots of eigenvalues
                        # Correct: B is eigenvectors, D is sqrt of eigenvalues
                        eigvals, eigvecs = np.linalg.eigh(C)
                        D = np.sqrt(np.maximum(eigvals, 1e-20))
                        B = eigvecs
                    
                    x_mean = x_mean_new
                    
                    # Generate new population
                    pop = []
                    for i in range(lambda_pop):
                        z = np.random.randn(dim_local)
                        x = x_mean + sigma * (B @ (D * z))
                        pop.append(x)
                    
                    # Evaluate
                    fitness = []
                    for x in pop:
                        if time.time() > deadline or eval_count >= budget:
                            return x_mean, float('inf')
                        try:
                            f = wrapped_func(x)
                            fitness.append(f)
                        except Exception:
                            return x_mean, float('inf')
                    
                    # Sort
                    indices = np.argsort(fitness)
                    x_sorted = [pop[i] for i in indices]
                    fitness_sorted = [fitness[i] for i in indices]
                
                return x_mean, fitness_sorted[0] if fitness_sorted else float('inf')
            
            # Run CMA-ES with restarts
            max_restarts = 2
            evals_per_restart = max(20, remaining // max_restarts)
            
            for restart in range(max_restarts):
                if time.time() > deadline or eval_count >= budget:
                    break
                
                if restart == 0:
                    x0 = best_x.copy()
                    sigma0 = 1.0
                else:
                    x0 = best_x + np.random.normal(0, 0.5, size=dim)
                    x0 = np.clip(x0, -5, 5)
                    sigma0 = 0.5
                
                try:
                    x_res, f_res = cma_es(x0, sigma0, evals_per_restart)
                    if f_res < best_f:
                        best_f = f_res
                        best_x = x_res.copy()
                except Exception:
                    pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
