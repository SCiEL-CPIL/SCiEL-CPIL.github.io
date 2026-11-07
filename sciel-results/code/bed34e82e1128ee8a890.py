"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 320.0  # Safety margin before 330s kill
        
        best_x = None
        best_f = np.inf
        total_calls = 0
        
        # We will use a restart strategy.
        # Each restart gets a chunk of the budget.
        # Let's try to use the full budget in one run if possible, but CMA-ES 
        # can be slow to converge if not tuned. 
        # Given dim=5 and budget=500, a single CMA-ES run is likely sufficient 
        # if initialized well, but restarts help with local minima.
        
        # Let's do a single robust CMA-ES run with adaptive restarts if it stalls.
        # Actually, for Lunacek, a single CMA-ES run often finds the global min 
        # if the initial sigma is large enough. Let's start with a standard CMA-ES.
        
        # Initial point: center of the box
        x0 = np.zeros(self.dim)
        
        # Initial sigma: 2.0 is a good start for [-5,5]
        sigma = 2.0
        
        # CMA-ES parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(self.dim))
        # Number of selected points
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # Step sizes for covariance and mean
        c_sigma = (mueff + 2) / (self.dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (self.dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / self.dim) / (self.dim + 4 + 2 * mueff / self.dim)
        c_1 = (2 - mueff / (self.dim + 1) + 2 * w[0]) / ((self.dim + 1.3)**2 + mueff)
        c_mu = (2 * (mueff / (self.dim + 1) - 1) + c_1) / ((self.dim + 2)**2 + mueff)
        
        # Covariance matrix C and its decomposition
        C = np.eye(self.dim)
        C_eigvals = np.ones(self.dim)
        C_eigvecs = np.eye(self.dim)
        
        # Memory variables
        ps = np.zeros(self.dim)
        pc = np.zeros(self.dim)
        
        def update_covariance(C, C_eigvals, C_eigvecs, pc, ps, sigma, c_sigma, d_sigma, c_c, c_1, c_mu, w, mu, mueff, dim, yk, yk_prev):
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * yk / sigma
            
            # Check if ps is too large
            hsig = (np.sqrt((1 - (1 - c_sigma)**20) - 1) / np.sqrt(1 - (1 - c_sigma)**20)) * np.sqrt(dim + 1) - 1
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (len(yk_prev) + 1))) < (hsig + 0.1) * np.sqrt((dim + 1) * (1 - (1 - c_sigma)**2)):
                pass # h = 1
            else:
                h = 0
                c_sigma = 0 # effectively, we just don't update sigma as much? No, standard CMA sets h=0
            
            # Actually, standard CMA logic for h is complex. 
            # A simpler variant: just update sigma based on ps norm.
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (len(yk_prev) + 1))) - np.sqrt(dim)))
            
            # Update pc
            pc = (1 - c_c) * pc + h * np.sqrt(c_c * (2 - c_c) * mueff) * yk / sigma
            
            # Update C
            delta_h = (1 - h) * c_c * (2 - c_c)
            delta_w = c_1 * (2 - c_1)
            # Rank 1 update
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_mu / sigma**2 * np.outer(yk, yk) * 0 + c_1 * np.outer(pc, pc)
            # Wait, the formula is:
            # C = (1 + c1*delta_h - c1 - cmu) * C + c1 * pc*pc^T + cmu * sum(wi * yi*yi^T)
            # The sum term is rank-mu update.
            
            # Let's implement the standard update:
            # C = (1 - c1 - cmu) * C + c1 * pc*pc^T + cmu * (1/sigma^2) * sum(wi * (xi - xmean_old)*(xi - xmean_old)^T)
            # Where yi = (xi - xmean_old) / sigma
            
            # We have yk = (xk - xmean_old) / sigma
            # So the term is sum(wi * yi*yi^T)
            
            # We need to accumulate this in the main loop.
            pass
            
        # Since implementing full CMA-ES with restarts and time guards in a single block is complex,
        # and `cma` might not be available, let's use a simpler but effective approach:
        # **Nelder-Mead** is too local.
        # **Particle Swarm Optimization (PSO)** is robust and fast, but can be slow to converge.
        # **CMA-ES** is best. Let's try to use `scipy.optimize.minimize` with method='Nelder-Mead' or 'Powell' as a fallback? No, they are local.
        # `scipy.optimize.differential_evolution` is global but might be too slow for 500 calls?
        # DE with 500 calls in 5D: pop_size=10, iterations=50. 10*50=500 calls. This is a good global optimizer.
        
        # Let's use Differential Evolution (DE) via scipy if available, otherwise a manual implementation.
        # Manual DE is easy and fast.
        
        # DE Parameters
        pop_size = 15
        F = 0.8  # Mutation scale
        CR = 0.9  # Crossover rate
        
        # Initial population
        # Use a mix of random points and a central point
        pop = np.random.default_rng(0).uniform(-5, 5, (pop_size, self.dim))
        # Ensure one point is near center
        pop[0] = 0.0
        
        # Evaluate initial population
        fit = np.array([func(x) for x in pop])
        total_calls += pop_size
        
        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_f = fit[best_idx]
        
        # DE Loop
        # We have budget - total_calls remaining
        # Each generation costs pop_size evaluations
        max_gens = (self.budget - total_calls) // pop_size
        
        rng = np.random.default_rng(42)
        
        for gen in range(max_gens):
            # Time check
            if time.time() - start_time > timeout:
                break
            
            # Check if we should break due to stall? DE doesn't have a simple stall metric.
            # Just run until budget or time.
            
            for i in range(pop_size):
                # Time check
                if time.time() - start_time > timeout:
                    return best_x, best_f
                
                # Select 3 distinct individuals
                a, b, c = rng.choice(pop_size, 3, replace=False)
                
                # Mutation
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Clip to bounds
                v = np.clip(v, -5, 5)
                
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Evaluate
                ft = func(trial)
                total_calls += 1
                
                # Selection
                if ft < fit[i]:
                    pop[i] = trial
                    fit[i] = ft
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()
                
                # Check budget
                if total_calls >= self.budget:
                    return best_x, best_f
        
        return best_x, best_f
# EVOLVE-BLOCK-END
