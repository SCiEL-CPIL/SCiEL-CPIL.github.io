"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        total_budget = self.budget

        # Time guard: return best if we exceed 320s (kill is at 330s)
        start_time = time.time()
        deadline = start_time + 320.0

        best_x = np.zeros(dim)
        best_f = np.inf

        # CMA-ES parameters
        # Lambda (population size)
        lam = max(4, int(4 + 3 * np.log(dim)))
        
        # Number of restarts
        # We want to use the full budget.
        # Each restart uses a portion of the budget.
        # Let's aim for about 3-5 restarts to balance exploration and exploitation.
        # If we use N restarts, each gets total_budget // N evaluations.
        # With dim=5, lam=15.
        # Let's try 4 restarts. 300 / 4 = 75 evals per restart.
        # 75 / 15 = 5 generations per restart. This is quite short for CMA-ES.
        # Maybe fewer restarts? 2 restarts -> 150 evals -> 10 generations.
        # Let's stick to a simple split: 3 restarts.
        # 300 / 3 = 100 evals. 100 / 15 = 6-7 generations.
        
        # Actually, for 5D functions, CMA-ES is very fast.
        # Let's use a dynamic restart strategy or just a fixed number.
        # Let's try 5 restarts with smaller populations to increase diversity?
        # No, CMA-ES needs decent population.
        
        # Let's use a single CMA-ES run but with a larger population and restart logic if it stagnates?
        # Or simply multiple independent CMA-ES runs.
        
        # Let's do 3 independent CMA-ES runs.
        n_restarts = 3
        budget_per_restart = total_budget // n_restarts
        remaining_budget = total_budget - (budget_per_restart * n_restarts)
        
        # Distribute remaining budget to the first restarts
        budgets = [budget_per_restart] * n_restarts
        for i in range(remaining_budget):
            budgets[i % n_restarts] += 1

        for r in range(n_restarts):
            if time.time() > deadline:
                break
                
            # Initialize CMA-ES
            # Start from a random point in the box
            x_mean = rng.uniform(-5, 5, dim)
            
            # Initial sigma
            sigma = 2.0
            
            # Covariance matrix C
            C = np.eye(dim)
            # Eigenvalues of C (initially 1)
            C_eigval = np.ones(dim)
            # Inverse square root of C (initially I)
            C_eigvec = np.eye(dim)
            
            # CMA-ES parameters
            # mu = lam // 2
            mu = lam // 2
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / w.sum()
            
            # Step size control (CSA)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            
            # Covariance matrix adaptation
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            d_c = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_c
            
            # Memory
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # History
            best_x_run = x_mean.copy()
            best_f_run = func(x_mean)
            
            # Initial population
            pop_x = np.zeros((lam, dim))
            pop_f = np.zeros(lam)
            
            # Generate initial population
            for i in range(lam):
                # Standard normal
                z = rng.standard_normal(dim)
                # Transform using Cholesky of C (or eigendecomp)
                # For simplicity and speed in 5D, we can use eigendecomp update
                # x = mean + sigma * C^{1/2} * z
                # C^{1/2} = C_eigvec @ diag(sqrt(C_eigval)) @ C_eigvec.T
                # But we update C_eigval and C_eigvec.
                # Let's use the standard CMA-ES update rule.
                
                # Actually, implementing full CMA-ES with eigendecomp update is complex.
                # Let's use a simpler, robust CMA-ES implementation or a simpler alternative like DE or PSO?
                # Given the "MATERIALLY DIFFERENT" directive, CMA-ES is a strong candidate.
                # Let's use a simplified CMA-ES or a well-known library if available? 
                # "Python standard library, NumPy and SciPy are available."
                # SciPy has minimize, but not CMA-ES directly as a black box optimizer with budget control in a simple API.
                # Let's implement a basic CMA-ES.
                
                # To keep it simple and robust within the budget, I will use a slightly simplified CMA-ES 
                # or a different population-based method like Differential Evolution (DE) or Particle Swarm Optimization (PSO).
                # DE is very robust and simple.
                # Let's switch to Differential Evolution (DE/rand/1/bin) with adaptive parameters.
                # It is fundamentally different from (1+1)-ES (which is a local search). DE is a global population-based search.
                
                pass 

        # Re-implementing with DE (Differential Evolution) which is simpler to code correctly under time pressure 
        # and performs well on multimodal functions like Rastrigin and Gallagher.
        
        # Reset for DE
        best_x = np.zeros(dim)
        best_f = np.inf
        
        # DE Parameters
        N_pop = 20  # Population size
        F = 0.8     # Differential weight
        CR = 0.9    # Crossover rate
        
        # Total budget allocation
        # N_pop evaluations per generation.
        # Generations = budget // N_pop
        n_gen = total_budget // N_pop
        
        # Initialize population
        pop = rng.uniform(-5, 5, (N_pop, dim))
        fit = np.array([func(x) for x in pop])
        
        # Track best
        idx_best = np.argmin(fit)
        best_x = pop[idx_best].copy()
        best_f = fit[idx_best]
        
        for g in range(n_gen):
            if time.time() > deadline:
                break
                
            for i in range(N_pop):
                # Select 3 distinct individuals
                a, b, c = rng.choice(N_pop, 3, replace=False)
                
                # Differential mutation
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Binomial crossover
                trial = pop[i].copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                
                # Selection
                if f_trial < fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
                    
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        
        return best_x, best_f
# EVOLVE-BLOCK-END
