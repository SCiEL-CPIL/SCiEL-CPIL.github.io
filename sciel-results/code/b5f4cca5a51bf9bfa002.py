<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize


class Optimizer:
    """
    Hybrid optimizer: Sobol-based global sampling followed by Nelder-Mead local refinement.
    Leverages the unimodal nature of the Bent Cigar function for rapid convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # 1. Deterministic Sobol sampling for initial exploration
        # Use a moderate number of points for good coverage without exhausting budget
        n_sobol = min(100, budget // 2)
        
        # Generate Sobol points
        sampler = qmc.Sobol(d=dim, scramble=True)
        # We skip the first point (0,0,...,0) or just map [0,1] to [-5,5].
        # Mapping to [-5, 5] covers the entire search space.
        sobol_points = sampler.random(n_sobol)
        sobol_points = -5 + 10 * sobol_points
        
        best_x = sobol_points[0].copy()
        best_f = func(best_x)
        count = 1
        
        for i in range(1, n_sobol):
            if time.time() > deadline:
                break
            x = sobol_points[i]
            f = func(x)
            count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # 2. Local refinement with Nelder-Mead
        remaining_budget = budget - count
        if remaining_budget <= 0 or time.time() > deadline:
            return best_x, best_f
            
        # Wrapper to track calls and enforce budget/time
        call_count = [0]
        
        def wrapped_func(x):
            call_count[0] += 1
            if call_count[0] > remaining_budget or time.time() > deadline:
                # Return a large value to discourage further exploration in this direction
                # or raise an exception. Returning large value is safer for scipy.
                return 1e100
            return func(x)
        
        try:
            # Nelder-Mead options
            # maxiter: max number of iterations. Each iter is ~n+1 evals.
            # maxfev: max number of function evaluations.
            options = {
                'maxiter': 1000,
                'maxfev': remaining_budget,
                'xatol': 1e-6,
                'fatol': 1e-8
            }
            
            result = minimize(wrapped_func, best_x, method='Nelder-Mead', options=options)
            
            # Update best if result is better
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x.copy()
                
        except Exception:
            # If minimize crashes, we still have the best_x from Sobol
            pass
            
        return best_x, best_f
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES style optimizer with covariance adaptation for anisotropic landscapes.
    Uses (mu, lambda) selection with adaptive step size and covariance matrix
    to handle the Bent Cigar function's narrow valley structure.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size: 4*(4 + log(dim)) for dim=5 gives ~4*7=28, use 30
        lambda_pop = 30
        mu = lambda_pop // 2  # Selection size
        
        # Initial step size
        sigma = 2.0  # Start with reasonable step size for [-5,5]
        
        # Initial mean (center of search space)
        mean = np.zeros(dim)
        
        # Initial covariance (identity)
        C = np.eye(dim)
        
        # Eigendecomposition for sampling
        # We'll maintain C in eigenbasis for efficiency
        # Actually, let's just sample directly from C using Cholesky or eigen
        # For dim=5, we can afford to do eigendecomp each generation
        
        # Weighted recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 - mu_eff / (dim + 1)) / (dim + 4)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / (dim + 1)) / ((dim + 2)**2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        d_sigma = 1 + 0.35 * (1 + dim / 3)
        
        # Track best
        best_x = mean.copy()
        best_f = func(best_x)
        count = 1
        
        # History for covariance update
        # We'll use the standard CMA-ES update rule
        
        generation = 0
        while count < budget and time.time() < deadline:
            generation += 1
            if time.time() > deadline:
                break
            
            # Sample lambda points
            # Decompose C: C = B * D * B^T
            try:
                D, B = np.linalg.eigh(C)
                # Ensure positive eigenvalues
                D = np.maximum(D, 1e-20)
                sqrt_D = np.sqrt(D)
            except:
                # Fallback to identity
                B = np.eye(dim)
                sqrt_D = np.ones(dim)
            
            # Sample
            offspring = []
            for i in range(lambda_pop):
                if time.time() > deadline:
                    break
                # Sample from N(0, C)
                z = np.random.randn(dim)
                # Transform: x = mean + sigma * B * (sqrt(D) * z)
                y = B @ (sqrt_D * z)
                x = mean + sigma * y
                offspring.append((x, y))
            
            if len(offspring) == 0:
                break
            
            # Evaluate offspring
            offspring_f = []
            for x, y in offspring:
                if time.time() > deadline:
                    break
                f = func(x)
                count += 1
                offspring_f.append((f, x, y))
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            if len(offspring_f) < mu:
                break
            
            # Sort by fitness
            offspring_f.sort(key=lambda t: t[0])
            
            # Select top mu
            selected = offspring_f[:mu]
            
            # Update mean (weighted recombination)
            new_mean = np.zeros(dim)
            for i, (f, x, y) in enumerate(selected):
                new_mean += weights[i] * x
            mean = new_mean
            
            # Update p_sigma
            # Need to transform p_sigma to eigenbasis
            # p_sigma += (1/c_sigma) * (1/sigma) * (mean - old_mean) in eigenbasis
            # Actually, standard CMA-ES: p_sigma = (1-c_sigma)*p_sigma + c_sigma * (1/sigma) * B^T * (mean - old_mean)
            # But we need to be careful with the transformation
            
            # Let's use a simpler approach: maintain p_sigma in original space
            # p_sigma = (1-c_sigma)*p_sigma + c_sigma * (1/sigma) * (mean - old_mean)
            old_mean = mean - np.zeros(dim)  # This is wrong, need to save old mean
            # Actually, I made an error. Let me restructure.
            
            # Let me redo this more carefully
            break  # Skip this generation if the logic is getting too complex
            
        return best_x, best_f
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I made an error in the implementation. Let me provide a cleaner, correct CMA-ES implementation:

