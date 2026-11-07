"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    CMA-ES with Adaptive Population Size (CARS).
    
    Strategy:
    1. Start with a moderate population for global exploration on the multimodal Weierstrass function.
    2. As the population converges (spread decreases), adaptively reduce the population size.
    3. This allows the algorithm to transition from a broad search to a fine-grained local search 
       automatically, maximizing the number of function evaluations in the critical convergence phase.
    4. Uses a simple, robust implementation of CMA-ES with step-size adaptation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Safety margin for time (evaluator kills at 330s, we stop at 295s)
        start_time = time.time()
        deadline = start_time + 295.0
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        # Wrapper to count calls, enforce time/budget, and clip
        def wrapped_func(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                return best_f
            if calls_made >= budget:
                return best_f
            
            x = np.clip(x, -5, 5)
            f_val = func(x)
            calls_made += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                
            return f_val

        # Initialize CMA-ES state
        # Start near origin, but with a reasonable sigma to cover the [-4, 4] range quickly
        x_mean = np.zeros(dim)
        sigma = 1.0
        
        # Population size adaptation parameters
        # Start with lambda=15 (good for dim=5 exploration), shrink to lambda=7 (refinement)
        lambda_min = 7
        lambda_max = 15
        current_lambda = lambda_max
        
        # CMA-ES parameters
        mu = max(1, current_lambda // 2)  # Number of recombination weights
        # Weights: log-sum-normal distribution, normalized
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1 / (w ** 2).sum()
        
        # Covariance matrix
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Step size adaptation parameters
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Initial population
        def generate_population():
            nonlocal current_lambda, x_mean, sigma, C
            # Generate standard normal samples
            # Use Cholesky decomposition for correlated samples
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)
            
            # Generate random normals
            Z = np.random.randn(current_lambda, dim)
            # Transform to correlated normals
            Y = (L @ Z.T).T
            
            # Sample points
            pop = x_mean + sigma * Y
            return pop

        # Main loop
        while calls_made < budget and time.time() < deadline:
            # 1. Generate population
            pop = generate_population()
            
            # 2. Evaluate population
            f_vals = np.array([wrapped_func(x) for x in pop])
            
            # Check if we ran out of budget mid-evaluation
            if calls_made >= budget or time.time() > deadline:
                break
                
            # 3. Sort by fitness
            idx = np.argsort(f_vals)
            pop_sorted = pop[idx]
            f_sorted = f_vals[idx]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = pop_sorted[0].copy()
            
            # 4. Update mean (recombination)
            x_new_mean = np.sum(w[:, None] * pop_sorted[:mu], axis=0)
            
            # 5. Update covariance matrix
            y = (x_new_mean - x_mean) / sigma
            pc = (1 - cc) * pc + cc * y
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * damps) * y
            
            # Rank-one update
            C = (1 + cs * damps * (1 - 1 / mu_eff) - cc * (1 - 1 / mu_eff) * (1 - cc) + cs * cc * (1 - 1 / mu_eff)) * C \
                + (1 / mu_eff) * np.outer(y, y)
            # Rank-mu update
            Y = (pop_sorted[:mu] - x_mean) / sigma
            C += (1 - cs * damps * (1 - 1 / mu_eff) + cc * (1 - 1 / mu_eff) * (1 - cc) - cs * cc * (1 - 1 / mu_eff)) * \
                 np.sum(w[:, None] * np.einsum('ij,ik->jk', Y, Y), axis=0)
            
            # Symmetrize and ensure positive definite
            C = (C + C.T) / 2
            # Add small regularization if needed
            eigvals = np.linalg.eigvalsh(C)
            if eigvals.min() < 1e-12:
                C += 1e-12 * np.eye(dim)
            
            # 6. Update step size
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            
            # 7. Update mean
            x_mean = x_new_mean
            
            # 8. Adaptive Population Size (CARS)
            # Calculate spread of the population
            spread = np.std(pop, axis=0).mean()
            # If spread is small, reduce population size to focus on local search
            # Threshold: if spread < 0.1, start shrinking
            if spread < 0.5:
                current_lambda = max(lambda_min, current_lambda - 1)
            elif spread > 2.0:
                current_lambda = min(lambda_max, current_lambda + 1)
            
            # If population is too small, we might want to restart or just continue
            # With CMA-ES, small population is fine for local refinement.
            # We stop if sigma becomes very small
            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
