"""CMA-ES Optimizer for BBOB-style functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
    
    Strategy:
    1. Initialize a population of candidates around the current best point.
    2. Sample new candidates from a multivariate normal distribution defined by 
       the current mean, covariance matrix, and step size.
    3. Evaluate all candidates.
    4. Update the mean towards the best half of the candidates.
    5. Update the covariance matrix based on the successful candidates.
    6. Adapt the step size using a cumulative step size adaptation.
    7. Restart if the population collapses (convergence) or stagnates.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 320.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size: lambda = 4 + floor(3 * log(dim)) is standard, but for dim=5, lambda=10 is often robust.
        # Let's use lambda = 10 for better exploration in 5D.
        pop_size = 10 
        # Weights for recombination
        mu = pop_size // 2  # Use best half
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Initial step size
        # The box is [-5, 5]. A good initial sigma is roughly 1/5 of the range? 
        # Or centered at 0 with sigma=2.0 to cover most of the space initially.
        sigma0 = 2.0
        
        # Restart logic
        # If we don't improve for a certain number of generations, or if sigma becomes too small/large, restart.
        max_generations_per_restart = 30
        stagnation_limit = 15 # No improvement in best value for this many generations
        
        # Global best tracking
        best_x_global = np.zeros(dim)
        best_f_global = np.inf
        
        # Main loop
        while budget > 0 and time.time() - start_time < time_limit:
            # Initialize CMA-ES state
            # Start from the current global best if available, otherwise random
            if best_f_global == np.inf:
                mean = np.random.uniform(-4, 4, dim)
            else:
                mean = best_x_global.copy()
            
            sigma = sigma0
            C = np.eye(dim)
            cs = 0.81
            ds = 1.0 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1)
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cmu1 = 2 / ((dim + 1.3)**2 + mueff)
            cmu2 = min(1 - cmu1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            # Check if we have enough budget for at least one generation
            if budget < pop_size + 1:
                # Use remaining budget for random sampling or single point improvements
                # Since CMA-ES needs a population, if budget is too low, just do random search
                while budget > 0:
                    x = np.random.uniform(-5, 5, dim)
                    f = func(x)
                    budget -= 1
                    if f < best_f_global:
                        best_f_global = f
                        best_x_global = x.copy()
                break
            
            # Run CMA-ES until convergence, stagnation, or budget/time limit
            gen = 0
            no_improve_count = 0
            prev_best_f = best_f_global
            
            while budget >= pop_size and time.time() - start_time < time_limit:
                gen += 1
                
                # Check for stagnation
                if no_improve_count > stagnation_limit:
                    break
                
                # Sample population
                # We need to generate pop_size points
                # Standard CMA-ES uses a random matrix B and diagonal D
                # For simplicity and robustness in 5D, we can use Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset to identity
                    C = np.eye(dim)
                    L = np.eye(dim)
                
                # Generate random normal vectors
                # Using np.random.randn is fine
                random_vecs = np.random.randn(pop_size, dim)
                
                # Transform to have covariance C and scale sigma
                # x = mean + sigma * L @ random_vec
                # Note: L is lower triangular, so L @ v is correct for Cholesky
                # Actually, standard is B * D * z where C = B D B^T.
                # If C = L L^T, then x = mean + sigma * L^T @ z ? 
                # Let's stick to: x = mean + sigma * (L @ z) where L is from cholesky of C.
                # Wait, if C = L L^T, then Cov(L z) = L I L^T = C. Correct.
                
                population = mean[None, :] + sigma * (random_vecs @ L.T)
                
                # Clip to bounds
                population = np.clip(population, -5, 5)
                
                # Evaluate population
                fitness = np.zeros(pop_size)
                for i in range(pop_size):
                    if budget <= 0:
                        break
                    fitness[i] = func(population[i])
                    budget -= 1
                    
                # If budget ran out mid-population, break
                if budget < 0:
                    break
                
                # Sort by fitness
                order = np.argsort(fitness)
                # Take the best mu
                selected_indices = order[:mu]
                selected_population = population[selected_indices]
                selected_fitness = fitness[selected_indices]
                
                # Update global best
                best_idx_in_selected = np.argmin(selected_fitness)
                current_best_f = selected_fitness[best_idx_in_selected]
                current_best_x = selected_population[best_idx_in_selected].copy()
                
                if current_best_f < best_f_global:
                    best_f_global = current_best_f
                    best_x_global = current_best_x
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Update mean
                new_mean = np.sum(weights[:, None] * selected_population, axis=0)
                
                # Update step size
                # Cumulative step size adaptation
                # ps = (1-cs) * ps + cs * sqrt(mueff) * (1/sigma) * (new_mean - mean)
                ps = (1 - cs) * ps + cs * np.sqrt(mueff) / sigma * (new_mean - mean)
                
                # hsig condition
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1))) / chiN < 1.4 + 2 / (dim + 1)
                
                # Update covariance matrix
                # pc = (1-cc) * pc + hsig * cc * sqrt(mueff) * (1/sigma) * (new_mean - mean)
                pc = (1 - cc) * pc + hsig * cc * np.sqrt(mueff) / sigma * (new_mean - mean)
                
                # C = (1 - cmu1 - cmu2) * C + cmu1 * outer(ps, ps) + cmu2 * sum(weights_i * outer(xi - mean, xi - mean))
                # Note: The term with ps is for rank-1 update, and the sum is for rank-mu update
                # We use the old mean for the difference
                diff = selected_population - mean
                rank_mu_term = np.sum(weights[:, None, None] * np.einsum('ij,ik->ijk', diff, diff), axis=0)
                
                C = (1 - cmu1 - cmu2) * C + cmu1 * np.outer(ps, ps) + cmu2 * rank_mu_term
                
                # Ensure symmetry and positive semi-definiteness (optional but good for stability)
                C = (C + C.T) / 2
                # If C becomes singular or bad, we might need to regularize, but usually it's fine
                # Check condition number
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    if np.min(eigvals) < 1e-10:
                        C += 1e-10 * np.eye(dim)
                except:
                    C = np.eye(dim)
                
                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))
                
                # Update mean
                mean = new_mean
                
                # Check for convergence
                # If sigma is very small, we are converged
                if sigma < 1e-8:
                    break
                
                # Check for divergence (sigma too large)
                if sigma > 100:
                    break

            # If we broke out of the inner loop, we might want to restart
            # The outer loop will restart CMA-ES with the new global best
            
            # If we didn't improve at all in this restart, maybe the starting point was bad?
            # The restart logic in the outer loop handles this by starting from the global best.
            # If global best didn't change, we are stuck. 
            # To avoid infinite loop of restarting from the same point:
            # If no improvement in the entire restart, perturb the start point slightly?
            # Or just let it run out of budget?
            # Given the budget is 300, and pop_size is 10, we can do ~30 generations.
            # If we stagnate, we break inner loop. Outer loop restarts.
            # If we restart from the same best_x_global, we will get the same result unless random sampling changes.
            # To ensure progress, if no improvement in a restart, we can jitter the start.
            
            if no_improve_count > 0 and gen > 0:
                # If we ran some generations but didn't improve, it might be a local minimum.
                # The next restart will start from the same global best.
                # To escape, we can slightly perturb the global best before the next restart.
                # However, the CMA-ES itself explores around the mean.
                # If it converged to a local min, restarting from that min with fresh C and sigma
                # might just find the same local min.
                # A better strategy: If stagnation occurs, do a "Levy flight" or large random jump?
                # For now, let's just let the outer loop handle it. If it fails, it fails.
                # But to be safe, if we used a lot of budget and didn't improve, we should stop.
                if budget < pop_size * 2:
                    break

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
