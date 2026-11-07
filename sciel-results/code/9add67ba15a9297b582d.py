"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(mu+lambda)-ES with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Parameters
        mu = 5
        lam = 10
        
        # Initialize population
        pop = []
        for _ in range(lam):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            pop.append((fx, x))
            
        pop.sort(key=lambda item: item[0])
        best_x = pop[0][1].copy()
        best_f = pop[0][0]
        
        sigma = 1.0
        evals_used = lam
        
        while evals_used < budget:
            if time.time() > deadline:
                break
            
            # Save best parent value for sigma adaptation
            best_parent_f = pop[0][0]
            
            # Generate offspring
            offspring = []
            for i in range(mu):
                if evals_used >= budget:
                    break
                parent = pop[i][1]
                child = parent + sigma * rng.standard_normal(dim)
                child = np.clip(child, -5, 5)
                f_child = func(child)
                offspring.append((f_child, child))
                evals_used += 1
            
            if not offspring:
                break
                
            # Combine parents and offspring
            # We take the top mu parents and all offspring
            candidates = pop[:mu] + offspring
            candidates.sort(key=lambda item: item[0])
            pop = candidates[:lam]
            
            # Update global best
            if pop[0][0] < best_f:
                best_f = pop[0][0]
                best_x = pop[0][1].copy()
                
            # Adapt sigma
            # Check if the best offspring was better than the best parent
            best_offspring_f = offspring[0][0]
            if best_offspring_f < best_parent_f:
                sigma *= 1.2
            else:
                sigma *= 0.8
                
            # Clip sigma to reasonable range
            sigma = np.clip(sigma, 0.01, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
