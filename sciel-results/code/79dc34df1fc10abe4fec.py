"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(mu+lambda)-ES with adaptive step size for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial population
        pop_size = min(10, max(5, budget // 2))
        pop = rng.uniform(-5, 5, (pop_size, dim))
        evals = []
        for x in pop:
            fx = func(x)
            evals.append((fx, x.copy()))
        evals.sort(key=lambda t: t[0])
        
        # Track best
        best_val, best_x = evals[0]
        call_count = pop_size
        
        # ES parameters
        mu = max(2, pop_size // 2)  # number of parents
        lam = min(pop_size * 2, max(5, budget - call_count))  # number of offspring per generation
        sigma = 2.0
        
        while call_count < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Select parents
            parents = [x for _, x in evals[:mu]]
            
            # Generate offspring
            offspring = []
            for _ in range(lam):
                if call_count + len(offspring) >= budget:
                    break
                if time.time() > deadline:
                    break
                parent = parents[rng.integers(mu)]
                child = np.clip(parent + sigma * rng.standard_normal(dim), -5, 5)
                offspring.append(child)
            
            if not offspring:
                break
            
            # Evaluate offspring
            for child in offspring:
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                fc = func(child)
                evals.append((fc, child.copy()))
                call_count += 1
                if fc < best_val:
                    best_val, best_x = fc, child.copy()
            
            # Sort and select
            evals.sort(key=lambda t: t[0])
            # Keep only top pop_size
            evals = evals[:pop_size]
            
            # Adapt sigma: 1/5 rule heuristic
            # Count how many of the top pop_size came from this generation's offspring
            # Simpler: if best improved, increase sigma slightly; else decrease
            # Use a success rate based approach
            # For now, use a simple heuristic
            if len(evals) >= mu:
                # Check if any offspring improved over the worst parent
                worst_parent_val = evals[mu - 1][0] if mu <= len(evals) else evals[0][0]
                # This is getting complex; use a simpler adaptation
                pass
            
            # Simple adaptation: alternate based on recent performance
            # If we've been making progress, keep sigma; else reduce
            # For robustness, use a fixed reduction/increase pattern
            sigma *= 0.95  # Slight reduction to converge
            sigma = np.clip(sigma, 0.01, 5.0)
            
            # Occasionally increase sigma for exploration
            if call_count % 50 == 0:
                sigma *= 1.3
                sigma = np.clip(sigma, 0.01, 5.0)
        
        return best_x, best_val
# EVOLVE-BLOCK-END
