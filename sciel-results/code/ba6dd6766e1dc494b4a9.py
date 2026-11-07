"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using adaptive evolutionary search with LHS initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        
        # Latin Hypercube Sampling for initial population
        def lhs(n, dim):
            """Generate Latin Hypercube samples in [0,1]^dim."""
            samples = np.zeros((n, dim))
            for d in range(dim):
                perm = np.arange(n)
                np.random.shuffle(perm)
                samples[:, d] = (perm + np.random.random(n)) / n
            return samples
        
        # Initial population: use LHS for better coverage
        pop_size = min(40, budget // 5)
        if pop_size < 10:
            pop_size = 10
            
        X = lo + (hi - lo) * lhs(pop_size, dim)
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Check time after initial evaluation
        if time.time() > deadline:
            return X
        
        # Adaptive evolutionary loop
        # Step size starts large and decreases
        base_step = 0.3  # Initial step size as fraction of range
        decay_rate = 0.98  # Step size decay per generation
        
        step = base_step
        gen = 0
        
        while used < budget and time.time() < deadline:
            gen += 1
            
            # Find non-dominated points
            non_dom_mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        non_dom_mask[i] = False
                        break
            
            # Select parents from non-dominated set (or all if none found)
            if np.any(non_dom_mask):
                parents_idx = np.where(non_dom_mask)[0]
            else:
                parents_idx = np.arange(len(F))
            
            # Generate children with adaptive mutation
            n_children = min(budget - used, max(1, pop_size // 2))
            
            for _ in range(n_children):
                if time.time() > deadline:
                    break
                if used >= budget:
                    break
                    
                # Select parent
                parent_idx = parents_idx[np.random.randint(len(parents_idx))]
                parent = X[parent_idx]
                
                # Adaptive mutation: Gaussian with decreasing step size
                mutation = np.random.randn(dim) * step * (hi - lo)
                child = parent + mutation
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                
                # Replace worst point (by sum of objectives)
                worst_idx = np.argmax(np.sum(F, axis=1))
                X[worst_idx] = child
                F[worst_idx] = fc
            
            # Decay step size
            step *= decay_rate
            
            # Periodically re-evaluate non-dominated set for better parent selection
            if gen % 5 == 0:
                # Keep only non-dominated points, fill rest with random
                non_dom_mask = np.ones(len(F), dtype=bool)
                for i in range(len(F)):
                    for j in range(len(F)):
                        if i == j:
                            continue
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            non_dom_mask[i] = False
                            break
                
                n_non_dom = np.sum(non_dom_mask)
                if n_non_dom < pop_size // 2:
                    # Add some random points to maintain diversity
                    n_add = pop_size // 2 - n_non_dom
                    for _ in range(n_add):
                        if used >= budget or time.time() > deadline:
                            break
                        rand_idx = np.random.randint(len(F))
                        rand_point = lo + (hi - lo) * np.random.random(dim)
                        rand_f = problem(rand_point)
                        used += 1
                        X[rand_idx] = rand_point
                        F[rand_idx] = rand_f
        
        return X
# EVOLVE-BLOCK-END
