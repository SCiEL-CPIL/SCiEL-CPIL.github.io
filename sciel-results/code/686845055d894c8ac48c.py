"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution for BBOB f7 (Step Ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        calls = 0
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls += 1

        def _eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Differential Evolution Parameters
        # For step functions, a larger population helps explore basins.
        # F and CR are standard, but we might adapt them.
        pop_size = 20  # Must be >= 4 for DE/rand/1
        F = 0.8
        CR = 0.9

        # Initialize population
        # Spread out to cover the box [-5, 5]^dim
        pop = np.zeros((pop_size, dim))
        pop[0] = best_x
        for i in range(1, pop_size):
            pop[i] = rng.uniform(-5.0, 5.0, size=dim)

        # Evaluate initial population
        fvals = np.full(pop_size, np.inf)
        fvals[0] = best_f
        for i in range(1, pop_size):
            fx = _eval(pop[i])
            if fx is not None:
                fvals[i] = fx
            else:
                break

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Main loop
        while calls < budget and time.time() < deadline - 5.0:
            # Find indices of best and random individuals
            # We need distinct indices for DE/rand/1
            if pop_size < 4:
                break
                
            # Select base and donors
            # We use DE/rand/1: x_i + F * (x_j - x_k)
            # To be efficient, we can pre-shuffle or just pick random indices.
            
            # Sort by fitness to keep track of best
            order = np.argsort(fvals)
            best_idx = order[0]
            
            # If we have stagnated, maybe adapt F? 
            # For now, keep it simple.
            
            for i in range(pop_size):
                if calls >= budget or time.time() > deadline - 5.0:
                    break
                
                # Select j, k != i
                # Simple rejection sampling for small pop_size
                while True:
                    j = rng.integers(0, pop_size)
                    if j != i:
                        break
                while True:
                    k = rng.integers(0, pop_size)
                    if k != i and k != j:
                        break
                
                # Create trial vector
                v = pop[i] + F * (pop[j] - pop[k])
                v = np.clip(v, -5.0, 5.0)
                
                # Evaluate trial
                fx = _eval(v)
                if fx is None:
                    break
                
                # Selection
                if fx <= fvals[i]:
                    pop[i] = v
                    fvals[i] = fx
                    if fx < best_f:
                        best_f = fx
                        best_x = v.copy()

        return best_x, best_f

# EVOLVE-BLOCK-END
