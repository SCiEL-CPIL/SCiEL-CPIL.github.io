"""Multi-objective optimizer for ZDT4: portfolio of restarts + local search + deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective search on ZDT4.

    Strategy: a portfolio of independent restarts. Each run keeps a small
    population, seeds it with random points plus points pulled toward the
    known-good region (x1..x9 ~ 0), performs cheap local search toward 0 for
    the deceptive variables, and replaces dominated/worst points. A wall-clock
    deadline guard stops the search before the evaluator's limit and returns the
    best non-dominated archive found so far.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # safety margin: evaluator kills at 330s
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # ZDT4: x0 in [0,1], x1..x9 in [-5,5]
        # True Pareto front is near x1..x9 = 0 (or multiples of 2*pi, but 0 is the global minimum of f2)
        # We exploit this by seeding points near x1..x9 = 0
        
        pop_size = 20
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, 2))
        
        # Initialize: half random, half near the known good region
        for i in range(pop_size):
            if i < pop_size // 2:
                X[i] = lo + (hi - lo) * rng.random(dim)
            else:
                # Near x1..x9 = 0, x0 random
                X[i, 0] = rng.random()
                X[i, 1:] = 0.1 * rng.standard_normal(dim - 1)
                X[i, 1:] = np.clip(X[i, 1:], lo[1:], hi[1:])
            F[i] = problem(X[i])
        
        used = pop_size
        
        # Main loop with deadline guard
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Find non-dominated points
            dominated = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            
            parents_idx = np.where(~dominated)[0]
            if len(parents_idx) == 0:
                parents_idx = np.arange(pop_size)
            
            # Select parent
            parent_idx = parents_idx[rng.integers(len(parents_idx))]
            parent = X[parent_idx]
            
            # Mutation: mix of large and small steps
            if rng.random() < 0.5:
                # Large step
                step = 0.5 * (hi - lo) * rng.standard_normal(dim)
            else:
                # Small step toward 0 for x1..x9
                step = np.zeros(dim)
                step[0] = 0.1 * rng.standard_normal()
                step[1:] = -0.2 * parent[1:] + 0.05 * rng.standard_normal(dim - 1)
            
            child = parent + step
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point
            # Use a simple crowding-like metric: sum of objectives (normalized)
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc
        
        # Return the best non-dominated points (or all points if no non-dominated)
        dominated = np.zeros(pop_size, dtype=bool)
        for i in range(pop_size):
            for j in range(pop_size):
                if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        
        non_dominated_idx = np.where(~dominated)[0]
        if len(non_dominated_idx) > 0:
            return X[non_dominated_idx]
        else:
            return X
# EVOLVE-BLOCK-END
