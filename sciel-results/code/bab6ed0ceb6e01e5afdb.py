"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simple evolutionary multi-objective search: random init, mutate non-dominated parents."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        pop = int(min(20, self.budget // 4))
        X = lo + (hi - lo) * rng.random((pop, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop
        while used < self.budget and (time.time() - start_time) < deadline:
            # Find non-dominated set
            dominated = np.zeros(pop, dtype=bool)
            for i in range(pop):
                for j in range(pop):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            parents_idx = np.where(~dominated)[0]
            if len(parents_idx) == 0:
                parents_idx = np.arange(pop)
            
            # Select parent from non-dominated set with some diversity
            parent_idx = parents_idx[rng.integers(len(parents_idx))]
            
            # Adaptive mutation step
            step_size = 0.05 + 0.05 * (used / self.budget)
            child = X[parent_idx] + step_size * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            
            # Replace worst point based on sum of objectives
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst], F[worst] = child, fc
        
        # Final polish: try small perturbations on best points
        if (time.time() - start_time) < deadline - 5:
            dominated = np.zeros(pop, dtype=bool)
            for i in range(pop):
                for j in range(pop):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            best_idx = np.where(~dominated)[0]
            if len(best_idx) == 0:
                best_idx = [int(np.argmin(np.sum(F, axis=1)))]
            
            for idx in best_idx[:3]:  # polish top 3 non-dominated points
                if used >= self.budget or (time.time() - start_time) >= deadline - 2:
                    break
                for _ in range(5):
                    if used >= self.budget or (time.time() - start_time) >= deadline - 1:
                        break
                    child = X[idx] + 0.01 * (hi - lo) * rng.standard_normal(self.dim)
                    child = np.clip(child, lo, hi)
                    fc = problem(child)
                    used += 1
                    if np.all(fc <= F[idx]):
                        X[idx], F[idx] = child, fc
                    else:
                        # Try another random perturbation
                        pass
        return X
# EVOLVE-BLOCK-END
