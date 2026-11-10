import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, a, b):
        """True if a dominates b (a <= b in all, a < b in at least one)."""
        return np.all(a <= b) and np.any(a < b)

    def _non_dominated_indices(self, F):
        """Return indices of non-dominated solutions in F (array of shape (N, n_obj))."""
        n = len(F)
        nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd[i]:
                continue
            for j in range(i + 1, n):
                if not nd[j]:
                    continue
                if self._dominates(F[i], F[j]):
                    nd[j] = False
                elif self._dominates(F[j], F[i]):
                    nd[i] = False
                    break
        return np.where(nd)[0]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size for DE
        pop_size = 20
        CR = 0.7
        F_factor = 0.5

        # Use a fixed seed for reproducibility within the optimizer
        # The evaluator sets np.random.seed, but we use our own RandomState
        # to ensure deterministic behavior independent of global state if possible,
        # or just use the global state. Let's use global state to align with evaluator seed.
        # Actually, using np.random directly is safer for "seed 0 and 1" consistency if we don't re-seed.
        
        # Initialize population uniformly
        X = lo + (hi - lo) * np.random.rand(pop_size, dim)
        F_obj = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for final hypervolume calculation
        # The task says "Every returned objective vector is recorded".
        # The evaluator records the return value of __call__? No, "The return value of __call__ is ignored."
        # "Every returned objective vector is recorded" likely means the evaluator intercepts `problem(x)` calls.
        # So we just need to make sure we don't exceed budget.
        
        # Main DE loop with Local Search refinement
        while used < budget:
            if time.time() > deadline:
                break

            # Select parents from non-dominated set for diversity
            nd_idx = self._non_dominated_indices(F_obj)
            
            if len(nd_idx) >= 3:
                parent_pool = X[nd_idx]
            else:
                parent_pool = X

            n_parents = len(parent_pool)
            
            # DE/rand/1/bin mutation
            if n_parents >= 3:
                p1, p2, p3 = np.random.choice(n_parents, size=3, replace=False)
                v = parent_pool[p1] + F_factor * (parent_pool[p2] - parent_pool[p3])
                
                # Binary crossover
                crossover_mask = np.random.rand(dim) < CR
                if not np.any(crossover_mask):
                    j_rand = np.random.randint(dim)
                    crossover_mask[j_rand] = True
                    
                # Pick a target from the pool
                target_idx = np.random.randint(n_parents)
                target = parent_pool[target_idx]
                
                child = np.where(crossover_mask, v, target)
            else:
                # Fallback: random perturbation
                target = parent_pool[np.random.randint(n_parents)]
                child = target + 0.1 * (hi - lo) * np.random.randn(dim)
            
            child = np.clip(child, lo, hi)
            
            # Evaluate child
            fc = problem(child)
            used += 1
            
            # Simple replacement: replace the worst individual in the population
            # "Worst" is defined by the sum of objectives (simple scalarization)
            # This is a heuristic, but effective for small populations.
            worst_idx = np.argmax(np.sum(F_obj, axis=1))
            X[worst_idx] = child
            F_obj[worst_idx] = fc

            # Periodic Local Search Refinement
            # Every 10 evaluations, pick the best non-dominated point and try to improve it
            # via a small random step (1+1 EA style) or coordinate descent.
            # Given the low budget, a simple random walk on the best point is a good structural addition.
            if used % 10 == 0 and used < budget - 5:
                # Find the "best" non-dominated point.
                # We can use the one with the smallest sum of objectives among non-dominated.
                nd_idx_now = self._non_dominated_indices(F_obj)
                if len(nd_idx_now) > 0:
                    # Select the one with minimum sum
                    sums = np.sum(F_obj[nd_idx_now], axis=1)
                    best_local_idx = nd_idx_now[np.argmin(sums)]
                    x_best = X[best_local_idx]
                    f_best = F_obj[best_local_idx]
                    
                    # Try a small perturbation
                    step_size = 0.05 * (hi - lo)
                    x_trial = x_best + np.random.randn(dim) * step_size
                    x_trial = np.clip(x_trial, lo, hi)
                    
                    if used < budget:
                        f_trial = problem(x_trial)
                        used += 1
                        
                        # Accept if it dominates the current best or is non-dominated and better in sum
                        if self._dominates(f_trial, f_best):
                            X[best_local_idx] = x_trial
                            F_obj[best_local_idx] = f_trial
                        elif not self._dominates(f_best, f_trial) and np.sum(f_trial) < np.sum(f_best):
                            # It's non-dominated but not strictly better in all, but better in sum
                            # This might help diversity or convergence depending on the front shape
                            # For now, only accept if it dominates to be safe for hypervolume?
                            # Actually, for hypervolume, we want the front to be as "outer" as possible.
                            # If it's non-dominated, it's on the front.
                            # Let's just keep the current best for now to avoid noise, 
                            # unless the trial is strictly better in at least one objective and not worse in others (dominates).
                            # The `self._dominates` check above handles that.
                            pass

        return X
