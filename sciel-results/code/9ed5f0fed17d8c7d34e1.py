"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective Optimizer using Multi-Start Weighted Sum with L-BFGS-B."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 290  # 40s safety margin for 330s limit

    def _check_time(self):
        if time.time() > self.deadline:
            return True
        return False

    def _evaluate(self, x, problem):
        self.eval_count += 1
        return problem(x)

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        if n == 0:
            return []
        
        S = [[] for _ in range(n)]
        n_dominates = [0] * n
        rank = [0] * n
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominates[i] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dominates[j] += 1

        front = []
        current_front = [i for i in range(n) if n_dominates[i] == 0]
        for i in current_front:
            rank[i] = len(front)
        front.append(current_front)
        
        k = 0
        while front[k]:
            next_front = []
            for i in front[k]:
                for j in S[i]:
                    n_dominates[j] -= 1
                    if n_dominates[j] == 0:
                        rank[j] = len(front)
                        next_front.append(j)
            if next_front:
                front.append(next_front)
                k += 1
            else:
                break
        
        return rank

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Phase 1: Initial Random Sampling to get a diverse set of points
        # Budget allocation: 100 for random, 100 for local refinement
        n_random = 100
        X = lo + (hi - lo) * np.random.rand(n_random, self.dim)
        F = np.array([self._evaluate(x, problem) for x in X])
        
        if self.eval_count >= self.budget or self._check_time():
            return X

        # Identify non-dominated points from the random sample to use as starts
        ranks = self._fast_non_dominated_sort(F)
        # Select top non-dominated points (rank 0) as starting points for local search
        # If there are too many, we might sample them, but with 100 points, rank 0 is usually small enough
        starts_idx = [i for i in range(n_random) if ranks[i] == 0]
        
        # If no non-dominated points (unlikely), fallback to best few
        if not starts_idx:
            # Sort by sum of objectives
            sums = np.sum(F, axis=1)
            starts_idx = np.argsort(sums)[:10]
        else:
            # Limit number of starts to conserve budget for refinement
            # We have ~100 budget left. L-BFGS-B might use 10-20 evals per run.
            # Let's aim for 5-10 starts.
            if len(starts_idx) > 8:
                 # Randomly sample 8 from the non-dominated set
                 rng = np.random.default_rng(0)
                 starts_idx = rng.choice(starts_idx, size=8, replace=False)
            
        # Phase 2: Local Refinement using Weighted Sum Scalarization
        # Generate diverse weight vectors
        n_weights = len(starts_idx)
        
        for i, start_idx in enumerate(starts_idx):
            if self.eval_count >= self.budget or self._check_time():
                break
                
            x_start = X[start_idx].copy()
            
            # Create a weight vector. To ensure coverage, we can use random weights
            # or deterministic ones. Random is fine given the diverse starts.
            w = np.random.rand(self.n_obj)
            w /= np.sum(w)
            
            def objective(x):
                f = problem(x)
                self.eval_count += 1
                return np.dot(w, f)
                
            # Define bounds for L-BFGS-B
            lb = lo
            ub = hi
            
            try:
                res = minimize(
                    objective, 
                    x_start, 
                    method='L-BFGS-B', 
                    bounds=list(zip(lb, ub)),
                    options={'maxiter': 50, 'maxfun': 20} # Limit function evaluations per local search
                )
                if res.success or res.fun is not None:
                    X = np.vstack([X, [res.x]])
                    F = np.vstack([F, [problem(res.x)]])
                    self.eval_count += 1 # Account for the final evaluation if not counted in minimize? 
                    # Actually, minimize calls objective which counts. 
                    # But we need the vector F. The last call in minimize returned a scalar.
                    # We need to evaluate res.x again to get the vector for F, OR 
                    # we can just trust that the last few points are good.
                    # Wait, `objective` counts evals. `problem(res.x)` adds 1 more.
                    # This is safe.
            except Exception:
                pass

        return X
# EVOLVE-BLOCK-END
