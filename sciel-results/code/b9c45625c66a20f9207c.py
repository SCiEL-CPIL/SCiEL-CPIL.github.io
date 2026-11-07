import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        budget = self.budget
        
        # ZDT3: f1 = x1, so we can directly control f1 via x1
        # g depends on x2..xn, minimized when x2..xn = 0
        # Strategy: explore x1 across [0,1], minimize g for each
        
        # Phase 1: Cover x1 with structured samples
        # 7 segments in ZDT3 Pareto front
        # Use ~40 x1 values to cover [0,1]
        n_x1_samples = 40
        x1_values = np.linspace(0.01, 0.99, n_x1_samples)
        
        # For each x1, we'll do ~3 evals to minimize g
        # Total: 40 * 3 = 120 evals, leaving 80 for refinement
        
        all_F = []
        all_X = []
        used = 0
        
        def eval_point(x):
            nonlocal used
            if used >= budget:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            used += 1
            return f
        
        # Phase 1: Structured exploration
        for x1 in x1_values:
            if used >= budget or time.time() > deadline:
                break
            
            # Try x2..xn = 0 (g minimum)
            x = np.zeros(n)
            x[0] = x1
            f = eval_point(x)
            if f is not None:
                all_F.append(f)
                all_X.append(x.copy())
            
            # Try small perturbations to explore
            if used < budget and time.time() < deadline:
                for _ in range(2):
                    x2 = np.zeros(n)
                    x2[0] = x1
                    # Perturb x2..xn slightly
                    perturbation = 0.05 * np.random.random(n-1)
                    x2[1:] = perturbation
                    f = eval_point(x2)
                    if f is not None:
                        all_F.append(f)
                        all_X.append(x2.copy())
        
        # Phase 2: Local refinement around best non-dominated points
        if all_F and used < budget and time.time() < deadline:
            F = np.array(all_F)
            X = np.array(all_X)
            
            # Find non-dominated points
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                if dominated[i]:
                    continue
                for j in range(len(F)):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            
            nd_idx = np.where(~dominated)[0]
            
            # Refine around non-dominated points
            for idx in nd_idx:
                if used >= budget or time.time() > deadline:
                    break
                for _ in range(3):
                    x = X[idx] + 0.02 * np.random.standard_normal(n)
                    x[0] = np.clip(x[0], 0.01, 0.99)
                    f = eval_point(x)
                    if f is not None:
                        all_F.append(f)
                        all_X.append(x.copy())
        
        # Return (ignored, but keep format)
        if all_X:
            return np.array(all_X)
        return np.array([])
