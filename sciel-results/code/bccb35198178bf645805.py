import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        
        call_count = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            x = np.asarray(x, dtype=float).flatten()
            if x.shape[0] != dim:
                x = np.resize(x, dim)
            x = np.clip(x, -5.0, 5.0)
            try:
                val = func(x)
                if not np.isfinite(val):
                    return np.inf
                return float(val)
            except Exception:
                return np.inf

        def time_left():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Phase 1: Differential Evolution (DE)
        # DE is effective for multimodal functions like Rastrigin.
        n_particles = 15
        F = 0.8  # Mutation scale factor
        CR = 0.9 # Crossover rate
        
        # Initialize population
        X = rng.uniform(-5, 5, (n_particles, dim))
        F_vals = np.array([safe_func(p) for p in X])
        
        # Initialize bests
        min_idx = np.argmin(F_vals)
        G_best = X[min_idx].copy()
        G_best_f = F_vals[min_idx]
        
        if G_best_f < best_f:
            best_f = G_best_f
            best_x = G_best.copy()

        # DE Iterations
        # Each iteration costs n_particles evaluations
        while call_count < budget - n_particles and time_left():
            # Check if we have enough budget for a full generation
            if call_count + n_particles > budget:
                break
                
            new_F_vals = F_vals.copy()
            new_X = X.copy()
            
            for i in range(n_particles):
                if call_count + 1 > budget:
                    break
                    
                # Select distinct indices for mutation
                # Ensure a, b, c are distinct and not i
                idxs = rng.choice(n_particles, 3, replace=False)
                while idxs[0] == i or idxs[1] == i or idxs[2] == i:
                    idxs = rng.choice(n_particles, 3, replace=False)
                
                a, b, c = X[idxs[0]], X[idxs[1]], X[idxs[2]]
                
                # Mutation: v = a + F * (b - c)
                v = a + F * (b - c)
                
                # Crossover
                j_rand = rng.integers(dim)
                u = rng.random(dim) < CR
                u[j_rand] = True
                trial = np.where(u, v, X[i])
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                f_trial = safe_func(trial)
                
                # Selection
                if f_trial <= new_F_vals[i]:
                    new_X[i] = trial
                    new_F_vals[i] = f_trial
                    if f_trial < G_best_f:
                        G_best_f = f_trial
                        G_best = trial.copy()
                        if G_best_f < best_f:
                            best_f = G_best_f
                            best_x = G_best.copy()
                
            X = new_X
            F_vals = new_F_vals

        # Phase 2: Local Refinement with (1+1)-ES
        if best_x is not None and time_left() and call_count < budget - 1:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            remaining = budget - 1 - call_count
            
            for _ in range(remaining):
                if not time_left() or call_count >= budget - 1:
                    break
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = safe_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.85
                
                sigma = np.clip(sigma, 0.001, 2.0)

        return best_x, best_f
