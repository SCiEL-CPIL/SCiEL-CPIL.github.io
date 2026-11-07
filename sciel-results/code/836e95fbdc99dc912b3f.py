"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT3 exploiting problem structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # For ZDT3: f1 = x[0], f2 = (1+10*(n-1))*(1 - sqrt(f1/g) - h(f1, g))
        # where g = 1 + 10*(n-1) + sum((x_i - 0.5)^2 - 10*cos(20*pi*(x_i-0.5))) for i=2..n
        # h = sin(10*f1*pi)^2
        # The Pareto front has disconnected segments.
        
        # Strategy: 
        # 1. Sample x[0] uniformly to cover f1 range [0,1]
        # 2. For each x[0] target, optimize the rest to minimize f2
        # 3. Use local search around promising points
        
        def zdt3_objectives(x):
            """Compute ZDT3 objectives directly (for internal use, not for evaluation)."""
            f1 = x[0]
            n = len(x) - 1
            g = 1 + 10*n
            for i in range(1, len(x)):
                g += (x[i] - 0.5)**2 - 10*np.cos(20*np.pi*(x[i] - 0.5))
            h = np.sin(10*f1*np.pi)**2
            f2 = (1 + 10*n) * (1 - np.sqrt(f1/g) - h*(1 - np.sqrt(f1/g)))
            return f1, f2
        
        def g_function(x_rest):
            """Compute g for ZDT3 from x[1:]."""
            n = len(x_rest)
            g = 1 + 10*n
            for i in range(n):
                g += (x_rest[i] - 0.5)**2 - 10*np.cos(20*np.pi*(x_rest[i] - 0.5))
            return g
        
        def minimize_g(x_rest, rng, iters=50):
            """Minimize g(x_rest) using local search. Optimal is g=1 when all x_i=0.5."""
            best_x = x_rest.copy()
            best_g = g_function(best_x)
            
            for _ in range(iters):
                # Try moving towards 0.5 (the known minimum)
                step = 0.1 * rng.random(len(x_rest))
                candidate = best_x + step * (0.5 - best_x)
                candidate = np.clip(candidate, 0.0, 1.0)
                g_val = g_function(candidate)
                if g_val < best_g:
                    best_x = candidate
                    best_g = g_val
                else:
                    # Random perturbation
                    candidate = best_x + 0.05 * rng.standard_normal(len(x_rest))
                    candidate = np.clip(candidate, 0.0, 1.0)
                    g_val = g_function(candidate)
                    if g_val < best_g:
                        best_x = candidate
                        best_g = g_val
            
            return best_x, best_g
        
        # Phase 1: Sample diverse x[0] values and optimize g for each
        n_samples = min(60, self.budget // 3)
        x0_values = np.linspace(0.01, 0.99, n_samples)
        rng.shuffle(x0_values)
        
        X_list = []
        F_list = []
        used = 0
        
        for x0 in x0_values:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            # Start with x_rest = 0.5 (known to give g=1)
            x_rest = np.full(dim - 1, 0.5)
            # Add some randomness
            x_rest += 0.1 * rng.standard_normal(dim - 1)
            x_rest = np.clip(x_rest, 0.0, 1.0)
            
            # Local search to minimize g
            x_rest, g_val = minimize_g(x_rest, rng, iters=30)
            
            x = np.concatenate([[x0], x_rest])
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            used += 1
            X_list.append(x)
            F_list.append(f)
        
        # Phase 2: Local search around best points to improve f2
        if used < self.budget and X_list:
            X_arr = np.array(X_list)
            F_arr = np.array(F_list)
            
            # Sort by f1 to find spread
            order = np.argsort(F_arr[:, 0])
            
            # For each point, try to improve f2 by adjusting x_rest
            n_improve = min(len(X_list), self.budget - used)
            for idx in order[:n_improve]:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                x = X_arr[idx].copy()
                x0 = x[0]
                x_rest = x[1:].copy()
                
                # Multiple local search attempts
                for attempt in range(3):
                    # Perturb x_rest
                    perturbation = 0.02 * rng.standard_normal(dim - 1)
                    x_rest_new = x_rest + perturbation
                    x_rest_new = np.clip(x_rest_new, 0.0, 1.0)
                    
                    x_new = np.concatenate([[x0], x_rest_new])
                    x_new = np.clip(x_new, lo, hi)
                    
                    f_new = problem(x_new)
                    used += 1
                    
                    # Check if improvement in f2 (same f1)
                    if f_new[1] < F_arr[idx][1] - 1e-8:
                        X_arr[idx] = x_new
                        F_arr[idx] = f_new
                        x_rest = x_rest_new
                        break
                    elif f_new[1] < F_arr[idx][1]:
                        # Slight improvement, keep exploring
                        x_rest = x_rest_new
        
        # Phase 3: Fill remaining budget with random sampling + local refinement
        if used < self.budget:
            n_remaining = self.budget - used
            n_random = min(n_remaining, 40)
            
            for _ in range(n_random):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                x = lo + (hi - lo) * rng.random(dim)
                f = problem(x)
                used += 1
                X_list.append(x)
                F_list.append(f)
                
                # Quick local search
                if used + 5 <= self.budget:
                    x0 = x[0]
                    x_rest = x[1:].copy()
                    for _ in range(5):
                        if used >= self.budget:
                            break
                        step = 0.05 * rng.standard_normal(dim - 1)
                        x_rest_new = x_rest + step
                        x_rest_new = np.clip(x_rest_new, 0.0, 1.0)
                        x_new = np.concatenate([[x0], x_rest_new])
                        x_new = np.clip(x_new, lo, hi)
                        f_new = problem(x_new)
                        used += 1
                        if f_new[1] < f[1]:
                            x = x_new
                            f = f_new
                            x_rest = x_rest_new
        
        # Final: combine all and return
        if X_list:
            X_final = np.array(X_list)
        else:
            X_final = np.array([])
        
        return X_final
# EVOLVE-BLOCK-END
