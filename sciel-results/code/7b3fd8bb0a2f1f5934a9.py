"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Differential Evolution with Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_used = 0

        def count_func(x):
            nonlocal calls_used
            if calls_used >= budget:
                raise ValueError("Budget exceeded")
            calls_used += 1
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        # Differential Evolution Parameters
        NP = 10  # Population size
        F_min, F_max = 0.5, 0.8
        CR = 0.9  # Crossover rate (not explicitly used in standard DE/rand/1/bin implementation here, 
                  # but we can incorporate it if we do binomial crossover. 
                  # For simplicity and robustness, we use the standard vector update which is equivalent to CR=1.0 
                  # or we can implement CR. Let's stick to standard DE/rand/1/bin with full replacement for simplicity 
                  # as it works well for low dimensions. 
                  # Actually, CR < 1.0 helps exploration. 
                  # Let's implement standard binomial crossover.
        
        # Initialize Population
        P = rng.uniform(-5, 5, (NP, dim))
        F_vals = np.empty(NP)
        for i in range(NP):
            F_vals[i] = count_func(P[i])
        
        best_idx = np.argmin(F_vals)
        best_x = P[best_idx].copy()
        best_f = F_vals[best_idx]
        
        # Determine switch point for exploitation
        switch_call_count = int(budget * 0.8)
        
        # Main DE Loop
        while calls_used < budget:
            if budget - calls_used < NP:
                break
            
            use_best = calls_used >= switch_call_count
            
            for i in range(NP):
                if calls_used >= budget:
                    break
                
                # Select 3 distinct random indices
                candidates = list(range(NP))
                candidates.remove(i)
                a, b, c = rng.choice(candidates, size=3, replace=False)
                
                F_val = rng.uniform(F_min, F_max)
                
                if use_best:
                    # DE/best/1/bin
                    v = P[best_idx] + F_val * (P[b] - P[c])
                else:
                    # DE/rand/1/bin
                    v = P[a] + F_val * (P[b] - P[c])
                
                # Binomial Crossover
                j_rand = rng.integers(0, dim)
                trial = P[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = count_func(trial)
                
                # Selection
                if f_trial <= F_vals[i]:
                    P[i] = trial
                    F_vals[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        best_idx = i
                        
        # Final Polish with Nelder-Mead
        remaining = budget - calls_used
        if remaining > 5:
            def run_nm(start_x, max_iter):
                nonlocal best_x, best_f, calls_used
                try:
                    res = minimize(
                        count_func,
                        start_x,
                        method='Nelder-Mead',
                        options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except ValueError:
                    pass
                except Exception:
                    pass
            
            max_iter = max(1, remaining // 10)
            run_nm(best_x, max_iter)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
