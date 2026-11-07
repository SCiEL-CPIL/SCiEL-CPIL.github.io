"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Simulated Annealing for BBOB-style Schwefel function on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Initialize at center
        x = np.zeros(dim)
        try:
            f_x = func(x)
        except Exception:
            f_x = np.inf
        evals = 1
        
        best_x = x.copy()
        best_f = f_x
        
        # Simulated Annealing parameters
        # Geometric cooling schedule
        T_initial = 10.0
        T_final = 0.001
        # Number of iterations per temperature level
        # We have budget - 1 remaining evaluations
        remaining = budget - 1
        # Use a geometric progression for temperature
        # T_k = T_initial * (T_final / T_initial)^(k / N)
        # We'll iterate until budget exhausted or time runs out
        
        # Adaptive step size: start large, shrink over time
        # Step size scales with temperature
        T = T_initial
        # Cooling factor per iteration: T_new = T * alpha
        # Choose alpha so that after 'remaining' iterations, T ~= T_final
        # alpha = (T_final / T_initial)^(1 / remaining)
        alpha = (T_final / T_initial) ** (1.0 / max(1, remaining))
        
        while evals < budget and time.time() < deadline:
            # Step size proportional to current temperature
            # Base step size: when T=T_initial, step ~ 4.0 (covers most of [-5,5])
            # When T=T_final, step ~ 0.001
            step_size = 4.0 * T / T_initial
            
            # Generate random perturbation
            z = rng.standard_normal(dim)
            x_trial = x + step_size * z
            np.clip(x_trial, -5, 5, out=x_trial)
            
            try:
                f_trial = func(x_trial)
            except Exception:
                f_trial = np.inf
            evals += 1
            
            # Acceptance criterion
            if f_trial < f_x:
                # Always accept better solutions
                x = x_trial
                f_x = f_trial
                
                if f_x < best_f:
                    best_f = f_x
                    best_x = x.copy()
            else:
                # Accept worse solutions with probability exp(-delta_f / T)
                delta_f = f_trial - f_x
                if delta_f > 0 and T > 1e-10:
                    # Avoid overflow in exp
                    if delta_f / T < 500:  # exp(-500) ~ 1e-217, negligible
                        accept_prob = np.exp(-delta_f / T)
                        if rng.random() < accept_prob:
                            x = x_trial
                            f_x = f_trial
            
            # Cool down
            T *= alpha
            # Ensure T doesn't go below T_final
            if T < T_final:
                T = T_final
        
        return best_x, best_f
# EVOLVE-BLOCK-END
