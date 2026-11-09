"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Adaptive Simulated Annealing (ASA) for robust global optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit, keep 30s safety margin
        deadline = start_time + 270.0
        
        dim = self.dim
        budget = self.budget
        
        global_best_x = None
        global_best_f = float('inf')
        calls_made = 0
        
        # ASA Parameters
        # Multi-start strategy:
        # 1. Initial phase: Explore with multiple starts and high temperature.
        # 2. Refinement phase: Focus on the best found region with lower temperature.
        
        # We divide the budget into chunks for different restarts.
        # Let's use 5 restarts.
        n_restarts = 5
        budget_per_restart = budget // n_restarts
        
        # ASA specific parameters
        # T0: Initial temperature. Should be large enough to accept uphill moves initially.
        # T0 is often scaled by the range of the function, but we don't know it.
        # For box [-5, 5]^5, a reasonable T0 is around 10-50.
        T0 = 20.0
        # Cooling rate: T_{k+1} = T_k * cooling_rate
        # To reach T_min in N steps: cooling_rate = (T_min / T0)^(1/N)
        T_min = 1e-4
        # Number of iterations per restart
        # We have budget_per_restart calls. 
        # ASA uses 2 calls per iteration (one for x', one for x'' in the direction calculation? 
        # Standard ASA: 
        # 1. Generate x' from x
        # 2. Generate x'' from x
        # 3. Calculate direction
        # 4. Generate x_new from x using direction
        # 5. Accept x_new if better or with prob exp(-delta/T)
        # This is 3 evaluations per step (x', x'', x_new). 
        # Wait, x and x_new are the current and next. 
        # Evaluations: f(x'), f(x''), f(x_new). f(x) is known from previous step.
        # So 3 new function evaluations per ASA step.
        
        # Let's use a simpler variant or adjust budget.
        # Actually, standard ASA requires f(x), f(x'), f(x'').
        # If we have f(x), we generate x', x''.
        # If x' < x'' (assuming minimization), direction is x' - x''.
        # We propose x_new = x + direction * scale? No, ASA uses a specific rule.
        
        # Let's use the standard ASA update rule:
        # x_{k+1} = x_k + (x'_k - x''_k) if f(x'_k) < f(x''_k)
        #         = x_k - (x'_k - x''_k) if f(x'_k) > f(x''_k)
        # Then accept x_{k+1} if f(x_{k+1}) < f(x_k) or with probability exp(-(f(x_{k+1})-f(x_k))/T_k)
        
        # Cost per step: 3 evaluations (x', x'', x_new).
        # Max steps per restart = budget_per_restart // 3
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
            
            # Initialize start point
            # First restart: center (0,0,0,0,0) is a good guess for Griewank-like functions.
            # Subsequent restarts: random points, or points near current best with perturbation.
            if r == 0:
                x = np.zeros(dim)
            else:
                # Start near the current global best to refine, or random for exploration?
                # Mixed strategy: 50% near best, 50% random.
                if global_best_x is not None and np.random.rand() < 0.5:
                    x = global_best_x + np.random.normal(0, 1.0, dim)
                else:
                    x = np.random.uniform(-5, 5, dim)
                
            x = np.clip(x, -5, 5)
            f_x = func(x)
            calls_made += 1
            
            if f_x < global_best_f:
                global_best_f = f_x
                global_best_x = x.copy()
            
            # ASA Loop for this restart
            # Determine number of steps we can afford
            # We need to leave some budget for the next restart? 
            # The loop structure handles this by checking budget at the start of the restart.
            # But we should check inside the ASA loop too.
            
            steps_in_restart = 0
            # Max steps is limited by remaining budget
            remaining_budget = budget - calls_made
            max_steps = remaining_budget // 3
            
            # Cooling schedule for this restart
            # We want to cool from T0 to T_min over max_steps
            if max_steps > 0:
                cooling_rate = (T_min / T0) ** (1.0 / max_steps)
            else:
                cooling_rate = 1.0
                
            T = T0
            
            while calls_made < budget:
                if time.time() > deadline:
                    break
                
                # Generate x' and x''
                # x' = x + random_normal
                # x'' = x + random_normal
                # Scale of random normal? 
                # ASA is scale-invariant to some degree, but the step size matters.
                # The direction is (x' - x''). The magnitude of this difference depends on the noise scale.
                # If noise is too small, we move slowly. If too large, we jump around.
                # Let's use a noise scale that adapts or is fixed.
                # Fixed noise scale of 1.0 is reasonable for [-5, 5] box.
                noise_scale = 1.0
                
                x_prime = x + np.random.normal(0, noise_scale, dim)
                x_double_prime = x + np.random.normal(0, noise_scale, dim)
                
                f_prime = func(x_prime)
                calls_made += 1
                if f_prime < global_best_f:
                    global_best_f = f_prime
                    global_best_x = x_prime.copy()
                    
                if calls_made >= budget:
                    break
                    
                f_double_prime = func(x_double_prime)
                calls_made += 1
                if f_double_prime < global_best_f:
                    global_best_f = f_double_prime
                    global_best_x = x_double_prime.copy()
                    
                if calls_made >= budget:
                    break
                
                # Calculate direction
                if f_prime < f_double_prime:
                    direction = x_prime - x_double_prime
                else:
                    direction = x_double_prime - x_prime
                    
                # Propose new point
                x_new = x + direction
                # Note: Standard ASA doesn't scale the direction, it uses the full difference.
                # This can lead to large steps. 
                # Some variants use a scaling factor. Let's stick to standard for now.
                # However, if x' and x'' are close, the step is small. If far, step is large.
                
                f_new = func(x_new)
                calls_made += 1
                if f_new < global_best_f:
                    global_best_f = f_new
                    global_best_x = x_new.copy()
                    
                # Acceptance criterion
                if f_new < f_x:
                    # Always accept if better
                    x = x_new
                    f_x = f_new
                else:
                    # Metropolis criterion
                    delta = f_new - f_x
                    if delta < 0: # Should not happen if f_new > f_x
                        accept = True
                    else:
                        # Probability of accepting worse solution
                        # Avoid overflow in exp
                        if T > 1e-8:
                            prob = np.exp(-delta / T)
                        else:
                            prob = 0.0
                        accept = np.random.rand() < prob
                        
                    if accept:
                        x = x_new
                        f_x = f_new
                        # If we accept a worse solution, we might want to update global best?
                        # No, global best is tracked separately.
                        
                # Cool down
                T *= cooling_rate
                
                # Optional: If we are stuck (no improvement in many steps), we could restart early?
                # For now, let the temperature handle it.
                
                steps_in_restart += 1
                if steps_in_restart >= max_steps:
                    break
                    
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
