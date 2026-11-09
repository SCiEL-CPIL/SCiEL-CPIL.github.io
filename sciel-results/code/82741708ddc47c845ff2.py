"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid SPSA + (1+1)-ES for BBOB-style f14 (Different powers).
    
    Strategy:
    1. Phase 1: SPSA (Simultaneous Perturbation Stochastic Approximation) 
       to leverage gradient information with 2 evaluations per iteration.
       This is efficient for smooth, unimodal functions like f14.
    2. Phase 2: Switch to (1+1)-ES with adaptive step size for fine-tuning
       once SPSA steps become small or stagnation is detected.
    
    Key features:
    - Starts at origin (good initial guess for BBOB with shifts).
    - Time-decaying step sizes for SPSA.
    - Adaptive sigma for ES phase.
    - Time guard for safety.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Time management
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        # RNG setup
        rng = np.random.default_rng(42)
        
        # Initialize at origin
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1
        
        # SPSA parameters
        # a_k = a / (k + alpha)^(ca)
        # c_k = c / (k + gamma)^(cg)
        a = 0.1
        c = 0.5
        alpha = 0.602
        ca = 0.801
        gamma = 0.101
        cg = 0.602
        
        # Phase control
        spsa_iterations = 0
        max_spsa_iterations = 50  # Switch to ES after some SPSA iterations
        spsa_stagnation = 0
        spsa_stagnation_limit = 10
        
        # ES parameters
        sigma = 1.0
        successes = []
        window_size = 10
        
        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Decide which phase to use
            if spsa_iterations < max_spsa_iterations and spsa_stagnation < spsa_stagnation_limit:
                # SPSA phase
                k = spsa_iterations
                a_k = a / (k + alpha) ** ca
                c_k = c / (k + gamma) ** cg
                
                # Generate random direction delta in {-1, 1}^n
                delta = rng.choice([-1, 1], size=n)
                
                # Evaluate at x + c_k * delta
                x_plus = np.clip(x + c_k * delta, -5, 5)
                f_plus = func(x_plus)
                calls += 1
                
                # Evaluate at x - c_k * delta
                x_minus = np.clip(x - c_k * delta, -5, 5)
                f_minus = func(x_minus)
                calls += 1
                
                # Gradient estimate: (f_plus - f_minus) / (2 * c_k * delta_i)
                # Update: x_new = x - a_k * gradient_estimate
                # Note: delta_i is +/-1, so 1/delta_i = delta_i
                gradient_estimate = (f_plus - f_minus) / (2 * c_k) * delta
                x_new = x - a_k * gradient_estimate
                x_new = np.clip(x_new, -5, 5)
                
                # Evaluate new point
                f_new = func(x_new)
                calls += 1
                
                spsa_iterations += 1
                
                # Track stagnation
                if f_new < best_f - 1e-8:
                    best_f = f_new
                    best_x = x_new.copy()
                    spsa_stagnation = 0
                else:
                    spsa_stagnation += 1
                
                x = x_new
                fx = f_new
                
            else:
                # ES phase
                # Generate candidate
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls += 1
                
                # Update global best
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    successes.append(1)
                else:
                    successes.append(0)
                
                # Update current state
                if fy < fx:
                    x = y
                    fx = fy
                    # Increase step size on success
                    sigma *= 1.05
                else:
                    # Decrease step size on failure
                    sigma *= 0.95
                
                # Maintain sliding window
                if len(successes) > window_size:
                    successes.pop(0)
                
                # Adapt sigma using 1/5 rule
                if len(successes) == window_size:
                    success_rate = np.mean(successes)
                    if success_rate > 0.2:
                        sigma *= 1.1
                    elif success_rate < 0.15:
                        sigma /= 1.1
                
                # Clamp sigma to reasonable bounds
                sigma = np.clip(sigma, 1e-4, 10.0)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
