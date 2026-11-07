import time
import numpy as np
from scipy.optimize import differential_evolution

class Optimizer:
    """
    Multi-start Differential Evolution for global exploration,
    followed by an adaptive (1+1)-ES with 1/5-rule step-size adaptation
    for local refinement on BBOB Weierstrass (f16).
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        # Safety margin: evaluator kills at 330s, we aim to finish by 290s
        deadline = time.time() + 290.0
        
        n_calls = [0]
        stop_flag = [False]

        def safe_func(x):
            if stop_flag[0] or n_calls[0] >= budget or time.time() > deadline:
                stop_flag[0] = True
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            try:
                return float(func(x))
            except Exception:
                return np.inf

        bounds = [(-5.0, 5.0) for _ in range(dim)]
        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: Multi-run Differential Evolution for global exploration
        # Weierstrass is non-separable and multi-modal. DE is robust.
        # Split budget into runs with decreasing size to allow refinement of search space
        # if time permits, but primarily to restart exploration if stuck.
        # Total DE budget ~ 400, leaving ~100 for ES.
        runs = [250, 150]
        
        for i, run_budget in enumerate(runs):
            if n_calls[0] >= budget or time.time() > deadline:
                break
            
            stop_flag[0] = False
            start_calls = n_calls[0]
            
            try:
                # popsize=15 is a good balance for dim=5
                # latinhypercube init ensures good spread
                res = differential_evolution(
                    safe_func,
                    bounds,
                    maxiter=100, # High maxiter, callback controls stopping
                    popsize=15,
                    tol=1e-10,
                    seed=42 + i,
                    init='latinhypercube',
                    # Callback stops DE if budget/time exceeded or stop_flag set
                    callback=lambda xk, convergence: True if (
                        stop_flag[0] or 
                        n_calls[0] - start_calls >= run_budget or 
                        time.time() > deadline
                    ) else False
                )
                
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 2: Adaptive (1+1)-ES with 1/5-rule step-size adaptation
        # This exploits local structure for rapid convergence in the best basin found.
        remaining = budget - n_calls[0]
        if remaining > 5 and time.time() < deadline:
            # Initialize step size. 
            # DE usually lands close enough that a moderate sigma is safe.
            # If DE failed completely, sigma might need to be larger, but 
            # given the box constraints and DE's global nature, 0.1-0.5 is reasonable.
            # Let's start with 0.5 to allow some movement if DE wasn't perfect.
            sigma = 0.5
            x = best_x.copy()
            f_x = best_f
            
            # 1/5-rule parameters
            # c_sigma controls how fast sigma adapts
            c_sigma = 1.0 / (5 * dim)
            # rho is time constant for success rate estimation
            rho = 1.0 / (5 * dim + 1)
            # target success rate
            success_rate = 0.5
            
            # Track success rate in a window
            n_success = 0
            n_total = 0
            window_size = 5 * dim + 1
            
            # Local RNG for reproducibility within the run
            rng_local = np.random.default_rng(123)
            
            while n_calls[0] < budget and time.time() < deadline and not stop_flag[0]:
                # Generate one offspring
                z = rng_local.standard_normal(dim)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5.0, 5.0)
                
                f_trial = safe_func(x_trial)
                
                if not np.isfinite(f_trial):
                    break

                # Update success rate
                n_total += 1
                if f_trial < f_x:
                    n_success += 1
                    # Accept step
                    x = x_trial
                    f_x = f_trial
                    best_f = f_trial
                    best_x = x_trial.copy()
                else:
                    # Reject step, keep x and f_x
                    pass

                # Adapt sigma using 1/5-rule
                # We update sigma based on the current success rate
                # if n_success / n_total > target, increase sigma, else decrease
                if n_total >= window_size:
                    current_sr = n_success / n_total
                    # Adjust log(sigma)
                    # If SR > 0.5, increase sigma (multiply by factor > 1)
                    # If SR < 0.5, decrease sigma (multiply by factor < 1)
                    # The 1/5 rule updates: sigma = sigma * (1 + c_sigma * (sr - 0.5))
                    # But standard 1/5 rule is often implemented as:
                    # sigma = sigma * exp(c_sigma * (sr - 0.5))
                    # Or simply: if sr > 0.5: sigma *= 1.1 else sigma *= 0.9
                    # Let's use the exponential form for smoothness
                    
                    log_sigma_factor = c_sigma * (current_sr - 0.5)
                    sigma *= np.exp(log_sigma_factor)
                    
                    # Reset window
                    n_success = 0
                    n_total = 0
                    
                    # Clamp sigma to avoid explosion or vanishing
                    sigma = np.clip(sigma, 1e-8, 10.0)

        return best_x, best_f
