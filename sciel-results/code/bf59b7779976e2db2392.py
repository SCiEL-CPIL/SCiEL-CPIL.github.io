"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time


def construct():
    # Deadline guard: 150s wall-clock limit, return best found with safety margin
    t_start = time.time()
    deadline = t_start + 140.0  # 10s safety margin
    
    # Initialize with the all-ones sequence (known baseline: value = 1/N)
    best_x = np.ones(N)
    
    # Compute the baseline value
    def compute_value(x):
        # Autoconvolution
        ac = np.convolve(x, x, mode='full')
        # Max absolute value
        max_abs = np.max(np.abs(ac))
        # Squared integral (sum squared)
        integral_sq = np.sum(x) ** 2
        if integral_sq < 1e-12:
            return np.inf
        return max_abs / integral_sq
    
    best_val = compute_value(best_x)
    
    # Objective function for optimization: minimize max(|autoconv|) / (sum(x))^2
    # We use a smooth surrogate: soft-max of |autoconv| via log-sum-exp
    # and handle the denominator carefully
    
    def objective(x):
        ac = np.convolve(x, x, mode='full')
        abs_ac = np.abs(ac)
        # Soft-max approximation of max: (1/t) * log(sum(exp(t*abs_ac)))
        t = 10.0  # temperature parameter
        # Numerically stable log-sum-exp
        max_ac = np.max(abs_ac)
        if max_ac < 1e-12:
            soft_max = 0.0
        else:
            soft_max = (max_ac + np.log(np.sum(np.exp(t * (abs_ac - max_ac)))) / t)
        integral_sq = np.sum(x) ** 2
        if integral_sq < 1e-12:
            return 1e12
        return soft_max / integral_sq
    
    # Gradient of the objective (numerical)
    def objective_grad(x):
        eps = 1e-7
        grad = np.zeros(N)
        for i in range(N):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
        return grad
    
    # Constraints: keep x in a reasonable range to avoid blowup
    # We want to explore signed step-like functions, so constrain to [-2, 2]
    constraints = [
        {'type': 'ineq', 'fun': lambda x: 2.0 - x},
        {'type': 'ineq', 'fun': lambda x: x + 2.0}
    ]
    
    # Try a few restarts with different initializations
    num_restarts = 5
    rng = np.random.default_rng(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            x0 = best_x.copy()
        else:
            # Random initialization with step-like structure
            x0 = rng.choice([-1.0, 1.0], size=N).astype(np.float64)
            # Add some smoothing to help optimization
            from scipy.ndimage import gaussian_filter1d
            x0 = gaussian_filter1d(x0, sigma=3.0)
            x0 = np.clip(x0, -1.0, 1.0)
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=[(-2.0, 2.0)] * N,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 200}
            )
            
            if result.success and result.fun < best_val:
                best_x = result.x.copy()
                best_val = result.fun
                # Polish with tighter tolerance
                if time.time() < deadline - 20.0:
                    result2 = minimize(
                        objective,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-2.0, 2.0)] * N,
                        options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 100}
                    )
                    if result2.success and result2.fun < best_val:
                        best_x = result2.x.copy()
                        best_val = result2.fun
                        
        except Exception:
            continue
        
        # Check deadline
        if time.time() > deadline - 30.0:
            break
    
    # Final evaluation using the exact (non-smooth) objective
    final_val = compute_value(best_x)
    
    # The constant sequence gives 1/N ≈ 0.0025
    # If our optimization found something better, use it
    # Otherwise, fall back to the optimal constant sequence
    if final_val > 1.0 / N + 1e-12:
        return np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
