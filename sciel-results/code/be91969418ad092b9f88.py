"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _compute_upper_bound(h):
    """Compute the upper bound on C3 for given heights h."""
    conv = np.convolve(h, h, mode='full')
    max_abs_conv = np.max(np.abs(conv))
    sum_abs_h = np.sum(np.abs(h))
    if sum_abs_h == 0:
        return np.inf
    return max_abs_conv / (sum_abs_h ** 2)


def _objective(x):
    """Smooth objective function for optimization."""
    h = np.tanh(x)
    conv = np.convolve(h, h, mode='full')
    abs_conv = np.abs(conv)
    
    # Smooth max using log-sum-exp
    T = 2.0
    max_val = np.max(abs_conv)
    if max_val < 1e-10:
        log_max = 1e-10
    else:
        log_sum_exp = np.log(np.sum(np.exp((abs_conv - max_val) / T))) + max_val / T
        log_max = T * log_sum_exp
    
    sum_abs_h = np.sum(np.abs(h))
    if sum_abs_h < 1e-10:
        return 1e10
    
    return log_max / (sum_abs_h ** 2)


def _objective_grad(x):
    """Gradient of the smooth objective (numerical approximation for L-BFGS-B)."""
    # Use finite differences for gradient
    eps = 1e-7
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (_objective(x_plus) - _objective(x_minus)) / (2 * eps)
    return grad


def construct():
    """Construct the signed step function heights."""
    # Try multiple initializations to avoid local minima
    best_h = None
    best_value = np.inf
    
    # Initialization 1: All positive
    x0_1 = np.full(N, 0.0)
    
    # Initialization 2: Alternating signs
    x0_2 = np.array([5.0 if i % 2 == 0 else -5.0 for i in range(N)])
    
    # Initialization 3: Random binary
    np.random.seed(42)
    x0_3 = np.random.choice([-5.0, 5.0], size=N)
    
    # Initialization 4: Block pattern
    x0_4 = np.zeros(N)
    block_size = 10
    for i in range(0, N, block_size * 2):
        x0_4[i:i+block_size] = 5.0
        x0_4[i+block_size:i+2*block_size] = -5.0
    
    initializations = [x0_1, x0_2, x0_3, x0_4]
    
    for x0 in initializations:
        try:
            res = minimize(
                _objective, 
                x0, 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_candidate = np.tanh(res.x)
            value = _compute_upper_bound(h_candidate)
            
            if value < best_value:
                best_value = value
                best_h = h_candidate
        except Exception as e:
            continue
    
    if best_h is None:
        # Fallback to seed
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
