"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4]
    to minimize the upper bound on C3.
    
    The metric is: upper_bound = max_k |conv(h, h)[k]| / (sum_i |h_i|)^2
    
    We use gradient-based optimization with a smooth approximation of the objective.
    """
    def objective(x):
        # Map x to [-1, 1] using tanh
        h = np.tanh(x)
        
        # Compute autoconvolution
        conv = np.convolve(h, h, mode='full')
        
        # Smooth absolute value
        eps = 1e-10
        abs_conv = np.sqrt(conv**2 + eps)
        
        # Soft max with temperature T
        T = 2.0
        max_abs = np.max(abs_conv)
        # Stable log-sum-exp
        shifted = (abs_conv - max_abs) / T
        log_sum_exp = np.log(np.sum(np.exp(shifted))) + max_abs / T
        soft_max = T * log_sum_exp
        
        # Sum of absolute values of h
        sum_abs_h = np.sum(np.sqrt(h**2 + eps))
        
        # Objective: soft_max / (sum_abs_h)^2
        value = soft_max / (sum_abs_h ** 2)
        return value
    
    # Initial guess: start with alternating signs to reduce convolution peaks
    # This helps escape the local minimum of all-ones
    x0 = np.zeros(N)
    # Set every other element to have opposite sign
    x0[::2] = 2.0   # tanh(2) ≈ 0.964
    x0[1::2] = -2.0  # tanh(-2) ≈ -0.964
    
    # Use L-BFGS-B with multiple iterations
    res = minimize(objective, x0, method='L-BFGS-B', 
                   options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15})
    
    h_opt = np.tanh(res.x)
    
    # Also try from all-ones initialization and take the better result
    x0_ones = np.full(N, 3.0)  # tanh(3) ≈ 0.995
    res_ones = minimize(objective, x0_ones, method='L-BFGS-B', 
                        options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15})
    h_ones = np.tanh(res_ones.x)
    
    # Compare the two results
    def eval_h(h):
        conv = np.convolve(h, h, mode='full')
        eps = 1e-10
        abs_conv = np.sqrt(conv**2 + eps)
        sum_abs_h = np.sum(np.sqrt(h**2 + eps))
        return np.max(abs_conv) / (sum_abs_h ** 2)
    
    val_alt = eval_h(h_opt)
    val_ones = eval_h(h_ones)
    
    if val_alt < val_ones:
        return h_opt
    else:
        return h_ones
# EVOLVE-BLOCK-END
