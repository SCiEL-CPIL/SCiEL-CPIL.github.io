"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _walsh_hadamard_basis(N):
    """Generate Walsh-Hadamard basis vectors of length N."""
    # Use the sequency-ordered Walsh functions via the fast Walsh-Hadamard transform
    # We'll use the standard Hadamard matrix construction
    # For efficiency, we'll generate a subset of basis vectors
    
    # Start with H_1 = [[1]]
    H = np.array([[1.0]])
    while H.shape[0] < N:
        # Kronecker product with [[1, 1], [1, -1]]
        H = np.block([[H, H], [H, -H]])
    
    H = H[:N, :N]
    # Normalize columns to unit norm
    H = H / np.sqrt(N)
    return H

def _autoconvolution(x):
    """Compute the full autoconvolution of x."""
    return np.convolve(x, x, mode='full')

def _max_abs_autoconv(x):
    """Maximum absolute value of autoconvolution."""
    return np.max(np.abs(_autoconvolution(x)))

def _integral_squared(x):
    """Squared integral (sum of squares) of x."""
    return np.sum(x ** 2)

def _objective(coeffs, H, N):
    """Objective: maximize the ratio of squared integral to max abs autoconv.
    We minimize the negative log of this ratio for smoothness, or directly minimize
    the max abs autoconv / integral_squared."""
    x = H @ coeffs
    ac = _autoconvolution(x)
    max_abs = np.max(np.abs(ac))
    int_sq = np.sum(x ** 2)
    if int_sq < 1e-12:
        return 1e10
    return max_abs / int_sq

def _soft_max_abs_autoconv(x, beta=10.0):
    """Soft approximation of max(abs(autoconv(x))) using log-sum-exp."""
    ac = _autoconvolution(x)
    # Use soft max of |ac|
    # log-sum-exp of |ac| * beta, then divide by beta
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-12:
        return 0.0
    # Shift for numerical stability
    shifted = abs_ac - m
    log_sum_exp = m + np.log(np.sum(np.exp(beta * shifted))) / beta
    return log_sum_exp

def _soft_objective(coeffs, H, beta=10.0):
    """Soft objective using log-sum-exp for the max."""
    x = H @ coeffs
    ac = _autoconvolution(x)
    int_sq = np.sum(x ** 2)
    if int_sq < 1e-12:
        return 1e10
    soft_max = _soft_max_abs_autoconv(x, beta)
    return soft_max / int_sq

def _sign_optimize(x, max_iters=50):
    """Local search: flip signs to reduce max abs autoconv."""
    x = x.copy()
    best_score = _max_abs_autoconv(x) / _integral_squared(x)
    
    for _ in range(max_iters):
        improved = False
        # Try flipping each sign
        for i in range(N):
            x_new = x.copy()
            x_new[i] = -x_new[i]
            score_new = _max_abs_autoconv(x_new) / _integral_squared(x_new)
            if score_new < best_score - 1e-10:
                best_score = score_new
                x = x_new
                improved = True
                break
        if not improved:
            break
    
    return x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Leave 30s margin before 150s kill
    
    # Generate Walsh-Hadamard basis
    H = _walsh_hadamard_basis(N)
    
    # Start with a small number of basis functions
    num_basis = min(20, N)
    H_sub = H[:, :num_basis]
    
    # Initialize coefficients
    np.random.seed(42)
    coeffs = np.zeros(num_basis)
    coeffs[0] = 1.0  # DC component
    
    best_x = None
    best_score = np.inf
    
    # Gradient descent on the soft objective
    beta = 10.0
    lr = 0.01
    
    for epoch in range(200):
        if time.time() > deadline - 10:
            break
        
        # Compute gradient numerically or analytically
        # Use numerical gradient for simplicity
        grad = np.zeros(num_basis)
        eps = 1e-5
        for j in range(num_basis):
            coeffs_plus = coeffs.copy()
            coeffs_plus[j] += eps
            f_plus = _soft_objective(coeffs_plus, H_sub, beta)
            
            coeffs_minus = coeffs.copy()
            coeffs_minus[j] -= eps
            f_minus = _soft_objective(coeffs_minus, H_sub, beta)
            
            grad[j] = (f_plus - f_minus) / (2 * eps)
        
        # Update
        coeffs -= lr * grad
        
        # Project to sign domain
        x = H_sub @ coeffs
        x_sign = np.sign(x)
        x_sign[x_sign == 0] = 1.0
        
        # Evaluate on sign domain
        score = _max_abs_autoconv(x_sign) / _integral_squared(x_sign)
        if score < best_score:
            best_score = score
            best_x = x_sign.copy()
        
        # Local sign optimization
        x_opt, score_opt = _sign_optimize(x_sign, max_iters=10)
        if score_opt < best_score:
            best_score = score_opt
            best_x = x_opt.copy()
    
    if best_x is None:
        return np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
