"""Combined program using analytic quadrature and global optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

# Deadline guard: 170 seconds to be safe within 180s limit
_START_TIME = time.time()
_DEADLINE = _START_TIME + 170.0

def _time_remaining():
    return _DEADLINE - time.time()

def _hermite_poly(n, x):
    """Physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    elif n == 2:
        return 4 * x**2 - 2
    elif n == 3:
        return 8 * x**3 - 12 * x
    elif n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    elif n == 8:
        return 256 * x**8 - 4480 * x**6 + 30240 * x**4 - 67200 * x**2 + 16800
    elif n == 12:
        return (4096 * x**12 - 129024 * x**10 + 1814400 * x**8 
                - 14417280 * x**6 + 55776000 * x**4 
                - 91452000 * x**2 + 47900160)
    else:
        raise ValueError(f"Hermite polynomial of degree {n} not implemented")

# Precompute Gauss-Hermite nodes and weights
# 200 nodes is very accurate for polynomials of degree up to ~400
_N_QUAD = 200
_X_NODES, _W_NODES = np.polynomial.hermite.hermgauss(_N_QUAD)

# Precompute powers of x_nodes for efficiency
_X2 = _X_NODES**2
_X3 = _X_NODES**3
_X4 = _X_NODES**4
_X5 = _X_NODES**5
_X6 = _X_NODES**6
_X7 = _X_NODES**7
_X8 = _X_NODES**8
_X9 = _X_NODES**9
_X10 = _X_NODES**10
_X11 = _X_NODES**11
_X12 = _X_NODES**12

# Precompute H_n and H'_n at nodes
_H0 = np.ones_like(_X_NODES)
_H4 = 16 * _X4 - 48 * _X2 + 12
_H8 = 256 * _X8 - 4480 * _X6 + 30240 * _X4 - 67200 * _X2 + 16800
_H12 = 4096 * _X12 - 129024 * _X10 + 1814400 * _X8 - 14417280 * _X6 + 55776000 * _X4 - 91452000 * _X2 + 47900160

# Derivatives
# H0' = 0
# H4' = 64x^3 - 96x
_H4p = 64 * _X3 - 96 * _X_NODES
# H8' = 2048x^7 - 26880x^5 + 120960x^3 - 134400x
_H8p = 2048 * _X7 - 26880 * _X5 + 120960 * _X3 - 134400 * _X_NODES
# H12' = 49152x^11 - 1290240x^9 + 14515200x^7 - 86503680x^5 + 223104000x^3 - 182904000x
_H12p = (49152 * _X11 - 1290240 * _X9 + 14515200 * _X7 
         - 86503680 * _X5 + 223104000 * _X3 - 182904000 * _X_NODES)

def _uncertainty_product(c0, c4, c8):
    """
    Compute the uncertainty product A(f) * A(F̂) for the test function
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is chosen so that g(0) = 0.
    """
    # g(0) = c0*1 + c4*12 + c8*16800 + c12*47900160 = 0
    c12 = -(c0 + 12 * c4 + 16800 * c8) / 47900160.0
    
    # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    P = c0 * _H0 + c4 * _H4 + c8 * _H8 + c12 * _H12
    
    # P'(x)
    P_prime = c4 * _H4p + c8 * _H8p + c12 * _H12p
    
    # Weights for quadrature: w_i * e^{-x_i^2} is already handled by hermgauss
    # hermgauss computes \int f(x) e^{-x^2} dx
    # So we just multiply f(x) by weights and sum.
    
    # Norm squared: \int P^2 e^{-x^2} dx
    P_sq = P * P
    norm_sq = np.dot(_W_NODES, P_sq)
    
    if norm_sq <= 1e-15 or not np.isfinite(norm_sq):
        return 1e18
    
    # Position variance numerator: \int x^2 P^2 e^{-x^2} dx
    x2_P_sq = _X2 * P_sq
    x2_norm_sq = np.dot(_W_NODES, x2_P_sq)
    
    # Frequency variance numerator: \int |g'|^2 dx = \int (P' - xP)^2 e^{-x^2} dx
    term = P_prime - _X_NODES * P
    term_sq = term * term
    freq_norm_sq = np.dot(_W_NODES, term_sq)
    
    if freq_norm_sq <= 1e-15 or not np.isfinite(freq_norm_sq):
        return 1e18
        
    var_x = x2_norm_sq / norm_sq
    var_freq = freq_norm_sq / norm_sq
    
    if var_x < 0 or var_freq < 0:
        return 1e18
        
    return np.sqrt(var_x * var_freq)

def _objective(c):
    return _uncertainty_product(c[0], c[1], c[2])

def construct():
    """
    Find the three coefficients (c0, c4, c8) that minimize the uncertainty product.
    Returns np.array([c0, c4, c8]).
    """
    best_coeffs = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_value = _uncertainty_product(best_coeffs[0], best_coeffs[1], best_coeffs[2])
    
    # Strategy 1: Global optimization using Differential Evolution
    # This helps escape local minima which local optimizers might get stuck in.
    # The coefficients can vary significantly. 
    # Based on seed and typical scales:
    # c0 ~ O(1), c4 ~ O(0.1), c8 ~ O(0.001)
    # We use a wide range to allow global search.
    
    if _time_remaining() > 60:
        try:
            bounds = [
                (-5.0, 5.0),
                (-1.0, 1.0),
                (-0.05, 0.05)
            ]
            
            # Differential Evolution can be slow, so limit popsize and maxiter
            # We rely on the fast objective function
            de_result = differential_evolution(
                _objective,
                bounds=bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                polish=False # We will polish manually
            )
            
            if de_result.fun < best_value:
                best_value = de_result.fun
                best_coeffs = de_result.x
        except Exception:
            pass
    
    # Strategy 2: Local refinement from the best point found
    # Use L-BFGS-B for fast convergence with gradient estimation (or analytic if available, but numerical is fine here)
    if _time_remaining() > 10:
        try:
            res = minimize(
                _objective,
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_value:
                best_value = res.fun
                best_coeffs = res.x
        except Exception:
            pass
            
    # Strategy 3: Final polish with Nelder-Mead for high precision
    if _time_remaining() > 2:
        try:
            res = minimize(
                _objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16}
            )
            if res.fun < best_value:
                best_value = res.fun
                best_coeffs = res.x
        except Exception:
            pass

    return best_coeffs
# EVOLVE-BLOCK-END
