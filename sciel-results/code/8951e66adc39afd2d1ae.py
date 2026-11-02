"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Constants for standard Physicists' Hermite Polynomials
# H_k(0) values for even k
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 665280.0  # Corrected from seed's 2032128

# Weights W_k = sqrt(pi) * 2^k * k!
# We only need relative weights, so we can drop sqrt(pi)
# W_k = 2^k * k!
W = {}
for k in [0, 4, 8, 12]:
    W[k] = (2 ** k) * np.math.factorial(k)

# Pre-compute (2k+1) * W_k
V_weight = {k: (2 * k + 1) * W[k] for k in W}

# Inverse of H12_0 for quick calculation of c12
INV_H12_0 = 1.0 / H12_0

def _compute_bound_exact(a0, a4, a8):
    """
    Compute the uncertainty product (variance) exactly using Hermite orthogonality.
    Returns the variance V = S1 / S0.
    """
    c0 = a0
    c4 = a4
    c8 = a8
    c12 = -(a0 + H4_0 * a4 + H8_0 * a8) * INV_H12_0
    
    # Sum of squares weighted by W_k
    s0 = (c0**2 * W[0] + c4**2 * W[4] + c8**2 * W[8] + c12**2 * W[12])
    
    if s0 < 1e-30:
        return 1e10
        
    # Sum of squares weighted by (2k+1)W_k
    s1 = (c0**2 * V_weight[0] + c4**2 * V_weight[4] + c8**2 * V_weight[8] + c12**2 * V_weight[12])
    
    return s1 / s0

def _compute_gradient_exact(a0, a4, a8):
    """
    Compute the gradient of the bound with respect to (a0, a4, a8).
    """
    c0 = a0
    c4 = a4
    c8 = a8
    c12 = -(a0 + H4_0 * a4 + H8_0 * a8) * INV_H12_0
    
    s0 = (c0**2 * W[0] + c4**2 * W[4] + c8**2 * W[8] + c12**2 * W[12])
    
    if s0 < 1e-30:
        return np.array([0.0, 0.0, 0.0])
        
    s1 = (c0**2 * V_weight[0] + c4**2 * V_weight[4] + c8**2 * V_weight[8] + c12**2 * V_weight[12])
    
    # Derivatives of c_k with respect to a_i
    # dc0/da = [1, 0, 0]
    # dc4/da = [0, 1, 0]
    # dc8/da = [0, 0, 1]
    # dc12/da = [-K, -12K, -1680K]
    
    K = INV_H12_0
    dc12_da = np.array([-K, -H4_0 * K, -H8_0 * K])
    
    # ds0/da_i = 2 * sum( c_k * W_k * dc_k/da_i )
    ds0_da0 = 2.0 * (c0 * W[0] * 1.0 + c12 * W[12] * dc12_da[0])
    ds0_da4 = 2.0 * (c4 * W[4] * 1.0 + c12 * W[12] * dc12_da[1])
    ds0_da8 = 2.0 * (c8 * W[8] * 1.0 + c12 * W[12] * dc12_da[2])
    
    # ds1/da_i = 2 * sum( c_k * (2k+1)W_k * dc_k/da_i )
    ds1_da0 = 2.0 * (c0 * V_weight[0] * 1.0 + c12 * V_weight[12] * dc12_da[0])
    ds1_da4 = 2.0 * (c4 * V_weight[4] * 1.0 + c12 * V_weight[12] * dc12_da[1])
    ds1_da8 = 2.0 * (c8 * V_weight[8] * 1.0 + c12 * V_weight[12] * dc12_da[2])
    
    # dV = (ds1 * s0 - s1 * ds0) / s0^2
    factor = 1.0 / (s0 * s0)
    
    grad0 = (ds1_da0 * s0 - s1 * ds0_da0) * factor
    grad4 = (ds1_da4 * s0 - s1 * ds0_da4) * factor
    grad8 = (ds1_da8 * s0 - s1 * ds0_da8) * factor
    
    return np.array([grad0, grad4, grad8])

def construct():
    start_time = time.time()
    deadline = 195.0  # Wall-clock limit with safety margin

    # Initial guesses
    # The seed's coefficients were optimized for a different H12, so they might not be optimal here.
    # However, they are a reasonable starting point.
    x0_seed = np.array([-1.13, 0.04, 1.0/3240.0])
    
    # Try a few diverse starting points
    rng = np.random.default_rng(42)
    starts = [
        x0_seed,
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]), # Will result in c12=0, pure H0? No, P(0)=1!=0. Invalid? 
        # If a0=a4=a8=0, c12=0. P=0. Norm=0.
        # We need to avoid zero function.
        np.array([-1.0, 0.1, 0.0]),
        np.array([-1.0, 0.0, 0.001]),
        np.array([-2.0, 0.0, 0.0]),
        np.array([-0.5, 0.0, 0.0]),
    ]
    
    best_x = starts[0].copy()
    best_score = _compute_bound_exact(best_x[0], best_x[1], best_x[2])
    
    for s in starts:
        val = _compute_bound_exact(s[0], s[1], s[2])
        if val < best_score:
            best_score = val
            best_x = s.copy()
            
    def objective(a):
        return _compute_bound_exact(a[0], a[1], a[2])
        
    def gradient(a):
        return _compute_gradient_exact(a[0], a[1], a[2])
        
    # Multi-start L-BFGS-B
    # Add random perturbations
    for _ in range(20):
        starts.append(best_x + rng.normal(0, 0.1, size=3))
        
    for i, start in enumerate(starts):
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = minimize(objective, start, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 5000, 'gtol': 1e-14, 'ftol': 1e-14, 'maxls': 100})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
            
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            res = minimize(objective, best_x, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 10000, 'gtol': 1e-16, 'ftol': 1e-16, 'maxls': 200})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
            
    return best_x

# EVOLVE-BLOCK-END
