"""Optimized uncertainty bound minimizer with exact gradients."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _precompute_constants():
    """Precompute all constants needed for the bound and its gradient."""
    degrees = np.array([0, 4, 8, 12])
    n_vals = [0, 4, 8, 12]
    sqrt_pi = np.sqrt(np.pi)
    
    # H_n(0) values
    H0_at_0 = 1.0
    H4_at_0 = -12.0
    H8_at_0 = 1680.0
    H12_at_0 = -103950.0
    
    # Norms: norm_n = 2^n * n! * sqrt(pi)
    norms = []
    for n in n_vals:
        norms.append(2.0**n * float(np.math.factorial(n)) * sqrt_pi)
    norms = np.array(norms)
    
    # Diagonal M terms: diag_M_n = (n + 0.5) * norm_n
    diag_M = (np.array(n_vals) + 0.5) * norms
    
    # Off-diagonal integrals for |n_i - n_j| == 2
    # int x^2 H_n H_m exp(-x^2) dx for m = n+2:
    # = 2^n * (n+2)! * sqrt(pi)
    # For pairs in our set: (0,2) not in set, (4,6) not in set, (8,10) not in set, (12,14) not in set
    # Actually we need to check which pairs in {0,4,8,12} differ by 2: none!
    # So there are NO off-diagonal terms in M for this specific set of degrees.
    
    # Wait, let me double-check. The degrees are 0, 4, 8, 12.
    # |0-4|=4, |0-8|=8, |0-12|=12, |4-8|=4, |4-12|=8, |8-12|=4
    # None differ by exactly 2, so indeed no off-diagonal terms.
    
    return {
        'degrees': degrees,
        'n_vals': n_vals,
        'sqrt_pi': sqrt_pi,
        'H0_at_0': H0_at_0,
        'H4_at_0': H4_at_0,
        'H8_at_0': H8_at_0,
        'H12_at_0': H12_at_0,
        'norms': norms,
        'diag_M': diag_M,
    }

_CONSTS = _precompute_constants()

def _bound_and_grad(c0, c4, c8):
    """
    Compute the uncertainty bound M/N and its gradient with respect to (c0, c4, c8).
    
    g(0) = c0 + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    => c12 = -(c0 + c4*H4(0) + c8*H8(0)) / H12(0)
    
    N = sum_n c_n^2 * norm_n
    M = sum_n c_n^2 * diag_M_n  (no off-diagonal terms for degrees {0,4,8,12})
    
    Bound = M/N
    """
    H4_0 = _CONSTS['H4_at_0']
    H8_0 = _CONSTS['H8_at_0']
    H12_0 = _CONSTS['H12_at_0']
    norms = _CONSTS['norms']
    diag_M = _CONSTS['diag_M']
    
    # Compute c12 from g(0) = 0
    c12 = -(c0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # Coefficients for degrees 0, 4, 8, 12
    c = np.array([c0, c4, c8, c12])
    
    # N = sum c_n^2 * norm_n
    N = np.dot(c**2, norms)
    
    # M = sum c_n^2 * diag_M_n
    M = np.dot(c**2, diag_M)
    
    if N <= 0:
        return np.inf, np.array([0.0, 0.0, 0.0])
    
    bound = M / N
    
    # Gradient of bound = M/N with respect to c0, c4, c8
    # d(bound)/dc_k = (1/N) * dM/dc_k - (M/N^2) * dN/dc_k
    # 
    # dN/dc_k = 2 * c_k * norm_k for k in {0,4,8}, and dc12/dc_k contributes
    # dM/dc_k = 2 * c_k * diag_M_k for k in {0,4,8}, and dc12/dc_k contributes
    #
    # dc12/dc0 = -1/H12_0
    # dc12/dc4 = -H4_0/H12_0
    # dc12/dc8 = -H8_0/H12_0
    
    dc12_dc0 = -1.0 / H12_0
    dc12_dc4 = -H4_0 / H12_0
    dc12_dc8 = -H8_0 / H12_0
    
    # N = c0^2*norm0 + c4^2*norm4 + c8^2*norm8 + c12^2*norm12
    # dN/dc0 = 2*c0*norm0 + 2*c12*norm12*dc12_dc0
    # dN/dc4 = 2*c4*norm4 + 2*c12*norm12*dc12_dc4
    # dN/dc8 = 2*c8*norm8 + 2*c12*norm12*dc12_dc8
    
    dN_dc0 = 2.0 * c0 * norms[0] + 2.0 * c12 * norms[3] * dc12_dc0
    dN_dc4 = 2.0 * c4 * norms[1] + 2.0 * c12 * norms[3] * dc12_dc4
    dN_dc8 = 2.0 * c8 * norms[2] + 2.0 * c12 * norms[3] * dc12_dc8
    
    # M = c0^2*diag_M0 + c4^2*diag_M4 + c8^2*diag_M8 + c12^2*diag_M12
    # dM/dc0 = 2*c0*diag_M0 + 2*c12*diag_M12*dc12_dc0
    # dM/dc4 = 2*c4*diag_M4 + 2*c12*diag_M12*dc12_dc4
    # dM/dc8 = 2*c8*diag_M8 + 2*c12*diag_M12*dc12_dc8
    
    dM_dc0 = 2.0 * c0 * diag_M[0] + 2.0 * c12 * diag_M[3] * dc12_dc0
    dM_dc4 = 2.0 * c4 * diag_M[1] + 2.0 * c12 * diag_M[3] * dc12_dc4
    dM_dc8 = 2.0 * c8 * diag_M[2] + 2.0 * c12 * diag_M[3] * dc12_dc8
    
    # d(bound)/dc_k = (1/N)*dM/dc_k - (M/N^2)*dN/dc_k
    grad0 = (1.0 / N) * dM_dc0 - (M / (N * N)) * dN_dc0
    grad4 = (1.0 / N) * dM_dc4 - (M / (N * N)) * dN_dc4
    grad8 = (1.0 / N) * dM_dc8 - (M / (N * N)) * dN_dc8
    
    return bound, np.array([grad0, grad4, grad8])

def _objective(x):
    """Objective function for optimization."""
    val, _ = _bound_and_grad(x[0], x[1], x[2])
    return val

def _objective_grad(x):
    """Objective function with gradient for L-BFGS-B."""
    val, grad = _bound_and_grad(x[0], x[1], x[2])
    return val, grad

def construct():
    """
    Tune a Fourier-invariant Hermite-polynomial times Gaussian test function to reduce
    the upper bound on A(f)*A(Fourier(f)).
    
    Returns: three coefficients (c0, c4, c8) of the Hermite combination.
    The degree-12 weight is fixed so g(0) = 0.
    """
    start_time = time.time()
    deadline = start_time + 170  # Safety margin: finish well before 210s
    
    # Initial guess from prior work
    c0_init, c4_init, c8_init = -113.0/100.0, 1.0/25.0, 1.0/3240.0
    x0 = np.array([c0_init, c4_init, c8_init])
    
    best_x = x0.copy()
    best_val = _objective(x0)
    
    # Phase 1: L-BFGS-B with exact gradient from initial point
    try:
        res = minimize(_objective_grad, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 60:
        return best_x
    
    # Phase 2: Multiple random restarts with L-BFGS-B
    np.random.seed(42)
    n_restarts = 100
    for i in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Generate random starting point
        scale = 5.0 * (1.0 - i / n_restarts) + 0.5
        x_rand = np.random.uniform(-scale, scale, 3)
        
        try:
            res = minimize(_objective_grad, x_rand, jac=True, method='L-BFGS-B',
                           options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Check time
    if time.time() > deadline - 15:
        return best_x
    
    # Phase 3: High-precision polish from best point
    try:
        res = minimize(_objective_grad, best_x, jac=True, method='L-BFGS-B',
                       options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 100})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 5:
        return best_x
    
    # Phase 4: Final ultra-precise polish
    try:
        res = minimize(_objective_grad, best_x, jac=True, method='L-BFGS-B',
                       options={'maxiter': 50000, 'ftol': 1e-20, 'gtol': 1e-16, 'maxls': 200})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    return best_x

# EVOLVE-BLOCK-END
