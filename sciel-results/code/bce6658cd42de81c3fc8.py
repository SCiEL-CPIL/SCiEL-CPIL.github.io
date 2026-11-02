"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hermite_coeffs():
    """Return coefficient arrays for H0, H4, H8, H12 in descending powers of x."""
    H0 = np.array([1.0])
    H4 = np.array([16.0, 0.0, -48.0, 0.0, 12.0])
    H8 = np.array([256.0, 0.0, -1280.0, 0.0, 2880.0, 0.0, -2880.0, 0.0, 168.0])
    H12 = np.array([4096.0, 0.0, -38400.0, 0.0, 126720.0, 0.0, -215040.0, 0.0,
                     193536.0, 0.0, -80640.0, 0.0, 10368.0])
    return H0, H4, H8, H12

def _poly_eval(coeffs, x):
    """Evaluate polynomial with given coefficients (descending powers) at x."""
    result = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coeffs):
        power = len(coeffs) - 1 - i
        result += c * (x ** power)
    return result

def _build_gram_matrices():
    """Build Gram matrices for L2 norms of g, xg, and g' using Hermite function orthogonality.
    
    We use the Hermite functions psi_n(x) = H_n(x) * exp(-x^2/2) / sqrt(2^n * n! * sqrt(pi))
    which are orthonormal: <psi_m, psi_n> = delta_mn
    
    For g(x) = sum_i a_i * H_i(x) * exp(-x^2/2), we write:
    g(x) = sum_i a_i * sqrt(2^i * i! * sqrt(pi)) * psi_i(x)
    
    So the coefficients in the orthonormal basis are c_i = a_i * sqrt(2^i * i! * sqrt(pi))
    
    The L2 norm: ||g||^2 = sum_i c_i^2
    The x-moment: <x*g, g> = sum_i c_i * <x*psi_i, psi_i>
    Since x*psi_i = sqrt((i+1)/2)*psi_{i+1} + sqrt(i/2)*psi_{i-1}
    <x*psi_i, psi_j> = sqrt((j+1)/2)*delta_{i,j+1} + sqrt(j/2)*delta_{i,j-1}
    
    For the derivative: g'(x) = sum_i c_i * psi_i'(x)
    psi_i'(x) = -x*psi_i(x) + sqrt(i/2)*psi_{i-1}(x) ... actually let's use:
    d/dx psi_i = -x*psi_i + sqrt(i/2)*psi_{i-1}
    
    ||g'||^2 = sum_{i,j} c_i * c_j * <psi_i', psi_j>
    """
    # Degrees we use: 0, 4, 8, 12
    degrees = [0, 4, 8, 12]
    n = len(degrees)
    
    # Precompute sqrt(2^i * i! * sqrt(pi)) for normalization
    from scipy.special import factorial
    norm_factors = np.zeros(n)
    for idx, deg in enumerate(degrees):
        norm_factors[idx] = np.sqrt(2.0**deg * factorial(deg, exact=True) * np.sqrt(np.pi))
    
    # Build Gram matrix for ||g||^2: G0[i,j] = <psi_{deg_i}, psi_{deg_j}>
    # Since psi are orthonormal, G0 = I
    G0 = np.eye(n)
    
    # Build Gram matrix for <x*g, g>: Gx[i,j] = <psi_{deg_i}, x*psi_{deg_j}>
    # x*psi_j = sqrt((deg_j+1)/2)*psi_{deg_j+1} + sqrt(deg_j/2)*psi_{deg_j-1}
    Gx = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            deg_i = degrees[i]
            deg_j = degrees[j]
            # <psi_i, x*psi_j> = sqrt((deg_j+1)/2)*<psi_i, psi_{deg_j+1}> + sqrt(deg_j/2)*<psi_i, psi_{deg_j-1}>
            val = 0.0
            # Check if deg_i == deg_j + 1
            if deg_i == deg_j + 1:
                val += np.sqrt((deg_j + 1) / 2.0)
            # Check if deg_i == deg_j - 1
            if deg_i == deg_j - 1 and deg_j > 0:
                val += np.sqrt(deg_j / 2.0)
            Gx[i, j] = val
    
    # Build Gram matrix for ||g'||^2: Gp[i,j] = <psi_i', psi_j>
    # psi_i' = -x*psi_i + sqrt(i/2)*psi_{i-1}
    # So <psi_i', psi_j> = -<x*psi_i, psi_j> + sqrt(deg_i/2)*<psi_{deg_i-1}, psi_j>
    # <x*psi_i, psi_j> = sqrt((deg_i+1)/2)*delta_{j,deg_i+1} + sqrt(deg_i/2)*delta_{j,deg_i-1}
    # So <psi_i', psi_j> = -sqrt((deg_i+1)/2)*delta_{j,deg_i+1} - sqrt(deg_i/2)*delta_{j,deg_i-1} + sqrt(deg_i/2)*delta_{deg_i-1,j}
    # The last two terms cancel when j = deg_i - 1, leaving:
    # <psi_i', psi_j> = -sqrt((deg_i+1)/2)*delta_{j,deg_i+1}
    # Wait, let me redo this more carefully.
    # psi_i'(x) = -x*psi_i(x) + sqrt(i/2)*psi_{i-1}(x)
    # <psi_i', psi_j> = -<x*psi_i, psi_j> + sqrt(deg_i/2)*<psi_{deg_i-1}, psi_j>
    # <x*psi_i, psi_j> = sqrt((deg_i+1)/2)*delta_{j,deg_i+1} + sqrt(deg_i/2)*delta_{j,deg_i-1}
    # So <psi_i', psi_j> = -sqrt((deg_i+1)/2)*delta_{j,deg_i+1} - sqrt(deg_i/2)*delta_{j,deg_i-1} + sqrt(deg_i/2)*delta_{deg_i-1,j}
    # When j = deg_i - 1: -sqrt(deg_i/2) + sqrt(deg_i/2) = 0
    # When j = deg_i + 1: -sqrt((deg_i+1)/2)
    # Otherwise: 0
    # So <psi_i', psi_j> = -sqrt((deg_i+1)/2)*delta_{j,deg_i+1}
    
    Gp = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            deg_i = degrees[i]
            deg_j = degrees[j]
            if deg_j == deg_i + 1:
                Gp[i, j] = -np.sqrt((deg_i + 1) / 2.0)
    
    return G0, Gx, Gp, norm_factors, degrees

def _compute_c4_upper_bound_analytic(a0, a4, a8):
    """Compute the upper bound on C4 using Gram matrices and analytic formulas."""
    G0, Gx, Gp, norm_factors, degrees = _build_gram_matrices()
    
    # Coefficients in the original basis: a = [a0, a4, a8, a12]
    # a12 from constraint g(0) = 0: a0*1 + a4*12 + a8*168 + a12*10368 = 0
    a12 = -(a0 * 1.0 + a4 * 12.0 + a8 * 168.0) / 10368.0
    a = np.array([a0, a4, a8, a12])
    
    # Coefficients in orthonormal basis: c_i = a_i * norm_factors[i]
    c = a * norm_factors
    
    # ||g||^2 = c^T G0 c = c^T c
    norm_sq = c @ G0 @ c
    if norm_sq < 1e-15:
        return 1e10
    
    # <x*g, g> = c^T Gx c
    x_mean = c @ Gx @ c
    
    # <x^2*g, g> = c^T (x^2 in operator form) c
    # x^2*psi_j = ((j+1)/2)*psi_j + sqrt((j+1)(j+2)/4)*psi_{j+2} + sqrt(j(j-1)/4)*psi_{j-2}
    # Actually: x^2 = (1/2)(a^+a^- + a^-a^+ + a^+a^+ + a^-a^-) ... let's use:
    # x*psi_j = sqrt((j+1)/2)*psi_{j+1} + sqrt(j/2)*psi_{j-1}
    # x^2*psi_j = x*(sqrt((j+1)/2)*psi_{j+1} + sqrt(j/2)*psi_{j-1})
    #            = sqrt((j+1)/2)*(sqrt((j+2)/2)*psi_{j+2} + sqrt((j+1)/2)*psi_j)
    #              + sqrt(j/2)*(sqrt(j/2)*psi_{j-1} + sqrt((j-1)/2)*psi_{j-2})
    #            = sqrt((j+1)(j+2)/4)*psi_{j+2} + (j+1)/2*psi_j + j/2*psi_{j-1} + sqrt(j(j-1)/4)*psi_{j-2}
    # Wait, that's not right. Let me redo:
    # x^2*psi_j = x*(x*psi_j) = x*(sqrt((j+1)/2)*psi_{j+1} + sqrt(j/2)*psi_{j-1})
    # For the first term: x*psi_{j+1} = sqrt((j+2)/2)*psi_{j+2} + sqrt((j+1)/2)*psi_j
    # For the second term: x*psi_{j-1} = sqrt(j/2)*psi_j + sqrt((j-1)/2)*psi_{j-2}  (if j>=1)
    # So x^2*psi_j = sqrt((j+1)/2)*[sqrt((j+2)/2)*psi_{j+2} + sqrt((j+1)/2)*psi_j]
    #              + sqrt(j/2)*[sqrt(j/2)*psi_j + sqrt((j-1)/2)*psi_{j-2}]
    #            = sqrt((j+1)(j+2)/4)*psi_{j+2} + (j+1)/2*psi_j + j/2*psi_j + sqrt(j(j-1)/4)*psi_{j-2}
    #            = sqrt((j+1)(j+2)/4)*psi_{j+2} + (2j+1)/2*psi_j + sqrt(j(j-1)/4)*psi_{j-2}
    
    x2_mean = 0.0
    for i in range(len(degrees)):
        for j in range(len(degrees)):
            deg_j = degrees[j]
            val = 0.0
            if degrees[i] == deg_j + 2:
                val += np.sqrt((deg_j + 1) * (deg_j + 2) / 4.0)
            elif degrees[i] == deg_j:
                val += (2 * deg_j + 1) / 2.0
            elif degrees[i] == deg_j - 2 and deg_j >= 2:
                val += np.sqrt(deg_j * (deg_j - 1) / 4.0)
            x2_mean += c[i] * val * c[j]
    
    var_x = x2_mean - x_mean**2
    if var_x < 1e-15:
        return 1e10
    
    # ||g'||^2 = c^T Gp c
    p2_mean = c @ Gp @ c
    if p2_mean < 1e-15:
        return 1e10
    
    return var_x * p2_mean

def _compute_c4_upper_bound(a0, a4, a8):
    """Compute the upper bound on C4 for given coefficients using analytic Gram matrix approach."""
    return _compute_c4_upper_bound_analytic(a0, a4, a8)

def construct():
    """Return three coefficients of the Hermite combination (degrees 0, 4, 8)."""
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin before 180s limit
    
    # Seed values
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_params = seed.copy()
    best_score = _compute_c4_upper_bound(*seed)
    
    # Multiple starting points for multi-start Nelder-Mead
    start_points = [
        seed,
        seed + np.array([0.05, 0.005, 0.0005]),
        seed - np.array([0.05, 0.005, 0.0005]),
        seed + np.array([0.2, 0.02, 0.002]),
        seed - np.array([0.2, 0.02, 0.002]),
        np.array([-1.0, 0.04, 0.0004]),
        np.array([-0.8, 0.032, 0.00032]),
        np.array([-1.2, 0.048, 0.00048]),
        np.array([-0.9, 0.036, 0.00036]),
        np.array([-1.1, 0.044, 0.00044]),
    ]
    
    # Multi-start Nelder-Mead optimization
    for i, sp in enumerate(start_points):
        if time.time() > deadline - 20.0:
            break
        try:
            result = minimize(_compute_c4_upper_bound, sp, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 10.0:
        try:
            result = minimize(_compute_c4_upper_bound, best_params, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
