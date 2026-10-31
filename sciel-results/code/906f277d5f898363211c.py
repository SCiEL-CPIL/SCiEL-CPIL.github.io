"""
Search for optimal Hermite polynomial coefficients (degrees 0, 4, 8)
to minimize the upper bound on A(f)*A(Fourier(f)) for the uncertainty inequality.

The test function is g(x) = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2/2)
where a12 is chosen so that g(0) = 0.

This version uses Differential Evolution (a global optimization algorithm)
with a reparameterization to handle the different scales of the coefficients.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import factorial


def _compute_hermite_moments():
    """
    Precompute the matrix C_{k,l} = integral x^2 H_k(x) H_l(x) exp(-x^2) dx
    for k, l in {0, 4, 8, 12}.
    """
    indices = [0, 4, 8, 12]
    n = len(indices)
    C = np.zeros((n, n))
    
    sqrt_pi = np.sqrt(np.pi)
    
    for i, k in enumerate(indices):
        for j, l in enumerate(indices):
            if l == k + 2:
                C[i, j] = 0.25 * (2 ** (k + 2)) * factorial(k + 2) * sqrt_pi
            elif l == k:
                C[i, j] = -(k + 0.5) * (2 ** k) * factorial(k) * sqrt_pi
            elif l == k - 2:
                C[i, j] = k * (k - 1) * (2 ** (k - 2)) * factorial(k - 2) * sqrt_pi
    
    return C, indices


# Precompute constants
C_MAT, INDICES = _compute_hermite_moments()
# Indices map: 0->0, 1->4, 2->8, 3->12
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 4790016.0

W = np.array([
    (2 ** 0) * factorial(0) * np.sqrt(np.pi),
    (2 ** 4) * factorial(4) * np.sqrt(np.pi),
    (2 ** 8) * factorial(8) * np.sqrt(np.pi),
    (2 ** 12) * factorial(12) * np.sqrt(np.pi)
])


def _compute_bound_analytic(a0, a4, a8):
    """
    Compute the uncertainty product A(g)*A(hat{g}) analytically.
    """
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    a = np.array([a0, a4, a8, a12])
    
    N = np.sum(a**2 * W)
    if N < 1e-15:
        return np.inf
    
    M_x = a @ C_MAT @ a
    
    M_xi = 0.0
    for i in range(4):
        for j in range(4):
            if i == j:
                M_xi += a[i] * a[j] * C_MAT[i, j]
            elif abs(INDICES[i] - INDICES[j]) == 2:
                M_xi -= a[i] * a[j] * C_MAT[i, j]
    
    if M_x < 0 or M_xi < 0:
        return np.inf
        
    return (M_x * M_xi) / (N * N)


def _objective_scaled(y):
    """
    Objective function for optimization using scaled variables.
    y[0] -> a0
    y[1] -> a4 * 100
    y[2] -> a8 * 10000
    """
    a0 = y[0]
    a4 = y[1] / 100.0
    a8 = y[2] / 10000.0
    
    try:
        val = _compute_bound_analytic(a0, a4, a8)
        if np.isnan(val) or np.isinf(val):
            return 1e10
        return val
    except Exception:
        return 1e10


def construct():
    """
    Find optimal coefficients a0, a4, a8 to minimize the uncertainty bound.
    Returns np.array([a0, a4, a8]).
    """
    # Seed solution in scaled coordinates
    seed_a0 = -113.0 / 100.0
    seed_a4 = 1.0 / 25.0
    seed_a8 = 1.0 / 3240.0
    
    seed_y = np.array([seed_a0, seed_a4 * 100.0, seed_a8 * 10000.0])
    
    # Define bounds for the scaled variables
    # a0 is around -1.1, let's allow [-5, 5]
    # a4 is around 0.04, scaled by 100 -> 4. Let's allow [-20, 20]
    # a8 is around 0.0003, scaled by 10000 -> 3. Let's allow [-10, 10]
    bounds = [(-5.0, 5.0), (-20.0, 20.0), (-10.0, 10.0)]
    
    best_val = _objective_scaled(seed_y)
    best_y = seed_y.copy()
    
    # Run Differential Evolution
    try:
        result = differential_evolution(
            _objective_scaled, 
            bounds, 
            seed=42, 
            maxiter=1000, 
            popsize=15, 
            tol=1e-10,
            polish=True
        )
        if result.fun < best_val:
            best_val = result.fun
            best_y = result.x
    except Exception:
        pass
        
    # Unscale the best solution
    best_a0 = best_y[0]
    best_a4 = best_y[1] / 100.0
    best_a8 = best_y[2] / 10000.0
    
    return np.array([best_a0, best_a4, best_a8])
# EVOLVE-BLOCK-END
