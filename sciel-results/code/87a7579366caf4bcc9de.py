"""Optimized program: analytic Hermite integrals + multi-start optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, L_BFGS_B

_START_TIME = time.time()
_DEADLINE = 175.0  # safety margin before 210s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _hermite_coeffs(c0, c4, c8):
    """Return all four coefficients given free ones, with g(0)=0 constraint."""
    # Probabilists' Hermite: H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395
    h0_0 = 1.0
    h4_0 = -3.0
    h8_0 = 105.0
    h12_0 = -10395.0
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    return c0, c4, c8, c12


def _compute_bound_analytic(c0, c4, c8):
    """
    Compute the upper bound using analytic Hermite polynomial properties.
    Uses O(1) computation via Hermite-Gaussian orthogonality relations.
    """
    c0, c4, c8, c12 = _hermite_coeffs(c0, c4, c8)
    
    # Coefficients array indexed by degree: [c0, c4, c8, c12]
    coeffs = np.array([c0, c4, c8, c12])
    degrees = np.array([0, 4, 8, 12])
    
    # N2 = ||f||_2^2 = sum_n c_n^2 * I_n where I_n = sqrt(pi) * 2^n * n!
    # For probabilists' Hermite: I_n = sqrt(pi) * 2^n * n!
    I = np.sqrt(np.pi) * 2.0**degrees * _factorial(degrees)
    
    N2 = np.sum(coeffs**2 * I)
    
    if N2 < 1e-15:
        return 1e10
    
    # M2 = ∫ x^2 |f(x)|^2 dx
    # Using x^2 H_n = H_{n+2} + (2n+1)H_n + n(n-1)H_{n-2}
    # M2 = sum_{n,m} c_n c_m * ∫ x^2 H_n H_m exp(-x^2) dx
    # By orthogonality, only terms where the resulting H_k matches H_m survive
    # ∫ x^2 H_n H_m exp(-x^2) dx = I_m * [δ_{m,n+2} + (2n+1)δ_{m,n} + n(n-1)δ_{m,n-2}]
    
    M2 = 0.0
    for i, n in enumerate(degrees):
        for j, m in enumerate(degrees):
            integral = 0.0
            if m == n + 2:
                integral += 1.0
            if m == n:
                integral += (2*n + 1)
            if m == n - 2:
                integral += n * (n - 1)
            M2 += coeffs[i] * coeffs[j] * I[j] * integral
    
    spread = M2 / N2
    C4 = spread * spread
    return C4


def _factorial(n):
    """Compute factorial for small integers (0 to 12)."""
    if isinstance(n, np.ndarray):
        return np.array([_factorial(int(x)) for x in n])
    result = 1
    for k in range(2, int(n) + 1):
        result *= k
    return result


def _compute_bound_fast(c0, c4, c8):
    """Fast version using precomputed constants for maximum speed."""
    c0, c4, c8, c12 = _hermite_coeffs(c0, c4, c8)
    
    # Precomputed I_n = sqrt(pi) * 2^n * n!
    I0 = np.sqrt(np.pi) * 1 * 1           # 0!
    I4 = np.sqrt(np.pi) * 16 * 24         # 4! = 24
    I8 = np.sqrt(np.pi) * 256 * 40320     # 8! = 40320
    I12 = np.sqrt(np.pi) * 4096 * 479001600  # 12! = 479001600
    
    coeffs = np.array([c0, c4, c8, c12])
    I_arr = np.array([I0, I4, I8, I12])
    
    N2 = np.sum(coeffs**2 * I_arr)
    
    if N2 < 1e-15:
        return 1e10
    
    # M2 computation using the formula
    degrees = np.array([0, 4, 8, 12])
    M2 = 0.0
    for i, n in enumerate(degrees):
        for j, m in enumerate(degrees):
            integral = 0.0
            if m == n + 2:
                integral += 1.0
            if m == n:
                integral += (2*n + 1)
            if m == n - 2:
                integral += n * (n - 1)
            M2 += coeffs[i] * coeffs[j] * I_arr[j] * integral
    
    spread = M2 / N2
    C4 = spread * spread
    return C4


def construct():
    """
    Find the three free coefficients (a0, a4, a8) for the Hermite combination
    H0 + a4*H4 + a8*H8 + a12*H12 (where a12 is fixed by P(0)=0) to minimize
    the upper bound C4 on A(f)*A(Fourier(f)).
    
    Returns:
        numpy array of shape (3,): [a0, a4, a8]
    """
    # Seed values
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _compute_bound_fast(*seed)
    best_params = seed.copy()
    
    # Try local optimization from seed with Nelder-Mead
    if _time_left() > 30:
        try:
            res = minimize(
                _compute_bound_fast,
                seed,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Try differential evolution for global search if time permits
    if _time_left() > 60:
        try:
            # Reasonable bounds for coefficients
            bounds = [(-50, 50), (-10, 10), (-0.5, 0.5)]
            res_de = differential_evolution(
                _compute_bound_fast,
                bounds,
                maxiter=300,
                popsize=20,
                seed=42,
                tol=1e-10,
                mutation=(0.5, 1.0)
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_params = res_de.x
        except Exception:
            pass
    
    # Refine with Nelder-Mead from best found
    if _time_left() > 20 and not np.allclose(best_params, seed, atol=1e-6):
        try:
            res2 = minimize(
                _compute_bound_fast,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x
        except Exception:
            pass
    
    # Additional refinement with L-BFGS-B if time permits
    if _time_left() > 10:
        try:
            res3 = minimize(
                _compute_bound_fast,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'gtol': 1e-12}
            )
            if res3.fun < best_val:
                best_val = res3.fun
                best_params = res3.x
        except Exception:
            pass
    
    # Final Nelder-Mead polish
    if _time_left() > 5:
        try:
            res4 = minimize(
                _compute_bound_fast,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-17}
            )
            if res4.fun < best_val:
                best_val = res4.fun
                best_params = res4.x
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
