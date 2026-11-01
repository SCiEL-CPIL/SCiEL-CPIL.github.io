"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hermite_at_zero(n):
    """Return H_n(0) for physicists' Hermite polynomial."""
    if n % 2 != 0:
        return 0.0
    # H_{2k}(0) = (-1)^k (2k)! / k!
    k = n // 2
    from math import factorial
    val = factorial(2*k) / factorial(k)
    if k % 2 == 1:
        val = -val
    return float(val)


def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound on C4 given coefficients c0, c4, c8.
    
    The test function is g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12)(x) * exp(-x^2/2)
    where c12 is determined by g(0) = 0: c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    
    The Fourier transform: ĝ(ξ) = (c0*H0 + c4*(-1)^2*H4 + c8*(-1)^4*H8 + c12*(-1)^6*H12)(ξ) * exp(-ξ^2/2)
    Since (-i)^k for even k = (-1)^{k/2}:
      k=0: (-i)^0 = 1
      k=4: (-i)^4 = 1  
      k=8: (-i)^8 = 1
      k=12: (-i)^12 = 1
    
    Wait, (-i)^4 = (-i)^2 * (-i)^2 = (-1)*(-1) = 1. Yes, all even powers give +1.
    
    So ĝ(ξ) = (c0*H0 + c4*H4 + c8*H8 + c12*H12)(ξ) * exp(-ξ^2/2) = g(ξ)
    
    The function is Fourier self-dual (up to sign which is +1 for these even terms).
    
    The uncertainty bound involves moments of |g|^2 and |ĝ|^2.
    For the specific C4 bound, we need:
    
    A(f) = sqrt(<x^2> / <1>) in position space
    A(F) = sqrt(<ξ^2> / <1>) in frequency space
    
    But the "upper bound on C4" is a specific constant. Let me think about this differently.
    
    From the uncertainty principle literature, the bound on the product involves:
    C4 = (∫x^2|g|^2 dx / ∫|g|^2 dx) * (∫ξ^2|ĝ|^2 dξ / ∫|ĝ|^2 dξ)
    
    Since g is real and even, and Fourier self-dual, we can compute these moments.
    
    Actually, for the specific formulation in the uncertainty principle with Hermite polynomials,
    the bound C4 is related to the ratio of second moments.
    """
    H0_0 = _hermite_at_zero(0)   # 1
    H4_0 = _hermite_at_zero(4)   # -12
    H8_0 = _hermite_at_zero(8)   # 1680
    H12_0 = _hermite_at_zero(12) # -133056
    
    # c12 from g(0) = 0
    if abs(H12_0) < 1e-15:
        return np.inf
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # Now compute the moments. For g(x) = P(x) * exp(-x^2/2) where P is a polynomial,
    # we need ∫x^{2m} P(x)^2 exp(-x^2) dx / ∫P(x)^2 exp(-x^2) dx
    
    # This can be computed using the orthogonality of Hermite polynomials with weight exp(-x^2).
    # ∫ H_m(x) H_n(x) exp(-x^2) dx = sqrt(pi) * 2^n * n! * δ_{mn}
    
    # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # P(x)^2 = sum_{i,j} c_i c_j H_i H_j
    # ∫ P^2 exp(-x^2) dx = sum_i c_i^2 * sqrt(pi) * 2^i * i!  (by orthogonality)
    
    # For x^2 P^2: use x^2 = (H_2 + 2)/4 ... wait, let me use the recurrence.
    # Actually, x H_n = (1/2) H_{n+1} + n H_{n-1}
    # So x^2 H_n = (1/4) H_{n+2} + (n/2) H_n + (n/2) H_n + ... let me be more careful.
    # x H_n = (1/2) H_{n+1} + n H_{n-1}
    # x^2 H_n = x * [(1/2) H_{n+1} + n H_{n-1}] = (1/2) x H_{n+1} + n x H_{n-1}
    #          = (1/2)[(1/2) H_{n+2} + (n+1) H_n] + n[(1/2) H_n + (n-1) H_{n-2}]
    #          = (1/4) H_{n+2} + (n+1)/2 H_n + n/2 H_n + n(n-1)/2 H_{n-2}
    #          = (1/4) H_{n+2} + (2n+1)/2 H_n + n(n-1)/2 H_{n-2}
    
    # So x^2 H_n = (1/4) H_{n+2} + (n+0.5) H_n + 0.5*n*(n-1) H_{n-2}
    
    # Then ∫ x^2 P^2 exp(-x^2) dx = sum_{i,j} c_i c_j ∫ x^2 H_i H_j exp(-x^2) dx
    # = sum_{i,j} c_i c_j [(1/4) ∫ H_{i+2} H_j exp(-x^2) dx + (i+0.5) ∫ H_i H_j exp(-x^2) dx + 0.5*i*(i-1) ∫ H_{i-2} H_j exp(-x^2) dx]
    
    # By orthogonality, ∫ H_m H_n exp(-x^2) dx = sqrt(pi) * 2^n * n! * δ_{mn}
    
    # So only terms where indices match contribute.
    
    coeffs = np.array([c0, c4, c8, c12])
    degrees = np.array([0, 4, 8, 12])
    
    # Compute ∫ P^2 exp(-x^2) dx
    norm_sq = 0.0
    for i in range(4):
        n = degrees[i]
        norm_sq += coeffs[i] ** 2 * np.sqrt(np.pi) * (2.0 ** n) * _factorial(n)
    
    if abs(norm_sq) < 1e-30:
        return np.inf
    
    # Compute ∫ x^2 P^2 exp(-x^2) dx
    x2_norm = 0.0
    for i in range(4):
        for j in range(4):
            ni = degrees[i]
            nj = degrees[j]
            term = coeffs[i] * coeffs[j]
            # Contribution from (1/4) H_{ni+2} H_nj
            if ni + 2 == nj:
                x2_norm += term * 0.25 * np.sqrt(np.pi) * (2.0 ** nj) * _factorial(nj)
            # Contribution from (ni+0.5) H_ni H_nj
            if ni == nj:
                x2_norm += term * (ni + 0.5) * np.sqrt(np.pi) * (2.0 ** nj) * _factorial(nj)
            # Contribution from 0.5*ni*(ni-1) H_{ni-2} H_nj
            if ni - 2 == nj:
                x2_norm += term * 0.5 * ni * (ni - 1) * np.sqrt(np.pi) * (2.0 ** nj) * _factorial(nj)
    
    # A(f) * A(F) = sqrt(<x^2>) * sqrt(<ξ^2>) where <x^2> = ∫x^2|g|^2 / ∫|g|^2
    # Since g is Fourier self-dual, <ξ^2> = <x^2>
    # So A(f)*A(F) = <x^2> = x2_norm / norm_sq
    
    return x2_norm / norm_sq


def _factorial(n):
    """Compute n! for small n."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return float(result)


def construct():
    """
    Tune a Fourier-invariant Hermite-polynomial times Gaussian test function
    to reduce the upper bound on A(f)*A(Fourier(f)).
    
    Returns:
        np.array: three coefficients [c0, c4, c8] of the Hermite combination.
                  The degree-12 weight is fixed so that g(0) = 0.
    """
    start_time = time.time()
    deadline = start_time + 195  # 15s safety margin before 210s limit
    
    # Prior state of the art (Goncalves et al. 2017)
    best_result = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    best_score = _compute_bound(*best_result)
    
    # Try to improve using optimization
    def objective(x):
        return _compute_bound(x[0], x[1], x[2])
    
    # Multi-start local optimization
    starts = [
        best_result,
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-1.1, 0.05, 0.0002]),
        np.array([-0.9, 0.03, 0.0004]),
        np.array([-1.05, 0.045, 0.00035]),
        np.array([-1.15, 0.055, 0.00025]),
    ]
    
    for start in starts:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
    
    # Try L-BFGS-B from the best point found
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_result, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
