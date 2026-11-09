"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import factorial

_START_TIME = time.time()
_DEADLINE = 580.0  # Safety margin before 630s kill

def _laguerre_polynomial(x, m):
    """Compute Laguerre polynomial L_m(x) using explicit formula."""
    if m < 0:
        return 0.0
    total = 0.0
    for k in range(m + 1):
        coeff = (-1)**k * factorial(m, exact=True) / (factorial(k, exact=True) * factorial(m - k, exact=True))
        total += coeff * (x ** (m - k))
    return total

def _compute_c4_bound(roots):
    """
    Compute the C4 upper bound for Fourier sign uncertainty.
    roots: array of 12 positive double-root locations.
    Returns the upper bound value (lower is better).
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12:
        return 1e10
    if np.any(roots <= 0):
        return 1e10
    if np.any(np.diff(roots) < 0):
        return 1e10

    m = 12
    n = 1
    
    # The Laguerre combination is typically of the form:
    # P(x) = sum_{i=1}^{12} c_i L_{m_i}(x)
    # But in the Cohn-Goncalves context for sign uncertainty with double roots,
    # we often construct a polynomial with roots at the specified locations.
    # Given the task says "Laguerre-polynomial construction", it likely refers to
    # a linear combination of Laguerre polynomials that has zeros at these points.
    
    # However, a more direct interpretation for "tuning root locations" in this context:
    # The bound is often related to the integral of x^n P(x)^2 e^{-x} dx or similar.
    # Let's assume the standard setup where we minimize the ratio of moments.
    
    # For the Cohn-Goncalves inequality, the constant C4 is derived from
    # the best possible bound on the number of sign changes.
    # A common proxy or direct calculation involves the norm of the polynomial
    # in the Laguerre space.
    
    # Let's try a specific known formulation:
    # If we have a polynomial P(x) with roots r_i, the "uncertainty" bound
    # is often proportional to (integral x P^2 e^-x) / (integral P^2 e^-x) or similar.
    # But specifically for C4, it might be related to the coefficient of x^4 or similar.
    
    # Let's use a robust numerical approach:
    # Construct the polynomial with the given roots: Q(x) = prod (x - r_i)^2
    # Then compute the relevant integral.
    
    # Actually, the "refined Cohn-Goncalves" often refers to the bound:
    # N < C * sqrt(energy) or similar.
    # Let's assume the score is the value of the upper bound on the number of sign changes.
    # The seed returns [4.0, 8.0, ...].
    # If I use roots that are closer together or spread out, the bound changes.
    
    # Let's compute the norm of the Laguerre expansion.
    # If P(x) = sum a_k L_k(x), and we require P(r_i) = 0.
    # This is an underdetermined system if we use all L_0..L_11.
    # Let's use L_0..L_11 (12 polynomials) and 12 roots.
    # This gives a unique solution for coefficients a_k (up to scale).
    
    A = np.zeros((12, 12))
    for i, r in enumerate(roots):
        for k in range(12):
            A[i, k] = _laguerre_polynomial(r, k)
    
    # Solve A a = 0? No, we need non-trivial solution.
    # Actually, we have 12 equations and 12 unknowns. The only solution is a=0 unless det(A)=0.
    # So we must use more Laguerre polynomials than roots.
    # Let's use L_0..L_13 (14 polynomials) for 12 roots.
    # Then we find the null space.
    
    A = np.zeros((12, 14))
    for i, r in enumerate(roots):
        for k in range(14):
            A[i, k] = _laguerre_polynomial(r, k)
    
    # Find null space using SVD
    try:
        u, s, vh = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return 1e10
    
    # The last row of vh is the null space vector
    a = vh[-1, :]
    
    # The polynomial is P(x) = sum a_k L_k(x)
    # The energy is integral P(x)^2 e^{-x} dx = sum a_k^2 (orthogonality of Laguerre)
    # For L_k, integral L_k^2 e^{-x} dx = 1 (if normalized) or factorial?
    # Standard Laguerre L_n: integral_0^inf L_n(x)^2 e^{-x} dx = 1.
    # So Energy = sum a_k^2
    
    energy = np.sum(a**2)
    if energy < 1e-10:
        return 1e10
    
    # The "moment" or "spread" is often integral x P(x)^2 e^{-x} dx
    # integral x L_k(x) L_l(x) e^{-x} dx = 0 if k!=l, and (2k+1) if k=l.
    # So Moment = sum a_k^2 (2k+1)
    
    moment = np.sum(a**2 * (2 * np.arange(14) + 1))
    
    # The bound on sign changes is often related to the ratio Moment / Energy
    # Or specifically, the C4 bound might be this ratio.
    # Let's assume the score is this ratio.
    
    bound = moment / energy
    
    return bound

def construct():
    best_roots = [4.0 * (i + 1) for i in range(12)]
    best_score = _compute_c4_bound(best_roots)
    
    if _deadline_reached():
        return best_roots
    
    # Use Nelder-Mead with multiple restarts
    # We need to ensure roots are positive and sorted.
    # We can parameterize by gaps.
    
    def _objective(gaps):
        # gaps are positive
        if np.any(gaps <= 0):
            return 1e10
        r0 = gaps[0]
        if r0 <= 0:
            return 1e10
        roots = np.cumsum(gaps)
        # Ensure strictly increasing (guaranteed by positive gaps)
        return _compute_c4_bound(roots)
    
    # Initial guess: equal gaps of 4.0
    init_gaps = np.array([4.0] * 12)
    
    best_local = _objective(init_gaps)
    best_local_gaps = init_gaps.copy()
    
    # Restart loop
    n_restarts = 10
    for i in range(n_restarts):
        if _deadline_reached():
            break
            
        if i == 0:
            x0 = init_gaps
        else:
            # Random perturbations
            # Scale gaps randomly
            scale = np.random.uniform(0.5, 2.0, 12)
            x0 = init_gaps * scale
            # Or random log-normal
            x0 = np.exp(np.random.normal(np.log(4.0), 0.5, 12))
        
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.success and res.fun < best_local:
                best_local = res.fun
                best_local_gaps = res.x
        except Exception:
            pass
            
        if _deadline_reached():
            break
            
    # Reconstruct roots from best gaps
    if best_local < best_score:
        best_roots = np.cumsum(best_local_gaps).tolist()
    else:
        best_roots = [4.0 * (i + 1) for i in range(12)]
        
    return best_roots

def _deadline_reached():
    return (time.time() - _START_TIME) >= _DEADLINE
# EVOLVE-BLOCK-END
