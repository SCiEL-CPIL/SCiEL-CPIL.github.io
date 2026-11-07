# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

_T0 = time.time()
_DEADLINE = 180.0  # Safety margin below the 210s kill

# ---------- Precompute Hermite Polynomial Coefficients ----------
# We need the coefficients of H0, H4, H8, H12 in the form of a vector of length 13 (degree 0 to 12).
# Physicists' Hermite polynomials:
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 256x^8 - 4480x^6 + 26880x^4 - 67200x^2 + 40320
# H_12(x) = 4096x^12 - 147456x^10 + 2903040x^8 - 29030400x^6 + 145152000x^4 - 393216000x^2 + 1032160

# Coefficients [c0, c1, ..., c12] where c_i is the coefficient of x^i
_H0_coeffs = np.zeros(13)
_H0_coeffs[0] = 1.0

_H4_coeffs = np.zeros(13)
_H4_coeffs[0] = 12.0
_H4_coeffs[2] = -48.0
_H4_coeffs[4] = 16.0

_H8_coeffs = np.zeros(13)
_H8_coeffs[0] = 40320.0
_H8_coeffs[2] = -67200.0
_H8_coeffs[4] = 26880.0
_H8_coeffs[6] = -4480.0
_H8_coeffs[8] = 256.0

_H12_coeffs = np.zeros(13)
_H12_coeffs[0] = 1032160.0
_H12_coeffs[2] = -393216000.0
_H12_coeffs[4] = 145152000.0
_H12_coeffs[6] = -29030400.0
_H12_coeffs[8] = 2903040.0
_H12_coeffs[10] = -147456.0
_H12_coeffs[12] = 4096.0

# Values at x=0
_H0_0 = _H0_coeffs[0]
_H4_0 = _H4_coeffs[0]
_H8_0 = _H8_coeffs[0]
_H12_0 = _H12_coeffs[0]

def _q_coeffs(c0, c1, c2):
    """
    Compute the coefficients of q(x) = g(x)/x^2.
    g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    c3 = -(c0*H0(0) + c1*H4(0) + c2*H8(0)) / H12(0)
    """
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    
    # g(x) coefficients (degree 0 to 12)
    g = c0 * _H0_coeffs + c1 * _H4_coeffs + c2 * _H8_coeffs + c3 * _H12_coeffs
    
    # q(x) = g(x) / x^2. Since g is even, g[1] is 0.
    # q coefficients are g[2], g[3], ..., g[12] shifted to degree 0 to 10.
    # However, g is even, so only even powers exist.
    # q(x) = sum_{k=1}^6 g[2k] x^{2k-2}
    # Let's just extract g[2:] and keep even indices for the z-polynomial, 
    # but np.roots works on the full polynomial vector.
    # q(x) is a polynomial of degree 10.
    # Coeffs of q: q[0] = g[2], q[1] = g[3]=0, q[2] = g[4], ...
    q_full = g[2:] 
    return q_full

def _smallest_positive_sign_changing_root(q_coeffs):
    """
    Find the smallest positive real root of the polynomial q_coeffs (low->high) 
    where the polynomial changes sign.
    """
    # Remove leading zeros to ensure correct degree for np.roots
    qc = q_coeffs.copy()
    while len(qc) > 1 and abs(qc[-1]) < 1e-12:
        qc = qc[:-1]
    
    if len(qc) < 2:
        return np.inf
        
    try:
        roots = np.roots(qc)
    except Exception:
        return np.inf
        
    best_r = np.inf
    for rt in roots:
        # Check if real
        if abs(rt.imag) > 1e-9 * max(1.0, abs(rt.real)):
            continue
        zr = float(rt.real)
        if zr <= 1e-12:
            continue
            
        # Check sign change.
        # A root is a sign change if the multiplicity is odd.
        # Numerically, we can check the sign of the polynomial at zr - eps and zr + eps.
        # However, if the root is very close to 0 or if eps is too large/small, this can fail.
        # A more robust check for simple roots: derivative is non-zero.
        # Let's use the derivative check.
        d_coeffs = np.polyder(np.flip(qc)) # High->Low for derivative
        d_val = np.polyval(d_coeffs, zr)
        
        # If derivative is close to zero, it might be a double root (no sign change) or ill-conditioned.
        # We require a sign change, so we reject double roots.
        if abs(d_val) < 1e-8 * max(1.0, abs(zr)):
            continue
            
        # Verify the root is actually a root (residual check)
        q_val = np.polyval(np.flip(qc), zr)
        if abs(q_val) > 1e-6 * max(1.0, abs(zr)):
            continue
            
        if zr < best_r:
            best_r = zr
            
    return best_r

def _objective(params):
    """
    Objective function for optimization.
    Returns value = r^2 / (2*pi). Lower is better.
    """
    c0, c1, c2 = params
    try:
        qc = _q_coeffs(c0, c1, c2)
        r = _smallest_positive_sign_changing_root(qc)
        if not np.isfinite(r) or r <= 0:
            return 1e12
        return r * r / (2.0 * np.pi)
    except Exception:
        return 1e12

def construct():
    global _T0
    try:
        # Seed
        seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
        best_val = _objective(seed)
        best_x = seed.copy()
        
        # 1. Differential Evolution for Global Search
        # Bounds: The seed is small. Let's search in a box around it.
        # If the true optimum has larger coefficients, we might miss it, 
        # but the problem structure (Hermite polynomials growing fast) suggests 
        # that large coefficients lead to large roots unless there's massive cancellation.
        # Given the seed is [-1.13, 0.04, 0.0003], a box of [-5, 5] is reasonable.
        bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
        
        # Use DE with a moderate population to stay within time limits.
        # maxiter=100, popsize=20 -> 2000 evaluations. Each eval is ~ms.
        # This is fast enough.
        if time.time() - _T0 < _DEADLINE - 30:
            try:
                res = differential_evolution(
                    _objective, 
                    bounds, 
                    seed=42, 
                    maxiter=150, 
                    popsize=30, 
                    tol=1e-10,
                    polish=False # We will polish manually to control time
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except Exception:
                pass

        # 2. Local Refinement (Nelder-Mead) on the best candidate
        # DE finds a good basin, NM polishes it to high precision.
        if time.time() - _T0 < _DEADLINE - 10 and best_val < 1e10:
            try:
                res_nm = minimize(
                    _objective, 
                    best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                )
                if res_nm.fun < best_val:
                    best_val = res_nm.fun
                    best_x = res_nm.x
            except Exception:
                pass

        # 3. Final Safety Check
        if not np.isfinite(best_val) or best_val >= 1e10:
            # Fallback to seed if search failed
            best_x = seed
            best_val = _objective(seed)

        return np.array([float(best_x[0]), float(best_x[1]), float(best_x[2])])
        
    except Exception:
        # Absolute fallback
        seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
        return np.array([float(seed[0]), float(seed[1]), float(seed[2])])

# EVOLVE-BLOCK-END
