"""Rewritten program for the suite task; optimized for uncertainty inequality."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_hermite
import time
import math

_DEADLINE = 175  # safety margin before 210s kill

# Precompute exact coefficients for Hermite polynomials H0, H4, H8, H12
# H_n(x) in physicists' convention: H_0=1, H_1=2x, H_{n}=2xH_{n-1}-2(n-1)H_{n-2}
# We need coefficients in descending order for np.roots / np.polyval

def _hermite_coeffs_desc(n):
    """Return coefficients of H_n(x) in descending order (x^n, x^{n-1}, ..., 1)."""
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([2.0, 0.0])
    # Build up using recurrence
    # H_0 = 1
    # H_1 = 2x
    # H_n = 2x*H_{n-1} - 2(n-1)*H_{n-2}
    h_nm2 = np.array([1.0])  # H_0
    h_nm1 = np.array([2.0, 0.0])  # H_1
    for k in range(2, n + 1):
        # 2x * H_{k-1}: shift h_nm1 by one position to the right, multiply by 2
        term1 = np.zeros(len(h_nm1) + 1)
        term1[:-1] = 2.0 * h_nm1
        # -2(k-1) * H_{k-2}
        term2 = -2.0 * (k - 1) * h_nm2
        # Pad term2 to same length as term1
        if len(term2) < len(term1):
            term2 = np.pad(term2, (0, len(term1) - len(term2)))
        h_n = term1 + term2
        h_nm2 = h_nm1
        h_nm1 = h_n
    return h_nm1

H0_c = _hermite_coeffs_desc(0)   # [1]
H4_c = _hermite_coeffs_desc(4)   # [16, 0, -48, 0, 12]
H8_c = _hermite_coeffs_desc(8)   # [256, 0, -4480, 0, 26880, 0, -67200, 0, 1680]
H12_c = _hermite_coeffs_desc(12) # [4096, 0, -147456, 0, 2580480, 0, -20321280, 0, 81729600, 0, -172972800, 0, 103219200]

# Pad all to length 13 (degree 12) in descending order
def _pad_desc(c, length):
    if len(c) >= length:
        return c[:length]
    return np.pad(c, (0, length - len(c)))

H0_p = _pad_desc(H0_c, 13)
H4_p = _pad_desc(H4_c, 13)
H8_p = _pad_desc(H8_c, 13)
H12_p = _pad_desc(H12_c, 13)

# Values at x=0 (constant terms, last element in descending order)
H0_0 = H0_p[-1]   # 1.0
H4_0 = H4_p[-1]   # 12.0
H8_0 = H8_p[-1]   # 1680.0
H12_0 = H12_p[-1] # 103219200.0

# Precompute a dense grid for fallback / verification
_N_grid = 4000
_x_grid = np.linspace(-12, 12, _N_grid)

def _compute_bound(coeffs):
    """
    Compute the upper bound on A(f)*A(Fourier(f)) for the Hermite-Gaussian test function.
    
    Since all Hermite degrees are multiples of 4, the Fourier transform of g(x)=P(x)e^{-x^2/2}
    satisfies F(k) = sqrt(2*pi) * g(k) (up to a phase/sign that doesn't affect the bound).
    
    The bound is sqrt(2*pi) * (sup_x |x * g(x)|)^2.
    
    We find the supremum by locating critical points of h(x) = x*P(x)*exp(-x^2/2).
    h'(x) = [P(x) + x*P'(x) - x^2*P(x)] * exp(-x^2/2)
    So critical points are roots of Q(x) = P(x) + x*P'(x) - x^2*P(x).
    """
    a0, a4, a8 = coeffs
    
    # Enforce g(0) = 0 constraint
    d = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # Construct P(x) coefficients in descending order
    P_c = a0 * H0_p + a4 * H4_p + a8 * H8_p + d * H12_p
    
    # P'(x) coefficients in descending order
    # If P_c = [p0, p1, ..., p12] for x^12, x^11, ..., x^0
    # P'(x) = 12*p0*x^11 + 11*p1*x^10 + ... + p11
    P_prime_c = np.zeros(12)
    for i in range(12):
        P_prime_c[i] = (12 - i) * P_c[i]
    
    # Q(x) = P(x) + x*P'(x) - x^2*P(x)
    # x*P'(x): shift P_prime_c down by one (multiply by x), so length 13
    x_P_prime_c = np.zeros(13)
    x_P_prime_c[1:] = P_prime_c
    
    # -x^2*P(x): shift P_c down by two, multiply by -1
    neg_x2_P_c = np.zeros(13)
    neg_x2_P_c[2:] = -P_c[:-2]
    
    Q_c = P_c + x_P_prime_c + neg_x2_P_c
    
    # Remove leading zeros for np.roots
    # Find first non-zero coefficient
    nz = np.nonzero(Q_c)[0]
    if len(nz) == 0:
        # Q is identically zero, fallback to grid
        P_vals = np.polyval(P_c, _x_grid)
        g_vals = P_vals * np.exp(-_x_grid**2 / 2.0)
        xg = _x_grid * g_vals
        Af = np.max(np.abs(xg))
        return math.sqrt(2 * math.pi) * Af * Af
    
    Q_c_trimmed = Q_c[nz[0]:]
    
    # Find roots of Q
    try:
        roots = np.roots(Q_c_trimmed)
    except Exception:
        # Fallback to grid
        P_vals = np.polyval(P_c, _x_grid)
        g_vals = P_vals * np.exp(-_x_grid**2 / 2.0)
        xg = _x_grid * g_vals
        Af = np.max(np.abs(xg))
        return math.sqrt(2 * math.pi) * Af * Af
    
    # Keep only real roots
    real_roots = roots[np.abs(roots.imag) < 1e-8].real
    
    if len(real_roots) == 0:
        # Fallback to grid
        P_vals = np.polyval(P_c, _x_grid)
        g_vals = P_vals * np.exp(-_x_grid**2 / 2.0)
        xg = _x_grid * g_vals
        Af = np.max(np.abs(xg))
    else:
        # Evaluate h(x) at real critical points
        P_vals = np.polyval(P_c, real_roots)
        g_vals = P_vals * np.exp(-real_roots**2 / 2.0)
        xg = real_roots * g_vals
        Af = np.max(np.abs(xg))
        
        # Also check a few points on the grid to be safe (in case of numerical issues)
        P_vals_grid = np.polyval(P_c, _x_grid)
        g_vals_grid = P_vals_grid * np.exp(-_x_grid**2 / 2.0)
        xg_grid = _x_grid * g_vals_grid
        Af_grid = np.max(np.abs(xg_grid))
        Af = max(Af, Af_grid)
    
    bound = math.sqrt(2 * math.pi) * Af * Af
    return bound

def _objective(params):
    """Objective function wrapper."""
    try:
        return _compute_bound(params)
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    
    # Seed solution
    seed = np.array([-1.13, 0.04, 1/3240.0])
    
    best_result = seed.copy()
    best_score = _compute_bound(seed)
    
    # Phase 1: Differential evolution for global search
    if time.time() < _DEADLINE - 40:
        try:
            bounds = [(-3.0, 0.5), (-0.3, 0.3), (-0.01, 0.015)]
            de_result = differential_evolution(
                _objective, bounds,
                maxiter=80, popsize=20,
                tol=1e-10, seed=42,
                polish=True,
                init='sobol',
                mutation=(0.5, 1.5),
                recombination=0.8
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_result = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement with multiple diverse starting points
    start_points = [
        best_result.copy(),
        seed.copy(),
        np.array([-1.1, 0.035, 0.00028]),
        np.array([-1.15, 0.045, 0.00032]),
        np.array([-1.05, 0.03, 0.00025]),
        np.array([-1.2, 0.05, 0.00035]),
        np.array([-0.95, 0.04, 0.0003]),
        np.array([-1.13, 0.02, 0.0002]),
        np.array([-1.13, 0.06, 0.0004]),
    ]
    
    for sp in start_points:
        if time.time() > _DEADLINE - 20:
            break
        try:
            # BFGS for smooth optimization
            res = minimize(_objective, sp, method='BFGS',
                          options={'maxiter': 500, 'gtol': 1e-12, 'disp': False})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Random restarts with Powell for robustness
    if time.time() < _DEADLINE - 15:
        np.random.seed(123)
        n_random = 50
        for _ in range(n_random):
            if time.time() > _DEADLINE - 15:
                break
            r_a0 = np.random.uniform(-2.5, 0.0)
            r_a4 = np.random.uniform(-0.2, 0.2)
            r_a8 = np.random.uniform(-0.008, 0.012)
            start_point = np.array([r_a0, r_a4, r_a8])
            
            try:
                res = minimize(_objective, start_point, method='Powell',
                             options={'maxiter': 150, 'ftol': 1e-12, 'disp': False})
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res.x.copy()
            except Exception:
                pass
    
    # Phase 4: Final high-precision polish
    if time.time() < _DEADLINE - 5:
        try:
            # Nelder-Mead with very tight tolerances
            result = minimize(_objective, best_result, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True, 'disp': False})
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass
        
        # BFGS final pass
        try:
            result = minimize(_objective, best_result, method='BFGS',
                             options={'maxiter': 1000, 'gtol': 1e-15, 'disp': False})
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass
        
        # One more Nelder-Mead polish
        try:
            result = minimize(_objective, best_result, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True, 'disp': False})
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_result)

# EVOLVE-BLOCK-END
