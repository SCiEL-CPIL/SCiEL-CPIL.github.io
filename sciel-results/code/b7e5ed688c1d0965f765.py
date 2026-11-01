"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
import time

_START_TIME = time.time()
_DEADLINE = 190  # safety margin before 210s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _hermite_coeffs():
    """Return coefficient vectors for H0, H4, H8, H12 as arrays of powers 0..12."""
    # H0(x) = 1
    H0 = np.zeros(13)
    H0[0] = 1.0
    # H4(x) = 16x^4 - 48x^2 + 12
    H4 = np.zeros(13)
    H4[4] = 16.0
    H4[2] = -48.0
    H4[0] = 12.0
    # H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    H8 = np.zeros(13)
    H8[8] = 256.0
    H8[6] = -3072.0
    H8[4] = 11520.0
    H8[2] = -13440.0
    H8[0] = 1680.0
    # H12(x) = 4096x^12 - 73728x^10 + 516096x^8 - 1693440x^6 + 2626560x^4 - 1741824x^2 + 221760
    H12 = np.zeros(13)
    H12[12] = 4096.0
    H12[10] = -73728.0
    H12[8] = 516096.0
    H12[6] = -1693440.0
    H12[4] = 2626560.0
    H12[2] = -1741824.0
    H12[0] = 221760.0
    return H0, H4, H8, H12

_H0, _H4, _H8, _H12 = _hermite_coeffs()

def _poly_coeffs(c0, c4, c8):
    """Compute full polynomial coefficients with c12 determined by P(0)=0."""
    # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # P(0) = c0*1 + c4*12 + c8*1680 + c12*221760 = 0
    c12 = -(c0 * _H0[0] + c4 * _H4[0] + c8 * _H8[0]) / _H12[0]
    coeffs = c0 * _H0 + c4 * _H4 + c8 * _H8 + c12 * _H12
    return coeffs

def _g_eval(x, coeffs):
    """Evaluate P(x) * exp(-x^2) where P has coefficients in `coeffs` (powers 0..12)."""
    # Horner's method
    p = np.zeros_like(x, dtype=float)
    for i in range(12, -1, -1):
        p = p * x + coeffs[i]
    return p * np.exp(-x**2)

def _g_abs_integral(coeffs):
    """Compute integral of |g(x)| dx over real line."""
    # g(x) is odd/even? Let's check: H0, H4, H8, H12 are all even polynomials.
    # So P(x) is even, e^{-x^2} is even, so g(x) is even.
    # But g(x) can change sign! So we need to find zeros.
    # For numerical integration, use quad with points at suspected zeros.
    
    # First, let's try direct integration with fine subdivision
    # g(x) = P(x)*e^{-x^2}, P is degree 12 even polynomial
    # Zeros of P are symmetric. Find them numerically.
    
    # Build polynomial for finding roots
    # coeffs[i] is coefficient of x^i
    # np.roots expects highest power first
    poly_roots = np.roots(coeffs[12::-1])
    real_roots = np.real(poly_roots[np.abs(poly_roots.imag) < 1e-10])
    real_roots = np.sort(real_roots)
    
    # Integration points: include roots and some breakpoints
    # Integrate |g(x)| from -inf to inf. Since g is even, integrate 0 to inf and double.
    # But |g| may not be even if g changes sign... actually g is even so |g| is even.
    
    # Break points in [0, inf]
    breakpoints = [0.0]
    for r in real_roots:
        if r > 1e-10:
            breakpoints.append(r)
    # Add some extra points for safety
    breakpoints = sorted(set(breakpoints))
    # Cap at reasonable number
    if len(breakpoints) > 50:
        breakpoints = breakpoints[:50]
    
    # Integrate piecewise
    total = 0.0
    for i in range(len(breakpoints) - 1):
        a = breakpoints[i]
        b = breakpoints[i + 1]
        val, _ = quad(lambda x: abs(_g_eval(x, coeffs)), a, b, limit=200)
        total += val
    
    # Last interval to infinity
    a = breakpoints[-1]
    val, _ = quad(lambda x: abs(_g_eval(x, coeffs)), a, np.inf, limit=200)
    total += val
    
    # Double for symmetry
    return 2.0 * total

def _fourier_transform_eval(w, coeffs):
    """Evaluate Fourier transform of g(x) = P(x)*e^{-x^2} at frequency w.
    
    Using: F[P(x)*e^{-x^2}](w) = (1/sqrt(2*pi)) * integral P(x)*e^{-x^2}*e^{-iwx} dx
    
    We use the relation: if f(x) = e^{-x^2}, F(f)(w) = sqrt(pi)*e^{-w^2/4} (with 1/sqrt(2*pi) convention)
    Actually, let's use the convention F(f)(w) = integral f(x) e^{-iwx} dx (no 1/sqrt(2pi))
    
    For P(x)*e^{-x^2}, we can use: x^n e^{-x^2} has FT related to derivatives of e^{-w^2/4}.
    
    More practically: use the fact that multiplication by x corresponds to i d/dw in FT.
    FT[e^{-x^2}](w) = sqrt(pi) * e^{-w^2/4}
    FT[x^n e^{-x^2}](w) = (i d/dw)^n [sqrt(pi) e^{-w^2/4}]
    
    So if P(x) = sum_{k=0}^{12} c_k x^k, then
    FT[g](w) = sum_{k=0}^{12} c_k * (i d/dw)^k [sqrt(pi) e^{-w^2/4}]
    
    Let's compute this using Hermite polynomial relations or direct computation.
    
    Alternative: use the fact that the FT of x^n e^{-x^2} can be computed recursively.
    Let G_n(w) = FT[x^n e^{-x^2}](w)
    G_0(w) = sqrt(pi) e^{-w^2/4}
    G_{n+1}(w) = i d/dw G_n(w)
    
    This is doable but let's use numerical approach for robustness:
    Just compute the integral numerically.
    """
    # For efficiency, use the analytical derivative approach
    # FT[x^n e^{-x^2}](w) = (i)^n * d^n/dw^n [sqrt(pi) e^{-w^2/4}]
    
    # Let's compute using the recurrence with automatic differentiation-like approach
    # Actually, let's just do numerical quadrature for the FT magnitude
    
    # g(x) is even, so FT[g](w) = integral_0^inf 2*g(x)*cos(wx) dx (real part, since g is even and real)
    # Actually FT[g](w) = integral_{-inf}^{inf} g(x) e^{-iwx} dx = 2*integral_0^inf g(x) cos(wx) dx (since g is even)
    
    # But we need |FT[g](w)|. Since g is even and real, FT[g](w) is real and even.
    # So |FT[g](w)| = |2 * integral_0^inf g(x) cos(wx) dx|
    
    val, _ = quad(lambda x: 2.0 * _g_eval(x, coeffs) * np.cos(w * x), 0, np.inf, limit=200)
    return val

def _fourier_abs_integral(coeffs):
    """Compute integral of |FT[g](w)| dw over real line."""
    # FT[g](w) is even and real (since g is even and real)
    # So integrate |FT[g](w)| from 0 to inf and double
    
    # Need to find zeros of FT[g](w) to break up the absolute value
    # This is harder. Let's use adaptive integration with many subintervals.
    
    # Strategy: sample FT[g](w) on a grid to find sign changes, then integrate piecewise
    w_max = 20.0  # FT decays as e^{-w^2/4} times polynomial, so beyond ~10 it's tiny
    n_samples = 200
    w_grid = np.linspace(0, w_max, n_samples)
    ft_vals = np.array([_fourier_transform_eval(w, coeffs) for w in w_grid])
    
    # Find sign changes
    signs = np.sign(ft_vals)
    breakpoints = [0.0]
    for i in range(n_samples - 1):
        if signs[i] != signs[i + 1] and signs[i] != 0 and signs[i + 1] != 0:
            # Linear interpolation to find zero
            w0 = w_grid[i]
            w1 = w_grid[i + 1]
            f0 = ft_vals[i]
            f1 = ft_vals[i + 1]
            wz = w0 - f0 * (w1 - w0) / (f1 - f0)
            if wz > breakpoints[-1] + 1e-6:
                breakpoints.append(wz)
    
    breakpoints.append(w_max)
    
    total = 0.0
    for i in range(len(breakpoints) - 1):
        a = breakpoints[i]
        b = breakpoints[i + 1]
        val, _ = quad(lambda w: abs(_fourier_transform_eval(w, coeffs)), a, b, limit=100)
        total += val
    
    # Tail from w_max to inf
    val, _ = quad(lambda w: abs(_fourier_transform_eval(w, coeffs)), w_max, np.inf, limit=50)
    total += val
    
    return 2.0 * total

def _score(c0, c4, c8):
    """Compute the uncertainty product A(g)*A(FT[g])."""
    coeffs = _poly_coeffs(c0, c4, c8)
    a_g = _g_abs_integral(coeffs)
    a_ft = _fourier_abs_integral(coeffs)
    return a_g * a_ft

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Seed values
    best = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_score = _score(best[0], best[1], best[2])
    
    # Try some hand-tuned perturbations first
    candidates = [
        best.copy(),
        np.array([-1.1, 0.04, 0.0003]),
        np.array([-1.15, 0.045, 0.00031]),
        np.array([-1.12, 0.039, 0.000305]),
        np.array([-1.14, 0.041, 0.000315]),
    ]
    
    for c in candidates:
        if _time_left() < 30:
            break
        try:
            s = _score(c[0], c[1], c[2])
            if s < best_score:
                best_score = s
                best = c.copy()
        except Exception:
            pass
    
    # Local optimization using Nelder-Mead
    if _time_left() > 60:
        try:
            def objective(x):
                return _score(x[0], x[1], x[2])
            
            res = minimize(objective, best, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if _time_left() > 0:
                s = res.fun
                if s < best_score:
                    best_score = s
                    best = res.x.copy()
        except Exception:
            pass
    
    # Try a few more random perturbations around best
    if _time_left() > 30:
        rng = np.random.RandomState(42)
        for _ in range(20):
            if _time_left() < 15:
                break
            pert = best + rng.normal(0, 0.01, 3)
            try:
                s = _score(pert[0], pert[1], pert[2])
                if s < best_score:
                    best_score = s
                    best = pert.copy()
            except Exception:
                pass
    
    return best
# EVOLVE-BLOCK-END
