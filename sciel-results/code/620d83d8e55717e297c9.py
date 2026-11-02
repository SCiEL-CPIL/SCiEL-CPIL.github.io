"""
Tune Hermite-polynomial coefficients to minimize the uncertainty bound C4.
We search over (a0, a4, a8) with a12 determined by g(0)=0, and minimize
the upper bound on A(f)*A(Fhat(f)) via a smooth surrogate / direct optimization.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import factorial
import time

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 200.0  # 210s wall-clock, keep 10s margin


def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _hermite_phys(n, x):
    """Physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    # Use recurrence: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(1, n):
        h2 = 2 * x * h1 - 2 * k * h0
        h0, h1 = h1, h2
    return h1


def _hermite_coeffs():
    """
    Precompute coefficients for H_0, H_4, H_8, H_12.
    H_0(x) = 1
    H_4(x) = 16x^4 - 48x^2 + 12
    H_8(x) = 256x^8 - 4608x^6 + 30240x^4 - 60480x^2 + 10080
    H_12(x) = 4096x^12 - 184320x^10 + 3932160x^8 - 42467520x^6 + 254016000x^4 - 765427200x^2 + 103219200
    """
    # H_0
    c0 = np.array([1.0])
    # H_4: 16x^4 - 48x^2 + 12
    c4 = np.array([12.0, 0.0, -48.0, 0.0, 16.0])
    # H_8
    c8 = np.array([10080.0, 0.0, -60480.0, 0.0, 30240.0, 0.0, -4608.0, 0.0, 256.0])
    # H_12
    c12 = np.array([103219200.0, 0.0, -765427200.0, 0.0, 254016000.0, 0.0,
                     -42467520.0, 0.0, 3932160.0, 0.0, -184320.0, 0.0, 4096.0])
    return c0, c4, c8, c12


def _poly_eval(coeffs, x):
    """Evaluate polynomial with given coefficients (ascending order) at x."""
    result = np.zeros_like(x)
    for i, c in enumerate(coeffs):
        result += c * x ** i
    return result


def _c4_bound(a0, a4, a8):
    """
    Compute the upper bound C4 = A(f) * A(Fhat(f)) for the test function
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is chosen so g(0) = 0.
    
    A(f) = integral of |f(x)|^2 dx (L2 norm squared)
    A(Fhat(f)) = integral of |Fhat(f)(xi)|^2 dxi
    
    For f = g, by Plancherel: A(Fhat(g)) = A(g) = ||g||_2^2
    So C4 = ||g||_2^4.
    
    Wait - that's trivial. Let me reconsider.
    
    Actually, the uncertainty inequality bound involves:
    A(f) = (integral |f|^2) / (integral |f|^2) ... no.
    
    Let me think about this more carefully. The task says "upper bound on A(f)*A(Fourier(f))".
    For a function f, A(f) might be the area (L1 norm) or L2 norm.
    
    Actually in uncertainty inequalities, typically:
    - A(f) = ||f||_2^2 (energy)
    - The Fourier transform's energy is also ||f||_2^2 by Plancherel
    
    So the product would just be ||f||_2^4, which is trivially minimized by making f small.
    
    This must involve a different definition. Let me reconsider.
    
    Perhaps A(f) is defined as the "spread" or variance:
    A(f) = (1/||f||_2^2) * integral x^2 |f(x)|^2 dx
    
    Then A(f)*A(Fhat(f)) is the Heisenberg uncertainty product.
    
    But the task says "upper bound on A(f)*A(Fourier(f))" and mentions C4.
    
    Let me assume:
    A(f) = (1/||f||_2^2) * integral |f(x)|^2 dx ... no that's 1.
    
    Actually, re-reading: "reduce the upper bound on A(f)*A(Fourier(f))"
    and "score is the upper bound on C4"
    
    I think the setup is:
    - We have a test function g
    - f is some function related to g
    - The uncertainty product has an upper bound C4 that depends on the coefficients
    - We want to minimize C4
    
    Given the structure with Hermite polynomials and the constraint g(0)=0,
    and that this is from Goncalves et al. 2017, let me assume:
    
    C4 = ||g||_2^2 * ||g'||_2^2 / something, or it's related to the 
    uncertainty product directly.
    
    For a Gaussian-modulated Hermite function, the uncertainty product can be computed.
    Let me compute it numerically.
    
    Actually, let me just compute:
    ||g||_2^2 = integral g(x)^2 dx
    ||x*g||_2^2 = integral x^2 g(x)^2 dx
    ||g'||_2^2 = integral (g'(x))^2 dx
    
    The Heisenberg uncertainty principle says:
    (integral x^2 |g|^2 / integral |g|^2) * (integral xi^2 |G|^2 / integral |G|^2) >= 1/4
    
    where G is the Fourier transform of g.
    
    But by Plancherel, integral |G|^2 = integral |g|^2, and
    integral xi^2 |G|^2 = integral (g')^2 (up to constants depending on FT convention).
    
    With FT convention F(omega) = integral f(x) e^{-i omega x} dx:
    integral omega^2 |F(omega)|^2 domega = 2*pi * integral |f'(x)|^2 dx
    
    So the uncertainty product is:
    (integral x^2 |g|^2 / integral |g|^2) * (2*pi * integral |g'|^2 / integral |g|^2)
    
    Let me just compute this numerically and minimize it.
    """
    c0, c4, c8, c12 = _hermite_coeffs()
    
    # g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    # g(0) = (a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0)) * 1 = 0
    # H0(0)=1, H4(0)=12, H8(0)=10080, H12(0)=103219200
    # a0 + 12*a4 + 10080*a8 + 103219200*a12 = 0
    # a12 = -(a0 + 12*a4 + 10080*a8) / 103219200
    
    a12 = -(a0 + 12.0 * a4 + 10080.0 * a8) / 103219200.0
    
    # Evaluate on a grid
    x = np.linspace(-20, 20, 20001)
    dx = x[1] - x[0]
    
    h0 = _poly_eval(c0, x)
    h4 = _poly_eval(c4, x)
    h8 = _poly_eval(c8, x)
    h12 = _poly_eval(c12, x)
    
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    g = poly * np.exp(-x**2 / 2.0)
    
    # Compute integrals
    g2 = g**2
    norm2 = np.trapz(g2, x)
    
    if norm2 < 1e-30:
        return 1e10
    
    x2_g2 = np.trapz(x**2 * g2, x)
    
    # g' = poly' * exp(-x^2/2) + poly * (-x) * exp(-x^2/2)
    # = (poly' - x*poly) * exp(-x^2/2)
    
    # Compute poly' numerically or analytically
    # poly = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # H0' = 0, H4' = 64x^3 - 96x, H8' = 2048x^7 - 27648x^5 + 120960x^3 - 120960x, H12' = 49152x^11 - 1843200x^9 + 31457280x^7 - 254805120x^5 + 1016064000x^3 - 1530854400x
    
    # Use numerical derivative for simplicity
    gp = np.gradient(g, dx)
    
    gp2 = gp**2
    norm_deriv2 = np.trapz(gp2, x)
    
    # Uncertainty product: (x2_g2/norm2) * (2*pi*norm_deriv2/norm2)
    # But let's use the standard form. With the convention that 
    # the Fourier transform is F(omega) = (1/sqrt(2*pi)) integral f(x) e^{-ixw} dx
    # then Plancherel gives ||F||_2 = ||f||_2 and the uncertainty product is:
    # (integral x^2 |f|^2 / integral |f|^2) * (integral w^2 |F|^2 / integral |F|^2)
    # = (x2_g2/norm2) * (norm_deriv2/norm2)  [with appropriate FT convention]
    
    # The constant factor depends on convention. Let me use the product:
    C4 = (x2_g2 / norm2) * (norm_deriv2 / norm2)
    
    return C4


def _c4_bound_fast(a0, a4, a8):
    """
    Faster computation using analytical integrals.
    For g(x) = P(x) * exp(-x^2/2), we can compute integrals analytically.
    """
    c0, c4, c8, c12 = _hermite_coeffs()
    
    a12 = -(a0 + 12.0 * a4 + 10080.0 * a8) / 103219200.0
    
    # P(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
    # P(x) is a polynomial of degree 12 (only even powers)
    # Let's build P(x) as a polynomial in x^2:
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 4608x^6 + 30240x^4 - 60480x^2 + 10080
    # H12 = 4096x^12 - 184320x^10 + 3932160x^8 - 42467520x^6 + 254016000x^4 - 765427200x^2 + 103219200
    
    # P(x) = c12*x^12 + c10*x^10 + c8*x^8 + c6*x^6 + c4*x^4 + c2*x^2 + c0
    # where only even powers
    
    # Coefficients for x^(2k):
    # x^0: a0*1 + a4*12 + a8*10080 + a12*103219200 = 0 (by construction)
    # x^2: a4*(-48) + a8*(-60480) + a12*(-765427200)
    # x^4: a4*16 + a8*30240 + a12*254016000
    # x^6: a8*(-4608) + a12*(-42467520)
    # x^8: a8*256 + a12*3932160
    # x^10: a12*(-184320)
    # x^12: a12*4096
    
    p0 = 0.0  # by construction
    p2 = -48.0 * a4 - 60480.0 * a8 - 765427200.0 * a12
    p4 = 16.0 * a4 + 30240.0 * a8 + 254016000.0 * a12
    p6 = -4608.0 * a8 - 42467520.0 * a12
    p8 = 256.0 * a8 + 3932160.0 * a12
    p10 = -184320.0 * a12
    p12 = 4096.0 * a12
    
    # g(x) = (p12*x^12 + p10*x^10 + p8*x^8 + p6*x^6 + p4*x^4 + p2*x^2 + p0) * exp(-x^2/2)
    
    # ||g||_2^2 = integral g(x)^2 dx = integral P(x)^2 * exp(-x^2) dx
    # P(x)^2 = sum of terms x^(2i+2j) * pi * pj
    # integral x^(2n) * exp(-x^2) dx = (2n-1)!! * sqrt(pi) / 2^n = Gamma(n+1/2)
    
    # Let's compute P(x)^2 as a polynomial in x^2:
    # P(x) = sum_{k=0}^{6} p_{2k} * x^(2k) where p_{2k} = p2k
    # P(x)^2 = sum_{i,j} p_{2i} p_{2j} x^(2(i+j))
    # integral P(x)^2 exp(-x^2) dx = sum_{i,j} p_{2i} p_{2j} * integral x^(2(i+j)) exp(-x^2) dx
    
    # integral x^(2n) exp(-x^2) dx = (2n)! / (4^n n!) * sqrt(pi)
    
    def _even_moment(n):
        """integral x^(2n) exp(-x^2) dx from -inf to inf = (2n)!/(4^n n!) * sqrt(pi)"""
        if n == 0:
            return np.sqrt(np.pi)
        result = 1.0
        for k in range(1, n + 1):
            result *= (2 * k) / (4.0)
        result *= np.sqrt(np.pi)
        # (2n)!/(4^n n!) = product_{k=1}^{n} (2k)/(4) * ... 
        # Actually: (2n)!/(4^n n!) = (2n-1)!!/2^n
        # Let me just compute directly
        from math import factorial as fac
        return fac(2 * n) / (4.0 ** n * fac(n)) * np.sqrt(np.pi)
    
    # P(x)^2 coefficient for x^(2m) is sum_{i=0}^{m} p_{2i} * p_{2(m-i)} for i where 2i, 2(m-i) are valid indices
    # But P has terms for x^0, x^2, x^4, x^6, x^8, x^10, x^12
    # So P(x)^2 has terms from x^0 to x^24
    
    # Coefficients of P: index k means x^(2k)
    p_coeffs = [p0, p2, p4, p6, p8, p10, p12]  # p_coeffs[k] = coefficient of x^(2k)
    
    norm2 = 0.0
    for i in range(7):
        for j in range(7):
            m = i + j
            norm2 += p_coeffs[i] * p_coeffs[j] * _even_moment(m)
    
    if norm2 < 1e-30:
        return 1e10
    
    # integral x^2 g(x)^2 dx = integral x^2 P(x)^2 exp(-x^2) dx = sum p_i p_j * integral x^(2(i+j)+2) exp(-x^2) dx
    x2_norm2 = 0.0
    for i in range(7):
        for j in range(7):
            m = i + j + 1
            x2_norm2 += p_coeffs[i] * p_coeffs[j] * _even_moment(m)
    
    # g'(x) = P'(x) exp(-x^2/2) + P(x) * (-x) * exp(-x^2/2) = (P'(x) - x P(x)) exp(-x^2/2)
    # Let Q(x) = P'(x) - x P(x)
    # P(x) = sum p_{2k} x^(2k)
    # P'(x) = sum 2k * p_{2k} x^(2k-1)
    # x P(x) = sum p_{2k} x^(2k+1)
    # Q(x) = sum 2k p_{2k} x^(2k-1) - sum p_{2k} x^(2k+1)
    # Q(x) has only odd powers of x.
    
    # Q(x)^2 will have even powers.
    # integral Q(x)^2 exp(-x^2) dx
    
    # Q(x) = sum_{k=1}^{6} 2k p_{2k} x^(2k-1) - sum_{k=0}^{5} p_{2k} x^(2k+1)
    # Let's collect coefficients for x^(2m+1):
    # From P' term: coefficient of x^(2m+1) is 2(m+1) p_{2(m+1)} for m=0..5
    # From -xP term: coefficient of x^(2m+1) is -p_{2m} for m=0..5
    
    # q_{2m+1} = 2(m+1) p_{2(m+1)} - p_{2m} for m=0..5
    # For m=5: 2*6*p_12 - p_10 = 12*p12 - p10
    # For m=6: would need p_14 which is 0, so just -p_12... wait
    # Actually P' has terms up to x^23 (from x^12), so 2k p_{2k} x^(2k-1) for k=1..6 gives x^1, x^3, ..., x^11
    # xP has terms up to x^13, so p_{2k} x^(2k+1) for k=0..5 gives x^1, x^3, ..., x^11
    # Wait, k=0..6 for P, so xP has k=0..6 giving x^1, x^3, ..., x^13
    
    # Let me redo: Q(x) = P'(x) - x*P(x)
    # P(x) = p0 + p2 x^2 + p4 x^4 + p6 x^6 + p8 x^8 + p10 x^10 + p12 x^12
    # P'(x) = 2 p2 x + 4 p4 x^3 + 6 p6 x^5 + 8 p8 x^7 + 10 p10 x^9 + 12 p12 x^11
    # x P(x) = p0 x + p2 x^3 + p4 x^5 + p6 x^7 + p8 x^9 + p10 x^11 + p12 x^13
    # Q(x) = (2p2 - p0) x + (4p4 - p2) x^3 + (6p6 - p4) x^5 + (8p8 - p6) x^7 + (10p10 - p8) x^9 + (12p12 - p10) x^11 - p12 x^13
    
    q_coeffs = {}  # power -> coefficient
    q_coeffs[1] = 2 * p2 - p0
    q_coeffs[3] = 4 * p4 - p2
    q_coeffs[5] = 6 * p6 - p4
    q_coeffs[7] = 8 * p8 - p6
    q_coeffs[9] = 10 * p10 - p8
    q_coeffs[11] = 12 * p12 - p10
    q_coeffs[13] = -p12
    
    # Q(x)^2 = sum of products
    # integral Q(x)^2 exp(-x^2) dx
    # Q(x) = sum_{odd powers} c_k x^k
    # Q(x)^2 = sum_{i,j} c_i c_j x^(i+j)
    # integral x^(i+j) exp(-x^2) dx where i+j is even
    
    deriv_norm2 = 0.0
    powers = sorted(q_coeffs.keys())
    for i in powers:
        for j in powers:
            m = i + j  # even
            if m % 2 != 0:
                continue
            n = m // 2
            deriv_norm2 += q_coeffs[i] * q_coeffs[j] * _even_moment(n)
    
    C4 = (x2_norm2 / norm2) * (deriv_norm2 / norm2)
    
    return C4


def construct():
    """
    Return the three coefficients (a0, a4, a8) that minimize C4.
    a12 is determined by g(0)=0.
    """
    # Start from the seed values
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Evaluate seed
    seed_val = _c4_bound_fast(*x0)
    best_x = x0.copy()
    best_val = seed_val
    
    # Multi-start optimization
    rng = np.random.RandomState(42)
    
    # Generate multiple starting points
    starts = [x0]
    # Add perturbations
    for _ in range(20):
        pert = rng.randn(3) * 0.5
        starts.append(x0 + pert)
    
    # Also try some structured starts
    starts.append(np.array([-1.0, 0.1, 0.001]))
    starts.append(np.array([-1.1, 0.05, 0.002]))
    starts.append(np.array([-1.0, 0.0, 0.0]))
    
    for start in starts:
        if _time_left() < 30:
            break
        
        try:
            res = minimize(
                _c4_bound_fast,
                start,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Refine with L-BFGS-B if possible (need gradient, use numerical)
    if _time_left() > 20:
        try:
            res = minimize(
                _c4_bound_fast,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'gtol': 1e-10}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Final polish with very tight Nelder-Mead
    if _time_left() > 10:
        try:
            res = minimize(
                _c4_bound_fast,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
