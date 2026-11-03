"""Tune Hermite coefficients to minimize the upper bound on C4."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import factorial
from scipy.optimize import differential_evolution, minimize

# Deadline guard: evaluator kills at 210s; use 195s as safety margin
_DEADLINE = time.time() + 195


def _time_left():
    return _DEADLINE - time.time()


def _hermite(n, x):
    """Physicists' Hermite polynomial H_n(x) via recurrence."""
    if n == 0:
        return np.ones_like(x, dtype=float)
    if n == 1:
        return 2 * x
    h0 = np.ones_like(x, dtype=float)
    h1 = 2 * x
    for k in range(2, n + 1):
        h2 = 2 * x * h1 - 2 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _hermite_moments(n, m):
    """
    Compute integral of H_n(x) H_m(x) e^{-x^2} dx over R.
    = 2^{(n+m)/2} * pi^{1/2} * n! / ((n-m)!!) if n>=m, m even... 
    Actually: ∫ H_n H_m e^{-x^2} dx = 2^n n! sqrt(pi) δ_{nm}
    Wait, that's for physicists' Hermite with weight e^{-x^2}.
    ∫ H_n(x) H_m(x) e^{-x^2} dx = 2^n n! sqrt(pi) if n=m, else 0.
    """
    if n == m:
        return 2.0 ** n * float(factorial(n, exact=True)) * np.sqrt(np.pi)
    return 0.0


def _compute_bound(coeffs):
    """
    Given three free coefficients [c0, c4, c8] for H0, H4, H8,
    derive c12 so that g(0) = 0, then compute the upper bound on C4.
    
    The test function is g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    (or similar Gaussian weight; need to match the notebook's convention)
    
    For the uncertainty inequality / Fourier-invariant setting:
    g(x) = P(Hermite) * Gaussian, and the bound involves ||g||_2 and ||g'||_2
    or similar norms that determine A(f) and A(Fourier(f)).
    
    Let's use the standard setup:
    g(x) = [c0 H0(x) + c4 H4(x) + c8 H8(x) + c12 H12(x)] * exp(-x^2/2)
    
    P(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395
    So: c0 - 3*c4 + 105*c8 - 10395*c12 = 0
    => c12 = (c0 - 3*c4 + 105*c8) / 10395
    
    The upper bound on C4 involves the ratio of norms.
    For a function g, A(g) * A(F(g)) is bounded by something involving
    ||g||_2, ||g'||_2, etc.
    
    In the standard formulation for the Fourier uncertainty principle:
    The sharp constant C4 satisfies: for all f, 
    (∫ x^2 |f|^2)(∫ ξ^2 |F(f)|^2) ≥ C4 * (∫|f|^2)^2
    
    For our test function g, we compute:
    ||g||_2^2 = ∫ g^2 dx
    ||x g||_2^2 = ∫ x^2 g^2 dx  
    ||g'||_2^2 = ∫ |g'|^2 dx  (related to ||ξ F(g)||_2^2 via Plancherel)
    
    The bound for this specific g gives an upper bound on C4.
    """
    c0, c4, c8 = coeffs
    # Derive c12 from P(0) = 0
    # H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395
    c12 = (c0 - 3.0 * c4 + 105.0 * c8) / 10395.0
    
    # Build the polynomial coefficients for the Hermite combination
    # We need to compute integrals of powers of x times the squared Hermite combo
    # g(x) = P(x) * exp(-x^2/2) where P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    
    # For efficiency, precompute the Hermite polynomials as coefficient arrays
    # H0(x) = 1
    # H1(x) = 2x
    # H2(x) = 4x^2 - 2
    # H3(x) = 8x^3 - 12x
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 4480x^6 + 20160x^4 - 20160x^2 + 3024
    # H12(x) = 4096x^12 - 129024x^10 + 1881600x^8 - 10982400x^6 + 26409600x^4 - 25401600x^2 + 5189184
    
    # Let me verify these by recurrence
    # H0 = 1
    # H1 = 2x
    # H2 = 2x*H1 - 2*1*H0 = 4x^2 - 2
    # H3 = 2x*H2 - 4*H1 = 8x^3 - 4x - 8x = 8x^3 - 12x
    # H4 = 2x*H3 - 6*H2 = 16x^4 - 24x^2 - 24x^2 + 12 = 16x^4 - 48x^2 + 12
    # Good.
    
    # For the integrals, we need ∫ x^k * P(x)^2 * exp(-x^2) dx for k=0,2
    # (since g = P * exp(-x^2/2), g^2 = P^2 * exp(-x^2))
    
    # P(x)^2 will be a polynomial of degree 24. We need coefficients.
    # This is expensive to do symbolically each time, so let's use a grid-based
    # numerical integration approach, or better, use the fact that we can
    # expand P(x) in monomials and compute the integrals analytically.
    
    # Let's represent each Hermite as a monomial coefficient array.
    def _hermite_coeffs(n):
        """Return array of coefficients for H_n(x) as monomial (index = power)."""
        if n == 0:
            return np.array([1.0])
        if n == 1:
            return np.array([0.0, 2.0])
        hc = np.zeros(n + 1)
        hc[0] = 1.0
        for k in range(2, n + 1):
            hc = np.zeros(k + 1)
            # H_k = 2x * H_{k-1} - 2(k-1) H_{k-2}
            # 2x * H_{k-1}: shift coefficients up by 1, multiply by 2
            prev1 = np.zeros(k)  # H_{k-1} has degree k-1
            prev2 = np.zeros(k-1)  # H_{k-2} has degree k-2
            # Rebuild from scratch is cleaner
            pass
        # Let me just hardcode up to degree 12
        if n == 0:
            return np.array([1.0])
        elif n == 1:
            return np.array([0.0, 2.0])
        elif n == 2:
            return np.array([-2.0, 0.0, 4.0])
        elif n == 3:
            return np.array([0.0, -12.0, 0.0, 8.0])
        elif n == 4:
            return np.array([12.0, 0.0, -48.0, 0.0, 16.0])
        elif n == 5:
            return np.array([0.0, -120.0, 0.0, 160.0, 0.0, 32.0])
        elif n == 6:
            return np.array([120.0, 0.0, -720.0, 0.0, 480.0, 0.0, 64.0])
        elif n == 7:
            return np.array([0.0, -1680.0, 0.0, 3360.0, 0.0, -560.0, 0.0, 128.0])
        elif n == 8:
            return np.array([3024.0, 0.0, -20160.0, 0.0, 20160.0, 0.0, -4480.0, 0.0, 256.0])
        elif n == 9:
            return np.array([0.0, -40320.0, 0.0, 181440.0, 0.0, -120960.0, 0.0, 26880.0, 0.0, 512.0])
        elif n == 10:
            return np.array([103950.0, 0.0, -1058400.0, 0.0, 4084000.0, 0.0, -5443200.0, 0.0, 3174400.0, 0.0, -230400.0, 0.0, 1024.0])
        elif n == 11:
            return np.array([0.0, -2126112.0, 0.0, 27707520.0, 0.0, -121545600.0, 0.0, 223385600.0, 0.0, -192256000.0, 0.0, 66585600.0, 0.0, -1126400.0, 0.0, 2048.0])
        elif n == 12:
            return np.array([5189184.0, 0.0, -25401600.0, 0.0, 26409600.0, 0.0, -10982400.0, 0.0, 1881600.0, 0.0, -129024.0, 0.0, 4096.0])
        else:
            raise ValueError("n too large")
    
    h0_c = _hermite_coeffs(0)
    h4_c = _hermite_coeffs(4)
    h8_c = _hermite_coeffs(8)
    h12_c = _hermite_coeffs(12)
    
    # P(x) = c0*h0 + c4*h4 + c8*h8 + c12*h12
    # Max degree is 12
    p_coeffs = np.zeros(13)
    p_coeffs += c0 * h0_c
    p_coeffs += c4 * h4_c
    p_coeffs += c8 * h8_c
    p_coeffs += c12 * h12_c
    
    # P(x)^2: degree 24
    p2_coeffs = np.convolve(p_coeffs, p_coeffs)
    
    # ∫ x^k * P(x)^2 * exp(-x^2) dx
    # = sum_{j} p2_coeffs[j] * ∫ x^{j+k} * exp(-x^2) dx
    # ∫ x^m * exp(-x^2) dx over R = 0 if m odd, 
    #   = 2^{(m-1)/2} * Gamma((m+1)/2) * sqrt(pi)/2 * 2 ... 
    # Actually: ∫_{-∞}^{∞} x^{2n} e^{-x^2} dx = (2n-1)!! * sqrt(pi) / 2^n = (2n)! * sqrt(pi) / (4^n * n!)
    
    def _even_moment(m):
        """∫ x^m e^{-x^2} dx over R, for m even. 0 if m odd."""
        if m % 2 != 0:
            return 0.0
        n = m // 2
        # (2n)! sqrt(pi) / (4^n n!)
        return float(factorial(2*n, exact=True)) * np.sqrt(np.pi) / (4.0**n * float(factorial(n, exact=True)))
    
    # ||g||_2^2 = ∫ P^2 e^{-x^2} dx = sum_j p2_coeffs[j] * _even_moment(j)
    norm2_sq = 0.0
    for j in range(len(p2_coeffs)):
        norm2_sq += p2_coeffs[j] * _even_moment(j)
    
    # ||x g||_2^2 = ∫ x^2 P^2 e^{-x^2} dx = sum_j p2_coeffs[j] * _even_moment(j+2)
    x_norm2_sq = 0.0
    for j in range(len(p2_coeffs)):
        x_norm2_sq += p2_coeffs[j] * _even_moment(j + 2)
    
    # For the Fourier side, we need ||g'||_2^2 or equivalently ||ξ F(g)||_2^2
    # g(x) = P(x) e^{-x^2/2}
    # g'(x) = [P'(x) - x P(x)] e^{-x^2/2}
    # ||g'||_2^2 = ∫ [P'(x) - x P(x)]^2 e^{-x^2} dx
    # = ∫ [P'(x)^2 - 2x P(x)P'(x) + x^2 P(x)^2] e^{-x^2} dx
    
    # P'(x): derivative of P
    p_deriv_coeffs = np.zeros(12)
    for j in range(1, 13):
        p_deriv_coeffs[j-1] = j * p_coeffs[j]
    
    # P'(x)^2: degree 22
    p2_deriv_coeffs = np.convolve(p_deriv_coeffs, p_deriv_coeffs)
    
    # ∫ P'(x)^2 e^{-x^2} dx
    pp2_sq = 0.0
    for j in range(len(p2_deriv_coeffs)):
        pp2_sq += p2_deriv_coeffs[j] * _even_moment(j)
    
    # ∫ x P(x) P'(x) e^{-x^2} dx: this is ∫ x * (1/2) d/dx[P(x)^2] e^{-x^2} dx
    # = (1/2) ∫ x d(P^2) e^{-x^2} dx
    # Integration by parts: = (1/2)[P^2 e^{-x^2} x]_{-∞}^{∞} - (1/2)∫ P^2 d/dx(x e^{-x^2}) dx
    # = 0 - (1/2) ∫ P^2 (e^{-x^2} - 2x^2 e^{-x^2}) dx
    # = -(1/2) ∫ P^2 e^{-x^2} dx + ∫ x^2 P^2 e^{-x^2} dx
    # = -(1/2) norm2_sq + x_norm2_sq
    
    cross_term = -(0.5) * norm2_sq + x_norm2_sq
    
    # ||g'||_2^2 = pp2_sq - 2*cross_term + x_norm2_sq
    # Wait: ∫ [P' - xP]^2 e^{-x^2} = ∫ P'^2 e^{-x^2} - 2∫ x P P' e^{-x^2} + ∫ x^2 P^2 e^{-x^2}
    # = pp2_sq - 2*cross_term + x_norm2_sq
    
    g_prime_sq = pp2_sq - 2.0 * cross_term + x_norm2_sq
    
    # The upper bound on C4:
    # C4 ≤ (||x g||_2^2 * ||g'||_2^2) / ||g||_2^4
    # This comes from the uncertainty principle applied to g.
    
    if norm2_sq <= 0:
        return 1e10
    
    bound = (x_norm2_sq * g_prime_sq) / (norm2_sq ** 2)
    
    return bound


def _objective(coeffs):
    """Objective function for optimization."""
    try:
        return _compute_bound(coeffs)
    except Exception:
        return 1e10


def construct():
    """
    Return three coefficients [c0, c4, c8] for the Hermite combination.
    The degree-12 weight is derived to impose g(0) = 0.
    """
    # Starting point from prior work
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_coeffs = x0.copy()
    best_bound = _objective(best_coeffs)
    
    if _time_left() < 30:
        return best_coeffs
    
    # Strategy 1: Differential evolution for global exploration
    try:
        bounds = [(-10, 10), (-10, 10), (-10, 10)]
        de_result = differential_evolution(
            _objective,
            bounds=bounds,
            seed=42,
            maxiter=200,
            tol=1e-10,
            popsize=15,
            polish=True,
            callback=lambda x, convergence: None,
            init='sobol',
        )
        if de_result.success or de_result.fun < best_bound:
            best_coeffs = de_result.x.copy()
            best_bound = de_result.fun
    except Exception:
        pass
    
    if _time_left() < 20:
        return best_coeffs
    
    # Strategy 2: Local refinement from best point found
    try:
        local_result = minimize(
            _objective,
            best_coeffs,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True},
        )
        if local_result.fun < best_bound:
            best_coeffs = local_result.x.copy()
            best_bound = local_result.fun
    except Exception:
        pass
    
    if _time_left() < 10:
        return best_coeffs
    
    # Strategy 3: Multi-start local optimization
    np.random.seed(123)
    for i in range(20):
        if _time_left() < 10:
            break
        start = best_coeffs + np.random.randn(3) * 0.5
        try:
            res = minimize(
                _objective,
                start,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True},
            )
            if res.fun < best_bound:
                best_coeffs = res.x.copy()
                best_bound = res.fun
        except Exception:
            pass
    
    # Final high-precision polish
    if _time_left() > 5:
        try:
            final_res = minimize(
                _objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True},
            )
            if final_res.fun < best_bound:
                best_coeffs = final_res.x.copy()
                best_bound = final_res.fun
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
