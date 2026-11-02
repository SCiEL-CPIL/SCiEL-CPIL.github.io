"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import factorial


def _phys_hermite_coeffs(n):
    """Return coefficients of the physicists' Hermite polynomial H_n(x).
    H_n(x) = sum_k c[k] x^k where k has same parity as n.
    Returns a list of length n+1 with c[k] = 0 for parity mismatch.
    """
    coeffs = [0.0] * (n + 1)
    if n == 0:
        coeffs[0] = 1.0
        return coeffs
    # Recurrence: H_{n+1} = 2x H_n - 2n H_{n-1}
    H_prev = [0.0] * (n + 1)  # H_0
    H_prev[0] = 1.0
    if n >= 1:
        H_curr = [0.0] * (n + 1)  # H_1 = 2x
        H_curr[1] = 2.0
    for m in range(1, n):
        H_next = [0.0] * (n + 1)
        # 2x * H_curr
        for k in range(n - 1, -1, -1):
            H_next[k + 1] += 2.0 * H_curr[k]
        # -2m * H_prev
        for k in range(n + 1):
            H_next[k] -= 2.0 * m * H_prev[k]
        H_prev = H_curr
        H_curr = H_next
    return H_curr


def _integrate_g2_coeff(g0, g4, g8, g12):
    """
    Compute the integral of g(x)^2 dx where
    g(x) = g0*H0(x) + g4*H4(x) + g8*H8(x) + g12*H12(x)
    and H_n are physicists' Hermite polynomials, with the Gaussian weight
    absorbed (i.e., we compute ∫ g(x)^2 e^{-x^2} dx / ∫ e^{-x^2} dx? 
    Actually for the uncertainty product we need specific moments.
    
    Let me reconsider. The task is about the uncertainty inequality A(f)*A(F(f)).
    For a function f(x) = g(x) * exp(-x^2/2) or similar Gaussian-weighted Hermite combo,
    the uncertainty product involves integrals of |f|^2 and |F(f)|^2.
    
    Actually, let's think about this more carefully. The standard setup for the
    uncertainty inequality with Hermite-Gaussian test functions:
    
    f(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    
    Then A(f) = (∫ x^2 |f|^2 dx / ∫ |f|^2 dx)^(1/2) or similar, and A(F(f)) similarly.
    
    The Fourier transform of H_n(x) * exp(-x^2/2) is related to H_n(x) * exp(-x^2/2)
    up to a phase factor (for the standard normalization).
    
    Actually, the Fourier transform of H_n(x) e^{-x^2/2} is (-i)^n H_n(x) e^{-x^2/2}
    (with appropriate normalization). So |F(f)|^2 has the same form as |f|^2 in terms
    of the Hermite expansion, just with phases that don't affect the magnitude squared.
    
    Wait, that's only true if we use the right normalization. Let me be more careful.
    
    With the convention F[f](ξ) = ∫ f(x) e^{-2πixξ} dx, the Hermite functions
    ψ_n(x) = H_n(x) e^{-x^2/2} / (2^n n! √π)^{1/2} are eigenfunctions of the
    Fourier transform with eigenvalue (-i)^n.
    
    So if f(x) = Σ a_n ψ_n(x), then F(f)(ξ) = Σ a_n (-i)^n ψ_n(ξ), and
    |F(f)|^2 has the same coefficient structure as |f|^2.
    
    Therefore A(f) = A(F(f)) and the product is A(f)^2.
    
    So we need to minimize A(f)^2 = (∫ x^2 |f|^2 dx / ∫ |f|^2 dx).
    
    Wait, but the task says "upper bound on A(f)*A(Fourier(f))" and "score is the upper
    bound on C4". Let me re-read.
    
    "Tune a Fourier-invariant Hermite-polynomial times Gaussian test function to reduce
    the upper bound on A(f)*A(Fourier(f))."
    
    "score is the upper bound on C4"
    
    Hmm, C4 might refer to a specific constant. Let me think about what the actual
    computation is.
    
    Given the structure (Hermite polynomials of degrees 0,4,8,12 with g(0)=0 constraint),
    and that the Fourier transform preserves the Hermite structure, I believe:
    
    g(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    f(x) = g(x) * exp(-x^2/2)
    
    The constraint g(0) = 0 determines c12 in terms of c0, c4, c8.
    
    H0(0) = 1
    H4(0) = -12  (H4(x) = 16x^4 - 48x^2 + 12, so H4(0) = 12... wait let me check)
    
    Actually: H0=1, H1=2x, H2=4x^2-2, H3=8x^3-12x, H4=16x^4-48x^2+12
    So H4(0) = 12, H8(0) = ?
    
    H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    H8(0) = 1680
    
    H12(0) = ?
    H12(x) has constant term. Let me compute: the constant term of H_{2n}(x) is (-1)^n (2n)!/n!
    H12: n=6, constant = (-1)^6 * 12!/6! = 479001600/720 = 665280
    
    So g(0) = c0*1 + c4*12 + c8*1680 + c12*665280 = 0
    => c12 = -(c0 + 12*c4 + 1680*c8) / 665280
    
    Now, A(f)^2 = ∫ x^2 |f|^2 dx / ∫ |f|^2 dx
    
    |f|^2 = g(x)^2 * e^{-x^2}
    
    ∫ |f|^2 dx = ∫ g(x)^2 e^{-x^2} dx
    ∫ x^2 |f|^2 dx = ∫ x^2 g(x)^2 e^{-x^2} dx
    
    Since g is a polynomial, g^2 is a polynomial, and we can compute these integrals
    exactly using the Gaussian integral formula: ∫ x^k e^{-x^2} dx = Γ((k+1)/2)/2 for even k.
    
    Let me code this up and then optimize c0, c4, c8 to minimize A(f)^2.
    """
    pass


def _compute_objective(c0, c4, c8):
    """Compute A(f)^2 = ∫x²g²e^{-x²}dx / ∫g²e^{-x²}dx with g(0)=0 constraint."""
    # Get Hermite polynomial coefficients
    H0 = _phys_hermite_coeffs(0)
    H4 = _phys_hermite_coeffs(4)
    H8 = _phys_hermite_coeffs(8)
    H12 = _phys_hermite_coeffs(12)
    
    # Build g(x) coefficients: g(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    max_deg = 12
    g = [0.0] * (max_deg + 1)
    for k in range(max_deg + 1):
        g[k] = c0 * H0[k] + c4 * H4[k] + c8 * H8[k]
    
    # Compute c12 from g(0) = 0
    # g(0) = g[0] + c12 * H12[0] = 0 => c12 = -g[0] / H12[0]
    if abs(H12[0]) < 1e-15:
        return np.inf
    c12 = -g[0] / H12[0]
    for k in range(max_deg + 1):
        g[k] += c12 * H12[k]
    
    # Compute g^2 coefficients
    g2 = [0.0] * (2 * max_deg + 1)
    for i in range(max_deg + 1):
        if abs(g[i]) < 1e-15:
            continue
        for j in range(max_deg + 1):
            if abs(g[j]) < 1e-15:
                continue
            g2[i + j] += g[i] * g[j]
    
    # Compute ∫ g^2 e^{-x^2} dx and ∫ x^2 g^2 e^{-x^2} dx
    # ∫ x^k e^{-x^2} dx = Γ((k+1)/2) / 2 for even k, 0 for odd k
    from math import gamma, sqrt, pi
    
    integral_0 = 0.0  # ∫ g^2 e^{-x^2} dx
    integral_2 = 0.0  # ∫ x^2 g^2 e^{-x^2} dx
    
    for k in range(len(g2)):
        if abs(g2[k]) < 1e-15:
            continue
        if k % 2 != 0:
            continue  # Odd powers integrate to 0
        # ∫ x^k e^{-x^2} dx = Γ((k+1)/2) / 2
        val = gamma(k // 2 + 0.5) / 2.0
        integral_0 += g2[k] * val
        # For x^2 * x^k = x^{k+2}
        val2 = gamma((k + 2) // 2 + 0.5) / 2.0
        integral_2 += g2[k] * val2
    
    if abs(integral_0) < 1e-15:
        return np.inf
    
    return integral_2 / integral_0


def construct():
    """Return three coefficients (c0, c4, c8) minimizing the uncertainty product."""
    deadline = time.time() + 180.0
    safety_margin = 15.0  # seconds
    
    # Start with the seed values
    best = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _compute_objective(best[0], best[1], best[2])
    
    # Use Nelder-Mead optimization
    from scipy.optimize import minimize
    
    def objective(x):
        return _compute_objective(x[0], x[1], x[2])
    
    # Try Nelder-Mead
    try:
        result = minimize(objective, best, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
        if result.fun < best_val:
            best_val = result.fun
            best = result.x
    except Exception:
        pass
    
    if time.time() > deadline - safety_margin:
        return best
    
    # Try L-BFGS-B with finite differences
    try:
        result2 = minimize(objective, best, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-15})
        if result2.fun < best_val:
            best_val = result2.fun
            best = result2.x
    except Exception:
        pass
    
    if time.time() > deadline - safety_margin:
        return best
    
    # Try a few random restarts
    rng = np.random.default_rng(42)
    for i in range(50):
        if time.time() > deadline - safety_margin:
            break
        start = rng.normal(0, 1, size=3)
        try:
            result3 = minimize(objective, start, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
            if result3.fun < best_val:
                best_val = result3.fun
                best = result3.x
        except Exception:
            pass
    
    return np.array(best)
# EVOLVE-BLOCK-END
