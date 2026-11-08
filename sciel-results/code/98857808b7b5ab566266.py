import numpy as np
from scipy.optimize import minimize
import time

# Analytical computation of integrals for Hermite-Gaussian functions
# g(x) = sum_k c_k * H_k(x) * exp(-x^2/2)
# where H_k are physicists' Hermite polynomials

def _hermite_coeffs(n):
    """Return coefficients of H_n(x) (physicists') as a list [c0, c1, ..., cn] where H_n = sum c_i x^i."""
    # H_0 = 1, H_1 = 2x, H_{n+1} = 2x*H_n - 2n*H_{n-1}
    H = [[1.0], [0.0, 2.0]]
    for k in range(2, n + 1):
        # H_k = 2x * H_{k-1} - 2(k-1) * H_{k-2}
        prev = H[k-1]
        prev2 = H[k-2]
        new = [0.0] * (k + 1)
        # 2x * H_{k-1}: shift prev by 1, multiply by 2
        for i in range(len(prev)):
            new[i+1] += 2.0 * prev[i]
        # -2(k-1) * H_{k-2}
        for i in range(len(prev2)):
            new[i] -= 2.0 * (k-1) * prev2[i]
        H.append(new)
    return H[n]

def _poly_product(a, b):
    """Multiply two polynomials given as coefficient lists (ascending order)."""
    result = [0.0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i+j] += ai * bj
    return result

def _poly_power(poly, n):
    """Raise polynomial to power n (n small, 2 or 4)."""
    result = [1.0]
    for _ in range(n):
        result = _poly_product(result, poly)
    return result

def _integrate_poly_exp_neg_x2_over_2(poly):
    """Compute integral from -inf to inf of P(x) * exp(-x^2/2) dx.
    Only even powers contribute. Use: int x^{2m} exp(-x^2/2) dx = (2m-1)!! * sqrt(2*pi)
    """
    total = 0.0
    for i, coeff in enumerate(poly):
        if i % 2 == 0:
            m = i // 2
            # (2m-1)!! = product of odd numbers from 1 to 2m-1; for m=0, (-1)!! = 1
            if m == 0:
                double_factorial = 1.0
            else:
                double_factorial = 1.0
                for k in range(1, 2*m, 2):
                    double_factorial *= k
            total += coeff * double_factorial * np.sqrt(2.0 * np.pi)
    return total

def _compute_moments(c0, c4, c8):
    """Compute A(f)^2 and A(FFT(f))^2 analytically.
    Returns (A_f_sq, A_fft_sq) where A_f = (int |g|^4)^(1/2) / (int |g|^2)
    so A_f^2 = int|g|^4 / (int|g|^2)^2
    """
    c12 = -(c0 + c4 + c8)
    coeffs = [c0, c4, c8, c12]
    degrees = [0, 4, 8, 12]
    
    # Build g(x)/exp(-x^2/2) as polynomial P(x) = sum c_k H_k(x)
    P = [0.0] * 13  # max degree 12
    for c, d in zip(coeffs, degrees):
        hc = _hermite_coeffs(d)
        for i, hi in enumerate(hc):
            P[i] += c * hi
    
    # |g|^2 = P(x)^2 * exp(-x^2)
    # int |g|^2 dx = int P(x)^2 * exp(-x^2) dx
    # Note: exp(-x^2/2)^2 = exp(-x^2)
    # int x^{2m} exp(-x^2) dx = (2m-1)!! / 2^m * sqrt(pi)
    
    P2 = _poly_product(P, P)
    
    int_g2 = 0.0
    for i, coeff in enumerate(P2):
        if i % 2 == 0:
            m = i // 2
            if m == 0:
                df = 1.0
            else:
                df = 1.0
                for k in range(1, 2*m, 2):
                    df *= k
            int_g2 += coeff * df / (2.0**m) * np.sqrt(np.pi)
    
    # |g|^4 = P(x)^4 * exp(-2x^2)
    P4 = _poly_power(P, 4)
    
    int_g4 = 0.0
    for i, coeff in enumerate(P4):
        if i % 2 == 0:
            m = i // 2
            # int x^{2m} exp(-2x^2) dx
            # Let u = sqrt(2)x, dx = du/sqrt(2), x^{2m} = u^{2m}/2^m
            # int u^{2m} exp(-u^2) du / sqrt(2) / 2^m = (2m-1)!! / 2^m * sqrt(pi) / (2^m * sqrt(2))
            # = (2m-1)!! * sqrt(pi) / (2^{2m} * sqrt(2))
            if m == 0:
                df = 1.0
            else:
                df = 1.0
                for k in range(1, 2*m, 2):
                    df *= k
            int_g4 += coeff * df * np.sqrt(np.pi) / (2.0**(2*m) * np.sqrt(2.0))
    
    A_f_sq = int_g4 / (int_g2 ** 2) if int_g2 != 0 else np.inf
    
    # For the Fourier transform: 
    # With convention F[f](xi) = int f(x) exp(-i x xi) dx
    # For g(x) = P(x) exp(-x^2/2), 
    # The Fourier transform of H_n(x) exp(-x^2/2) is (-i)^n H_n(xi) exp(-xi^2/2)
    # So FFT(g)(xi) = sum_k c_k (-i)^k H_k(xi) exp(-xi^2/2)
    # 
    # |FFT(g)|^2 involves cross terms with (-i)^k (-i)^{-l} = (-i)^{k-l}
    # For the integral of |FFT(g)|^2 and |FFT(g)|^4, we need to handle the complex coefficients.
    #
    # Actually, let's use the fact that the Fourier transform of a real Hermite-Gaussian is 
    # another (possibly complex) Hermite-Gaussian. For our purpose, we need:
    # int |FFT(g)|^2 dxi = int |g|^2 dx (Plancherel, up to 2pi factor depending on convention)
    # 
    # With F[f](xi) = (1/sqrt(2pi)) int f(x) exp(-ixxi) dx, Plancherel gives exact equality.
    # Let's use this convention.
    #
    # FFT(g)(xi) = sum_k c_k (-i)^k H_k(xi) exp(-xi^2/2) / sqrt(2pi) ... 
    # Actually, the Fourier transform of H_n(x)e^{-x^2/2} with F[f] = (1/sqrt(2pi))∫f(x)e^{-ixxi}dx
    # is (-i)^n H_n(xi) e^{-xi^2/2}.
    #
    # So FFT(g)(xi) = (1/sqrt(2pi)) * sum_k c_k (-i)^k H_k(xi) exp(-xi^2/2)
    #
    # |FFT(g)|^2 = (1/(2pi)) * |sum_k c_k (-i)^k H_k(xi)|^2 * exp(-xi^2)
    #
    # For the L4 norm of FFT(g):
    # |FFT(g)|^4 = (1/(2pi)^2) * |sum_k c_k (-i)^k H_k(xi)|^4 * exp(-2xi^2)
    #
    # This is complex because of the (-i)^k factors. We need to compute:
    # Q(xi) = sum_k c_k (-i)^k H_k(xi)
    # |Q(xi)|^2 = Q(xi) * conj(Q(xi)) = sum_{k,l} c_k c_l (-i)^k (i)^l H_k(xi) H_l(xi)
    #
    # This is getting complex. Let me compute it numerically but efficiently, or find a better approach.
    #
    # Alternative: use the fact that for the uncertainty product, we can compute both A(f) and A(FFT(f))
    # using the same polynomial structure but with complex coefficients for the FFT side.
    
    # Let's compute the FFT-side polynomial with complex coefficients
    # Q(xi) = sum_k c_k * (-1j)^k * H_k(xi)
    Q = [0.0 + 0.0j] * 13
    for c, d in zip(coeffs, degrees):
        hc = _hermite_coeffs(d)
        factor = (-1j) ** d
        for i, hi in enumerate(hc):
            Q[i] += c * factor * hi
    
    # |Q|^2 = Q * conj(Q) - this is a polynomial in xi with real coefficients (since Q*conj(Q) is real)
    # But computing this analytically for degree-12 polynomials is complex.
    # Let's use a numerical approach that's still fast: evaluate on a fine grid once and cache,
    # or use the fact that we can compute the integrals via the same moment method but with the
    # real polynomial |Q(xi)|^2.
    
    # Actually, |Q(xi)|^2 is a real polynomial of degree 24. Let's construct it.
    # Q(xi) = sum a_k xi^k where a_k are complex
    # |Q|^2 = (sum a_k xi^k)(sum conj(a_l) xi^l) = sum_{k,l} a_k conj(a_l) xi^{k+l}
    # The coefficient of xi^m is sum_{k+l=m} a_k conj(a_l), which is real.
    
    Q_coeffs = Q  # complex coefficients
    absQ2 = [0.0] * 25  # degree up to 24
    for k in range(13):
        for l in range(13):
            if k + l <= 24:
                val = Q_coeffs[k] * np.conj(Q_coeffs[l])
                absQ2[k+l] += val.real  # should be real
    
    # int |FFT(g)|^2 dxi = (1/(2pi)) * int |Q(xi)|^2 exp(-xi^2) dxi
    int_fft2 = 0.0
    for i, coeff in enumerate(absQ2):
        if i % 2 == 0:
            m = i // 2
            if m == 0:
                df = 1.0
            else:
                df = 1.0
                for k2 in range(1, 2*m, 2):
                    df *= k2
            int_fft2 += coeff * df / (2.0**m) * np.sqrt(np.pi)
    int_fft2 /= (2.0 * np.pi)
    
    # |FFT(g)|^4 = (1/(2pi)^2) * |Q(xi)|^4 * exp(-2xi^2)
    # |Q|^4 = (|Q|^2)^2
    absQ4 = _poly_product(absQ2, absQ2)
    
    int_fft4 = 0.0
    for i, coeff in enumerate(absQ4):
        if i % 2 == 0:
            m = i // 2
            if m == 0:
                df = 1.0
            else:
                df = 1.0
                for k2 in range(1, 2*m, 2):
                    df *= k2
            int_fft4 += coeff * df * np.sqrt(np.pi) / (2.0**(2*m) * np.sqrt(2.0))
    int_fft4 /= (2.0 * np.pi) ** 2
    
    A_fft_sq = int_fft4 / (int_fft2 ** 2) if int_fft2 != 0 else np.inf
    
    return A_f_sq, A_fft_sq

def _objective(x):
    c0, c4, c8 = x
    A_f_sq, A_fft_sq = _compute_moments(c0, c4, c8)
    if not np.isfinite(A_f_sq) or not np.isfinite(A_fft_sq):
        return 1e10
    # C4 = A(f) * A(FFT(f)) = sqrt(A_f_sq) * sqrt(A_fft_sq) = sqrt(A_f_sq * A_fft_sq)
    return np.sqrt(A_f_sq * A_fft_sq)

def construct():
    t_start = time.time()
    deadline = t_start + 180.0  # safety margin before 210s kill
    
    # Initial guess from prior art
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    best_x = x0.copy()
    best_val = _objective(x0)
    
    # Multi-start Nelder-Mead
    rng = np.random.default_rng(42)
    n_starts = 50
    
    for i in range(n_starts):
        if time.time() > deadline - 30.0:
            break
        
        if i == 0:
            x_init = x0.copy()
        else:
            x_init = x0 + rng.normal(0, 0.5, size=3)
        
        try:
            res = minimize(
                _objective, x_init,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 10.0:
        try:
            res = minimize(
                _objective, best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-14,
                    'fatol': 1e-16,
                    'adaptive': True
                }
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_x)
