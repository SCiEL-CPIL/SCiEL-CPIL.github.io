"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _I(n):
    """I(n) = ∫_{-∞}^{∞} x^(2n) exp(-x^2) dx = sqrt(pi) * (2n-1)!! / 2^n"""
    if n == 0:
        return np.sqrt(np.pi)
    result = 1.0
    for k in range(1, 2*n, 2):
        result *= k
    return np.sqrt(np.pi) * result / (2.0 ** n)


# Precompute I(0) through I(20) for speed
_I_CACHE = [float(_I(n)) for n in range(21)]


def _hermite_coeffs(c0, c4, c8, c12):
    """
    Build the polynomial P(x) = sum_{j=0}^{6} p_j x^{2j}
    from Hermite polynomial combination:
    P(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    
    Using physicists' Hermite polynomials:
    H0(x) = 1
    H4(x) = x^4 - 6x^2 + 3
    H8(x) = x^8 - 28x^6 + 420x^4 - 1680x^2 + 105
    H12(x) = x^12 - 60x^10 + 2100x^8 - 22680x^6 + 60480x^4 - 30240x^2 + 10395
    """
    p = np.zeros(7)
    # H0 contribution
    p[0] += c0
    # H4 contribution: x^4 - 6x^2 + 3
    p[0] += 3 * c4
    p[1] += -6 * c4
    p[2] += c4
    # H8 contribution: x^8 - 28x^6 + 420x^4 - 1680x^2 + 105
    p[0] += 105 * c8
    p[1] += -1680 * c8
    p[2] += 420 * c8
    p[3] += -28 * c8
    p[4] += c8
    # H12 contribution: x^12 - 60x^10 + 2100x^8 - 22680x^6 + 60480x^4 - 30240x^2 + 10395
    p[0] += 10395 * c12
    p[1] += -30240 * c12
    p[2] += 60480 * c12
    p[3] += -22680 * c12
    p[4] += 2100 * c12
    p[5] += -60 * c12
    p[6] += c12
    return p


def _compute_raw_value(c0, c4, c8):
    """
    Compute A(g)*A(F(g)) for g(x) = P(x)*exp(-x^2/2)
    where P is a degree-12 even polynomial with P(0)=0.
    
    A(g)^2 = <x^2>/<1> for g
    A(F(g))^2 = <k^2>/<1> for F(g)
    
    For g(x) = P(x) e^{-x^2/2}:
      <1> = ∫ P(x)^2 e^{-x^2} dx
      <x^2> = ∫ x^2 P(x)^2 e^{-x^2} dx
    
    For F(g)(k) = sqrt(2π) Q(k) e^{-k^2/2} where Q is related to P:
      The Fourier transform of x^n e^{-x^2/2} involves Hermite polynomials.
      Specifically, for even powers, F[x^{2m} e^{-x^2/2}](k) = sqrt(2π) * (2m-1)!! * H_{2m}(ik)/i^{2m} * e^{-k^2/2} / (2^m m!)
      ... this gets complex. Let's use the property that for the specific form,
      the uncertainty in k-space equals the uncertainty in x-space when P is a 
      polynomial in x^2 with specific structure.
    
    Actually, let's compute it directly using the polynomial coefficients.
    """
    # P(0) = 0 constraint determines c12
    # P(0) = c0*1 + c4*3 + c8*105 + c12*10395 = 0
    c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
    
    p = _hermite_coeffs(c0, c4, c8, c12)
    
    # Compute Q = P(x)^2 as polynomial in x^2
    # P(x) = sum_{i=0}^{6} p_i x^{2i}
    # P(x)^2 = sum_{i,j} p_i p_j x^{2(i+j)}
    q = np.zeros(13)
    for i in range(7):
        for j in range(7):
            if i + j <= 12:
                q[i+j] += p[i] * p[j]
    
    # norm_sq = <1> = ∫ P(x)^2 e^{-x^2} dx = sum_m q_m I(m)
    norm_sq = 0.0
    for m in range(13):
        norm_sq += q[m] * _I_CACHE[m]
    
    # x2_norm = <x^2> = ∫ x^2 P(x)^2 e^{-x^2} dx = sum_m q_m I(m+1)
    x2_norm = 0.0
    for m in range(13):
        x2_norm += q[m] * _I_CACHE[m+1]
    
    if norm_sq <= 0 or not np.isfinite(norm_sq):
        return 1e10
    
    # For the Fourier transform, we need <k^2> for F(g)
    # F(g)(k) = sqrt(2π) ∫ P(x) e^{-x^2/2} e^{-ikx} dx
    # For even P, this is real and even in k.
    # The key insight: if P(x) = sum a_n x^n, then F[P(x) e^{-x^2/2}](k)
    # can be computed using the fact that F[x^n e^{-x^2/2}](k) = (-i)^n d^n/dk^n [e^{-k^2/2}]
    # = (-i)^n H_n(k) e^{-k^2/2} / (2^{n/2} sqrt(π)) ... 
    
    # Actually, let's use a different approach. For the specific case where
    # P is a polynomial in x^2, the Fourier transform will also be a polynomial
    # in k^2 times e^{-k^2/2}. We can compute the coefficients.
    
    # F[x^{2m} e^{-x^2/2}](k) = sqrt(2π) * (-1)^m * (2m-1)!! / 2^m * H_{2m}(k) * e^{-k^2/2} / (2^m m!)
    # Wait, let me use the standard result:
    # F[x^n e^{-ax^2}](k) = (-i)^n / (2a)^{n/2} * d^n/dk^n [e^{-k^2/(4a)}] * sqrt(π/a)
    # For a = 1/2: F[x^n e^{-x^2/2}](k) = (-i)^n * 2^{n/2} * d^n/dk^n [e^{-k^2/2}] * sqrt(2π)
    # And d^n/dk^n [e^{-k^2/2}] = (-1)^n H_n(k) e^{-k^2/2}
    # So F[x^n e^{-x^2/2}](k) = (-i)^n * 2^{n/2} * (-1)^n H_n(k) e^{-k^2/2} * sqrt(2π)
    # = (-i)^n * (-1)^n * 2^{n/2} * sqrt(2π) * H_n(k) * e^{-k^2/2}
    # = (-i)^n * (-1)^n * sqrt(2π) * 2^{n/2} * H_n(k) * e^{-k^2/2}
    
    # For even n=2m: (-i)^{2m} * (-1)^{2m} = (-1)^m * 1 = (-1)^m
    # F[x^{2m} e^{-x^2/2}](k) = (-1)^m * sqrt(2π) * 2^m * H_{2m}(k) * e^{-k^2/2}
    
    # So F(g)(k) = sqrt(2π) * sum_{m=0}^{6} p_m * (-1)^m * 2^m * H_{2m}(k) * e^{-k^2/2}
    
    # Let R(k) = sum_{m=0}^{6} p_m * (-1)^m * 2^m * H_{2m}(k)
    # We need to express R(k) as a polynomial in k^2.
    
    # H_{2m}(k) for m=0..6:
    # H0(k) = 1
    # H2(k) = k^2 - 1
    # H4(k) = k^4 - 6k^2 + 3
    # H6(k) = k^6 - 15k^4 + 45k^2 - 15
    # H8(k) = k^8 - 28k^6 + 420k^4 - 1680k^2 + 105
    # H10(k) = k^10 - 45k^8 + 1260k^6 - 7560k^4 + 6300k^2 - 630
    # H12(k) = k^12 - 66k^10 + 2310k^8 - 27720k^6 + 39960k^4 - 13860k^2 + 10395
    
    # R(k) = sum_{m=0}^{6} r_m k^{2m} where:
    # r_m = sum_{j=m}^{6} p_j * (-1)^j * 2^j * [coeff of k^{2m} in H_{2j}(k)]
    
    # Coefficients of H_{2j}(k) as polynomial in k^2:
    # H0: [1]
    # H2: [-1, 1]  (const, k^2)
    # H4: [3, -6, 1]
    # H6: [-15, 45, -15, 1]
    # H8: [105, -1680, 420, -28, 1]
    # H10: [-630, 6300, -7560, 1260, -45, 1]
    # H12: [10395, -13860, 39960, -27720, 2310, -66, 1]
    
    H_coeffs = [
        [1],
        [-1, 1],
        [3, -6, 1],
        [-15, 45, -15, 1],
        [105, -1680, 420, -28, 1],
        [-630, 6300, -7560, 1260, -45, 1],
        [10395, -13860, 39960, -27720, 2310, -66, 1]
    ]
    
    r = np.zeros(7)
    for j in range(7):
        factor = p[j] * ((-1)**j) * (2**j)
        for m in range(j+1):
            r[m] += factor * H_coeffs[j][m]
    
    # Now F(g)(k) = sqrt(2π) * R(k) * e^{-k^2/2}
    # <1> for F(g) = 2π * ∫ R(k)^2 e^{-k^2} dk = 2π * sum_n r_n^2_combined * I(n)
    # where R(k)^2 = sum_n s_n k^{2n}
    
    s = np.zeros(13)
    for i in range(7):
        for j in range(7):
            if i + j <= 12:
                s[i+j] += r[i] * r[j]
    
    ft_norm_sq = 2.0 * np.pi * sum(s[m] * _I_CACHE[m] for m in range(13))
    ft_k2_norm = 2.0 * np.pi * sum(s[m] * _I_CACHE[m+1] for m in range(13))
    
    if ft_norm_sq <= 0 or not np.isfinite(ft_norm_sq):
        return 1e10
    
    A_g_sq = x2_norm / norm_sq
    A_Fg_sq = ft_k2_norm / ft_norm_sq
    
    if A_g_sq <= 0 or A_Fg_sq <= 0:
        return 1e10
    
    return np.sqrt(A_g_sq * A_Fg_sq)


def construct():
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin before 210s limit
    
    # Seed values
    seed = np.array([-113.0/100, 1.0/25, 1.0/3240])
    
    # Verify seed
    seed_val = _compute_raw_value(*seed)
    
    best_x = seed.copy()
    best_val = seed_val
    
    def objective(x):
        return _compute_raw_value(x[0], x[1], x[2])
    
    # Phase 1: Differential evolution for global exploration
    if time.time() < deadline - 60:
        try:
            bounds = [(-10, 10), (-10, 10), (-10, 10)]
            de_result = differential_evolution(objective, bounds, seed=42,
                                                maxiter=100, popsize=15,
                                                tol=1e-10, mutation=(0.5, 1.0),
                                                recombination=0.8,
                                                init='latinhypercube')
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_x = de_result.x
        except Exception:
            pass
    
    if time.time() > deadline - 50:
        return best_x
    
    # Phase 2: Nelder-Mead local refinement
    try:
        nm_result = minimize(objective, best_x, method='Nelder-Mead',
                            options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        if nm_result.fun < best_val:
            best_val = nm_result.fun
            best_x = nm_result.x
    except Exception:
        pass
    
    if time.time() > deadline - 30:
        return best_x
    
    # Phase 3: BFGS polish
    try:
        bfgs_result = minimize(objective, best_x, method='BFGS',
                              options={'maxiter': 5000, 'gtol': 1e-14})
        if bfgs_result.fun < best_val:
            best_val = bfgs_result.fun
            best_x = bfgs_result.x
    except Exception:
        pass
    
    if time.time() > deadline - 15:
        return best_x
    
    # Phase 4: L-BFGS-B final polish
    try:
        lbfgs_result = minimize(objective, best_x, method='L-BFGS-B',
                               options={'maxiter': 10000, 'gtol': 1e-16, 'ftol': 1e-16})
        if lbfgs_result.fun < best_val:
            best_val = lbfgs_result.fun
            best_x = lbfgs_result.x
    except Exception:
        pass
    
    return best_x
# EVOLVE-BLOCK-END
