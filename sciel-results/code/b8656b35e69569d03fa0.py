"""Optimized uncertainty bound via analytic Hermite-Gaussian moments + L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195  # 15s safety margin before 210s kill


def _hermite_coeffs():
    """Return coefficient arrays for H0, H4, H8, H12 (even polynomials in x)."""
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    h0 = np.array([1.0])
    h4 = np.array([3.0, -6.0, 1.0])
    h8 = np.array([105.0, -420.0, 210.0, -28.0, 1.0])
    h12 = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
    return h0, h4, h8, h12


def _poly_sq_moments(coefs):
    """
    Compute moments of P(x)^2 * exp(-x^2) where P(x) = sum_k c_k * x^(2k).
    Returns array of I_{2m} = integral_{-inf}^{inf} x^{2m} P(x)^2 exp(-x^2) dx.
    We need moments up to order 2*12 = 24, so m up to 12.
    """
    # P(x)^2 has terms x^{2j} x^{2k} = x^{2(j+k)} with coeff c_j * c_k
    # I_{2m} = sum_{j+k=m} c_j c_k * integral x^{2(j+k)} exp(-x^2) dx
    # integral x^{2n} exp(-x^2) dx = sqrt(pi) * (2n-1)!! = sqrt(pi) * (2n)! / (4^n n!)
    # Let's compute up to degree 24 (n up to 12)
    
    n_terms = len(coefs)
    max_deg = 2 * (n_terms - 1)  # max power of x in P(x)^2 is 2*(max_k) = 2*(n_terms-1)
    # We need moments I_{2m} for m = 0, 1, ..., max_deg/2
    # Actually we need the full set of moments of P(x)^2 exp(-x^2)
    
    # Precompute Gaussian moments: G[n] = integral x^{2n} exp(-x^2) dx = sqrt(pi) * (2n)!/(4^n n!)
    # We need G[n] for n = 0, 1, ..., max_deg
    max_n = max_deg  # since P(x)^2 has terms up to x^{2*max_k}, and we multiply by x^{2m}
    # Wait, let me reconsider. We need:
    # A(f) involves integral |f|^2 and integral x^2 |f|^2
    # For g(x) = P(x) exp(-x^2/2), |g|^2 = P(x)^2 exp(-x^2)
    # A(f) = sqrt( integral x^2 |g|^2 dx / integral |g|^2 dx )  [standard deviation]
    # Actually, the uncertainty is typically sqrt(<x^2>) * sqrt(<xi^2>) where <xi^2> is from Fourier side.
    
    # Let me think about this more carefully.
    # g(x) = P(x) * exp(-x^2/2)
    # |g(x)|^2 = P(x)^2 * exp(-x^2)
    # 
    # A(f) for position: sqrt( <x^2> ) where <x^2> = integral x^2 |g|^2 / integral |g|^2
    # A(f) for frequency: sqrt( <xi^2> ) where <xi^2> = integral xi^2 |G|^2 / integral |G|^2
    # By Plancherel, integral |G|^2 = integral |g|^2
    # And <xi^2> = integral xi^2 |G(xi)|^2 dxi / integral |G|^2
    # Using the fact that xi * G(xi) = F{g'(x)} (up to constants), 
    # <xi^2> = integral |g'(x)|^2 dx / integral |g(x)|^2 dx
    # 
    # So the product A(f)*A(F) = sqrt(<x^2>) * sqrt(<xi^2>) = sqrt( <x^2> * <xi^2> )
    # where <x^2> = M_2 / M_0 and <xi^2> = N_2 / M_0
    # M_0 = integral |g|^2 = integral P^2 exp(-x^2)
    # M_2 = integral x^2 |g|^2 = integral x^2 P^2 exp(-x^2)
    # N_2 = integral |g'|^2
    #
    # g'(x) = P'(x) exp(-x^2/2) + P(x) * (-x) exp(-x^2/2) = [P'(x) - x P(x)] exp(-x^2/2)
    # |g'|^2 = [P'(x) - x P(x)]^2 exp(-x^2)
    # N_2 = integral [P'(x) - x P(x)]^2 exp(-x^2) dx
    
    # So I need:
    # M_0 = integral P^2 exp(-x^2) dx
    # M_2 = integral x^2 P^2 exp(-x^2) dx
    # N_2 = integral [P' - xP]^2 exp(-x^2) dx
    
    # All of these can be computed from the moments of P^2 and (P'-xP)^2 times exp(-x^2).
    
    # Let me define a helper that, given a polynomial Q(x) = sum d_k x^{2k}, 
    # computes integral Q(x)^2 exp(-x^2) dx and integral x^2 Q(x)^2 exp(-x^2) dx.
    
    def _int_q2_exp(q_coefs, x2_weight=False):
        """
        Compute integral Q(x)^2 exp(-x^2) dx or integral x^2 Q(x)^2 exp(-x^2) dx
        where Q(x) = sum_k q_k x^{2k}.
        """
        n = len(q_coefs)
        max_power = 2 * (n - 1)  # max power in Q^2
        # Q^2(x) = sum_{j,k} q_j q_k x^{2(j+k)}
        # integral x^{2(j+k)} exp(-x^2) dx = G[j+k] where G[m] = integral x^{2m} exp(-x^2)
        # For x^2 weighted: integral x^2 x^{2(j+k)} exp(-x^2) = G[j+k+1]
        
        result = 0.0
        for j in range(n):
            for k in range(n):
                m = j + k
                if x2_weight:
                    m += 1
                # G[m] = sqrt(pi) * (2m)! / (4^m * m!)
                # Compute using log or direct
                if m == 0:
                    g_val = np.sqrt(np.pi)
                else:
                    # (2m)! / (4^m m!) = (2m-1)!! / 2^m ... let's just compute directly
                    # Use the recurrence: G[m] = G[m-1] * (2m-1)/2
                    pass
                # Let's precompute G values
                result += q_coefs[j] * q_coefs[k] * _gaussian_moment(m)
        return result
    
    return _int_q2_exp


def _gaussian_moment(m):
    """G[m] = integral_{-inf}^{inf} x^{2m} exp(-x^2) dx = sqrt(pi) * (2m)!/(4^m m!)"""
    # Use recurrence: G[0] = sqrt(pi), G[m] = (2m-1)/2 * G[m-1]
    if m == 0:
        return np.sqrt(np.pi)
    # Compute iteratively to avoid overflow
    result = np.sqrt(np.pi)
    for i in range(1, m + 1):
        result *= (2 * i - 1) / 2.0
    return result


def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product A(f)*A(F) for g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    with a12 chosen so that P(0) = 0.
    """
    h0, h4, h8, h12 = _hermite_coeffs()
    
    # P(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = a0 + a4*3 + a8*105 + a12*945
    # Set P(0) = 0: a12 = -(a0 + 3*a4 + 105*a8) / 945
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    # Build P(x) = sum c_k x^{2k}
    # H0: c_0 = 1
    # H4: c_0=3, c_1=-6, c_2=1
    # H8: c_0=105, c_1=-420, c_2=210, c_3=-28, c_4=1
    # H12: c_0=945, c_1=-7560, c_2=13860, c_3=-6930, c_4=1155, c_5=-66, c_6=1
    
    # P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # Coefficient of x^{2k} in P:
    # k=0: a0*1 + a4*3 + a8*105 + a12*945 = 0 (by constraint)
    # k=1: a4*(-6) + a8*(-420) + a12*(-7560)
    # k=2: a4*1 + a8*210 + a12*13860
    # k=3: a8*(-28) + a12*(-6930)
    # k=4: a8*1 + a12*1155
    # k=5: a12*(-66)
    # k=6: a12*1
    
    p_coefs = np.zeros(7)  # x^0, x^2, x^4, x^6, x^8, x^10, x^12
    p_coefs[0] = 0.0  # by constraint
    p_coefs[1] = a4 * (-6.0) + a8 * (-420.0) + a12 * (-7560.0)
    p_coefs[2] = a4 * 1.0 + a8 * 210.0 + a12 * 13860.0
    p_coefs[3] = a8 * (-28.0) + a12 * (-6930.0)
    p_coefs[4] = a8 * 1.0 + a12 * 1155.0
    p_coefs[5] = a12 * (-66.0)
    p_coefs[6] = a12 * 1.0
    
    # M_0 = integral P(x)^2 exp(-x^2) dx
    # M_2 = integral x^2 P(x)^2 exp(-x^2) dx
    # N_2 = integral [P'(x) - x*P(x)]^2 exp(-x^2) dx
    
    # For M_0 and M_2:
    # P(x)^2 = sum_{j,k} p_j p_k x^{2(j+k)}
    # M_0 = sum_{j,k} p_j p_k G[j+k]
    # M_2 = sum_{j,k} p_j p_k G[j+k+1]
    
    n_p = len(p_coefs)
    M_0 = 0.0
    M_2 = 0.0
    for j in range(n_p):
        if p_coefs[j] == 0:
            continue
        for k in range(n_p):
            if p_coefs[k] == 0:
                continue
            val = p_coefs[j] * p_coefs[k]
            M_0 += val * _gaussian_moment(j + k)
            M_2 += val * _gaussian_moment(j + k + 1)
    
    # For N_2: Q(x) = P'(x) - x*P(x)
    # P(x) = sum p_k x^{2k}
    # P'(x) = sum 2k p_k x^{2k-1}
    # x*P(x) = sum p_k x^{2k+1}
    # Q(x) = sum [2k p_k - p_k] x^{2k-1} for k>=1, and for k=0: -p_0 x
    # Wait: P'(x) = sum_{k=1}^{6} 2k p_k x^{2k-1}
    # x*P(x) = sum_{k=0}^{6} p_k x^{2k+1}
    # Q(x) = -p_0 x + sum_{k=1}^{6} (2k p_k - p_k) x^{2k-1} + ... 
    # Let me be more careful.
    # Q(x) = P'(x) - x P(x)
    # P'(x) = 2 p_1 x + 4 p_2 x^3 + 6 p_3 x^5 + 8 p_4 x^7 + 10 p_5 x^9 + 12 p_6 x^11
    # x P(x) = p_0 x + p_1 x^3 + p_2 x^5 + p_3 x^7 + p_4 x^9 + p_5 x^11 + p_6 x^13
    # Q(x) = (2p_1 - p_0)x + (4p_2 - p_1)x^3 + (6p_3 - p_2)x^5 + (8p_4 - p_3)x^7 + (10p_5 - p_4)x^9 + (12p_6 - p_5)x^11 - p_6 x^13
    #
    # So Q is an odd polynomial. Let Q(x) = sum_{m=0}^{6} q_m x^{2m+1}
    # q_0 = 2*p_1 - p_0
    # q_1 = 4*p_2 - p_1
    # q_2 = 6*p_3 - p_2
    # q_3 = 8*p_4 - p_3
    # q_4 = 10*p_5 - p_4
    # q_5 = 12*p_6 - p_5
    # q_6 = -p_6
    
    q_coefs = np.zeros(7)  # coefficients of x^1, x^3, x^5, x^7, x^9, x^11, x^13
    q_coefs[0] = 2.0 * p_coefs[1] - p_coefs[0]
    q_coefs[1] = 4.0 * p_coefs[2] - p_coefs[1]
    q_coefs[2] = 6.0 * p_coefs[3] - p_coefs[2]
    q_coefs[3] = 8.0 * p_coefs[4] - p_coefs[3]
    q_coefs[4] = 10.0 * p_coefs[5] - p_coefs[4]
    q_coefs[5] = 12.0 * p_coefs[6] - p_coefs[5]
    q_coefs[6] = -p_coefs[6]
    
    # N_2 = integral Q(x)^2 exp(-x^2) dx
    # Q(x)^2 = sum_{j,k} q_j q_k x^{(2j+1)+(2k+1)} = sum_{j,k} q_j q_k x^{2(j+k+1)}
    # integral x^{2(j+k+1)} exp(-x^2) dx = G[j+k+1]
    
    n_q = len(q_coefs)
    N_2 = 0.0
    for j in range(n_q):
        if q_coefs[j] == 0:
            continue
        for k in range(n_q):
            if q_coefs[k] == 0:
                continue
            N_2 += q_coefs[j] * q_coefs[k] * _gaussian_moment(j + k + 1)
    
    if M_0 <= 0:
        return np.inf
    
    # A(f) * A(F) = sqrt(<x^2>) * sqrt(<xi^2>) = sqrt(M_2/M_0) * sqrt(N_2/M_0) = sqrt(M_2 * N_2 / M_0^2)
    bound = np.sqrt(M_2 * N_2) / M_0
    return bound


def _objective(x):
    """Objective function for optimization."""
    if time.time() > _DEADLINE:
        return np.inf
    a0, a4, a8 = x
    try:
        return _compute_bound(a0, a4, a8)
    except:
        return np.inf


def _objective_grad(x):
    """Analytic gradient via finite differences (central) for reliability."""
    if time.time() > _DEADLINE:
        return np.array([0.0, 0.0, 0.0])
    eps = 1e-8
    grad = np.zeros(3)
    f0 = _objective(x)
    if f0 == np.inf:
        return grad
    for i in range(3):
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        fp = _objective(xp)
        fm = _objective(xm)
        if fp == np.inf or fm == np.inf:
            grad[i] = 0.0
        else:
            grad[i] = (fp - fm) / (2 * eps)
    return grad


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Returns np.array([a0, a4, a8]).
    """
    # Start from the prior state of the art
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_x = x0.copy()
    best_val = _compute_bound(x0[0], x0[1], x0[2])
    
    # Try L-BFGS-B with multiple starting points
    starts = [
        x0,
        x0 * 0.5,
        x0 * 2.0,
        np.array([-1.0, 0.1, 0.001]),
        np.array([-0.5, 0.05, 0.0005]),
        np.array([-1.5, 0.15, 0.0015]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([-0.8, 0.08, 0.0008]),
    ]
    
    for start in starts:
        if time.time() > _DEADLINE - 5:
            break
        try:
            res = minimize(
                _objective,
                start,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 100000},
            )
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Also try a few Nelder-Mead runs from different starts for robustness
    nm_starts = [
        x0,
        x0 * 1.1,
        x0 * 0.9,
    ]
    for start in nm_starts:
        if time.time() > _DEADLINE - 5:
            break
        try:
            res = minimize(
                _objective,
                start,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
