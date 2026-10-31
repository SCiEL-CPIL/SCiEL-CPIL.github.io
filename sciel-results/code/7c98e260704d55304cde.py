"""Optimized Fourier-invariant Hermite test function coefficients."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize, Bounds, LinearConstraint

# Deadline guard
_DEADLINE_SECONDS = 190  # safety margin before 210s limit
_start_time = time.time()


def _time_left():
    return _DEADLINE_SECONDS - (time.time() - _start_time)


def _hermite_coeffs(n, x):
    """
    Compute probabilists' Hermite polynomial H_n(x) coefficients.
    H_0 = 1, H_1 = x, H_2 = x^2 - 1, H_3 = x^3 - 3x, ...
    Recurrence: H_{n+1} = x*H_n - n*H_{n-1}
    Returns array of coefficients from lowest to highest degree.
    """
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 1.0])
    
    h_prev = np.array([1.0])  # H_0
    h_curr = np.array([0.0, 1.0])  # H_1
    
    for k in range(2, n + 1):
        # H_k = x*H_{k-1} - (k-1)*H_{k-2}
        # x*H_{k-1} shifts coefficients up by 1
        x_h_curr = np.zeros(len(h_curr) + 1)
        x_h_curr[1:] = h_curr
        # (k-1)*H_{k-2}
        scaled_prev = (k - 1) * h_prev
        # Pad to same length
        max_len = max(len(x_h_curr), len(scaled_prev))
        if len(x_h_curr) < max_len:
            x_h_curr = np.pad(x_h_curr, (0, max_len - len(x_h_curr)))
        if len(scaled_prev) < max_len:
            scaled_prev = np.pad(scaled_prev, (0, max_len - len(scaled_prev)))
        
        h_next = x_h_curr - scaled_prev
        h_prev, h_curr = h_curr, h_next
    
    return h_curr


def _hermite_matrix(n_max):
    """
    Build matrix where row n contains coefficients of H_n(x) (probabilists').
    Returns list of arrays.
    """
    coeffs_list = []
    for n in range(n_max + 1):
        coeffs_list.append(_hermite_coeffs(n))
    return coeffs_list


def _compute_moments(coeff_vec):
    """
    Given coefficients c0, c4, c8 (and derived c12), compute the moments
    needed for A(f) and A(F).
    
    f(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    
    We need:
    - A(f) = (∫ f^2 dx)^(-1/2) * (∫ x^2 f^2 dx)^(1/2)  [standard deviation]
    - A(F) = (∫ |F(f)|^2 dx)^(-1/2) * (∫ x^2 |F(f)|^2 dx)^(1/2)
    
    For Fourier-invariant functions, A(f)*A(F) >= 1/2 (uncertainty principle).
    The upper bound C4 is what we're minimizing.
    
    Actually, for the Hermite-Gaussian functions, the Fourier transform is related
    to the same function with a phase factor. For even Hermite polynomials,
    F(H_n * e^{-x^2/2}) = i^n * H_n * e^{-x^2/2} (up to normalization).
    
    So A(f) = A(F) for even combinations, and the product is A(f)^2.
    
    Let's compute:
    ∫ f^2 dx and ∫ x^2 f^2 dx
    
    f(x) = g(x) * exp(-x^2/2) where g = sum c_k H_k
    
    ∫ f^2 dx = ∫ g^2 exp(-x^2) dx
    ∫ x^2 f^2 dx = ∫ x^2 g^2 exp(-x^2) dx
    
    Using orthogonality: ∫ H_m H_n exp(-x^2) dx = sqrt(pi) * 2^n * n! * delta_{mn}
    
    So ∫ f^2 dx = sqrt(pi) * sum_k c_k^2 * 2^k * k!
    
    For ∫ x^2 f^2 dx, we use x^2 = H_2 + 1 = H_2 + H_0
    So x^2 g^2 involves products of Hermite polynomials.
    
    Alternatively, use the recurrence: x H_n = H_{n+1} + n H_{n-1}
    x^2 H_n = H_{n+2} + 2(n+1) H_n + n(n-1) H_{n-2}
    
    So ∫ x^2 H_n^2 exp(-x^2) dx = sqrt(pi) * [2^{n+2}(n+2)! + 2*2^n*n!*(n+1) + 2^{n-2}(n-2)!*n*(n-1)] * delta...
    
    Wait, let me be more careful. We need ∫ x^2 H_m H_n exp(-x^2) dx.
    
    x^2 H_m = H_{m+2} + 2(m+1)H_m + m(m-1)H_{m-2}
    
    So ∫ x^2 H_m H_n exp(-x^2) dx = sqrt(pi)[2^{m+2}(m+2)! delta_{m+2,n} + 2(m+1)2^m m! delta_{m,n} + m(m-1)2^{m-2}(m-2)! delta_{m-2,n}]
    
    For m=n: ∫ x^2 H_n^2 exp(-x^2) dx = sqrt(pi)[2^n n! * 2(n+1) + 2^{n+2}(n+2)! delta_{n,n+2}... wait no.
    
    Let me redo. ∫ x^2 H_m H_n e^{-x^2} dx.
    
    Using x^2 H_m = H_{m+2} + 2(m+1)H_m + m(m-1)H_{m-2}:
    
    ∫ x^2 H_m H_n e^{-x^2} dx = ∫ [H_{m+2} + 2(m+1)H_m + m(m-1)H_{m-2}] H_n e^{-x^2} dx
    
    = sqrt(pi) [2^{m+2}(m+2)! delta_{m+2,n} + 2(m+1) 2^m m! delta_{m,n} + m(m-1) 2^{m-2}(m-2)! delta_{m-2,n}]
    
    For m=n:
    = sqrt(pi) [2(m+1) 2^m m! + m(m-1) 2^{m-2}(m-2)!]  (the first term with delta_{m+2,m}=0)
    
    Wait, delta_{m+2,n} with n=m is 0 unless m+2=m which is never. So:
    = sqrt(pi) [2(m+1) 2^m m! + m(m-1) 2^{m-2}(m-2)!]
    
    For m=0: sqrt(pi) [2(1) 1 + 0] = 2 sqrt(pi)
    Check: ∫ x^2 * 1 * 1 * e^{-x^2} dx = sqrt(pi)/2. Hmm, that's not matching.
    
    Let me recheck the orthogonality. ∫ H_m H_n e^{-x^2} dx = sqrt(pi) 2^n n! delta_{mn}
    
    For m=n=0: ∫ 1*e^{-x^2} dx = sqrt(pi). And sqrt(pi)*2^0*0! = sqrt(pi). OK.
    
    For m=n=1: ∫ x^2 e^{-x^2} dx = sqrt(pi)/2. And sqrt(pi)*2^1*1! = 2 sqrt(pi). Not matching!
    
    So the orthogonality is: ∫ H_m H_n e^{-x^2/2} dx = sqrt(2*pi) 2^n n! delta_{mn} for probabilists' Hermite?
    
    Actually, I think I'm confusing physicists' and probabilists' Hermite polynomials.
    
    Probabilists' Hermite: He_n(x), recurrence He_{n+1} = x He_n - n He_{n-1}
    Orthogonality: ∫ He_m He_n e^{-x^2/2} dx = sqrt(2*pi) n! delta_{mn}
    
    Physicists' Hermite: H_n(x), recurrence H_{n+1} = 2x H_n - 2n H_{n-1}
    Orthogonality: ∫ H_m H_n e^{-x^2} dx = sqrt(pi) 2^n n! delta_{mn}
    
    The task says "Hermite-polynomial times Gaussian test function". The Fourier-invariant property suggests we're using the standard Hermite functions.
    
    Let me use physicists' Hermite polynomials with e^{-x^2/2}... actually, let me just use numerical integration to be safe, or use the known formula.
    
    For the standard Hermite functions psi_n(x) = (2^n n! sqrt(pi))^{-1/2} H_n(x) e^{-x^2/2}, these are orthonormal and F(psi_n) = (-i)^n psi_n.
    
    So if f = sum c_n psi_n, then F(f) = sum c_n (-i)^n psi_n.
    
    For even n, (-i)^n = (-1)^{n/2}.
    
    A(f) = std(f) = (∫ x^2 |f|^2 / ∫ |f|^2)^{1/2}
    
    For f = c0 psi_0 + c4 psi_4 + c8 psi_8 + c12 psi_12:
    
    ∫ |f|^2 = |c0|^2 + |c4|^2 + |c8|^2 + |c12|^2 (orthonormality)
    
    ∫ x^2 |f|^2 = sum_{m,n} c_m c_n^* ∫ x^2 psi_m psi_n
    
    The matrix M_{mn} = ∫ x^2 psi_m psi_n dx.
    
    Using x psi_n = sqrt((n+1)/2) psi_{n+1} + sqrt(n/2) psi_{n-1}:
    x^2 psi_n = sqrt((n+1)/2) x psi_{n+1} + sqrt(n/2) x psi_{n-1}
    = sqrt((n+1)/2)[sqrt((n+2)/2) psi_{n+2} + sqrt((n+1)/2) psi_n] + sqrt(n/2)[sqrt(n/2) psi_n + sqrt((n-1)/2) psi_{n-2}]
    = sqrt((n+1)(n+2)/4) psi_{n+2} + [(n+1)/2 + n/2] psi_n + sqrt(n(n-1)/4) psi_{n-2}
    = (1/2)sqrt((n+1)(n+2)) psi_{n+2} + (2n+1)/2 psi_n + (1/2)sqrt(n(n-1)) psi_{n-2}
    
    So ∫ x^2 psi_m psi_n = (1/2)sqrt((n+1)(n+2)) delta_{m,n+2} + (2n+1)/2 delta_{m,n} + (1/2)sqrt(n(n-1)) delta_{m,n-2}
    
    For our indices {0, 4, 8, 12}:
    
    M_{00} = (2*0+1)/2 = 1/2
    M_{44} = (2*4+1)/2 = 9/2
    M_{88} = (2*8+1)/2 = 17/2
    M_{12,12} = (2*12+1)/2 = 25/2
    
    M_{0,2} would be (1/2)sqrt(1*2) = 1/sqrt(2), but 2 is not in our set.
    M_{2,0} = (1/2)sqrt(0*(-1))... wait, for n=0: (1/2)sqrt(0*(-1)) is 0.
    
    Actually M_{mn} = ∫ x^2 psi_m psi_n. Let me use the formula:
    M_{mn} = (1/2)sqrt((n+1)(n+2)) delta_{m,n+2} + (2n+1)/2 delta_{m,n} + (1/2)sqrt(n(n-1)) delta_{m,n-2}
    
    For m=4, n=0: delta_{4,2}=0, delta_{4,0}=0, delta_{4,-2}=0. So M_{40}=0.
    For m=0, n=4: delta_{0,6}=0, delta_{0,4}=0, delta_{0,2}=0. So M_{04}=0.
    
    In general, M_{mn} is nonzero only if |m-n| is 0 or 2.
    
    So for our set {0,4,8,12}, the only nonzero off-diagonal elements would be between indices differing by 2, but none of our indices differ by 2 (they differ by 4). So M is diagonal!
    
    M = diag(1/2, 9/2, 17/2, 25/2)
    
    So ∫ x^2 |f|^2 = (1/2)|c0|^2 + (9/2)|c4|^2 + (17/2)|c8|^2 + (25/2)|c12|^2
    
    And ∫ |f|^2 = |c0|^2 + |c4|^2 + |c8|^2 + |c12|^2
    
    A(f)^2 = [(1/2)|c0|^2 + (9/2)|c4|^2 + (17/2)|c8|^2 + (25/2)|c12|^2] / [|c0|^2 + |c4|^2 + |c8|^2 + |c12|^2]
    
    Similarly for F(f) = sum c_n (-i)^n psi_n. For even n, (-i)^n is real (±1), so |F(f)|^2 has the same form.
    
    A(F)^2 = A(f)^2
    
    So A(f)*A(F) = A(f)^2 = [(1/2)c0^2 + (9/2)c4^2 + (17/2)c8^2 + (25/2)c12^2] / [c0^2 + c4^2 + c8^2 + c12^2]
    
    where c's are real (we can take them real WLOG).
    
    The constraint g(0) = 0 means f(0) = 0, i.e., sum c_n psi_n(0) = 0.
    
    psi_n(0) = (2^n n! sqrt(pi))^{-1/2} H_n(0) e^0
    
    H_0(0) = 1
    H_4(0) = -3 (physicists') or He_4(0) = 3 (probabilists')
    
    Let me use physicists': H_0(0)=1, H_4(0)=-3, H_8(0)=105, H_12(0)=-10395
    
    psi_n(0) = H_n(0) / sqrt(2^n n! sqrt(pi))
    
    psi_0(0) = 1/sqrt(sqrt(pi))
    psi_4(0) = -3/sqrt(16*24*sqrt(pi)) = -3/sqrt(384 sqrt(pi))
    psi_8(0) = 105/sqrt(256*40320*sqrt(pi))
    psi_12(0) = -10395/sqrt(4096*479001600*sqrt(pi))
    
    The constraint is: c0*psi_0(0) + c4*psi_4(0) + c8*psi_8(0) + c12*psi_12(0) = 0
    
    So c12 = -(c0*psi_0(0) + c4*psi_4(0) + c8*psi_8(0)) / psi_12(0)
    
    This is a linear constraint. We can substitute and minimize the quadratic form.
    
    Let me define:
    a = [1/2, 9/2, 17/2, 25/2]
    b = [psi_0(0), psi_4(0), psi_8(0), psi_12(0)]
    
    Objective: (a0*c0^2 + a1*c4^2 + a2*c8^2 + a3*c12^2) / (c0^2 + c4^2 + c8^2 + c12^2)
    
    with c12 = -(b0*c0 + b1*c4 + b2*c8) / b3
    
    This is a generalized eigenvalue problem! The minimum of the Rayleigh quotient with a linear constraint.
    
    Actually, let's set up: minimize (c^T A c) / (c^T c) subject to b^T c = 0, where c = [c0, c4, c8, c12].
    
    This is equivalent to finding the smallest eigenvalue of A restricted to the subspace b^T c = 0.
    
    Or: minimize c^T A c subject to c^T c = 1 and b^T c = 0.
    
    Using Lagrange multipliers: A c = lambda c + mu b
    
    With 3 free variables (since one constraint), we can parameterize.
    
    Let me just do this numerically with scipy.optimize.
    """
    pass


def construct():
    """
    Return three coefficients (c0, c4, c8) of the Hermite combination.
    The degree-12 coefficient is derived to impose g(0) = 0.
    Score is the upper bound on C4 = A(f)*A(F).
    """
    # Precompute constants
    # Physicists' Hermite polynomials at 0
    H0_0 = 1.0
    H4_0 = -3.0
    H8_0 = 105.0
    H12_0 = -10395.0
    
    # psi_n(0) = H_n(0) / sqrt(2^n * n! * sqrt(pi))
    pi = np.pi
    sqrt_pi = np.sqrt(pi)
    
    def psi_n_0(n):
        # 2^n * n! * sqrt(pi)
        log_val = n * np.log(2) + np.math.lgamma(n + 1) + 0.5 * np.log(pi)
        val = np.exp(log_val)
        H_n_0 = {0: 1.0, 4: -3.0, 8: 105.0, 12: -10395.0}[n]
        return H_n_0 / np.sqrt(val)
    
    psi0 = psi_n_0(0)
    psi4 = psi_n_0(4)
    psi8 = psi_n_0(8)
    psi12 = psi_n_0(12)
    
    # A matrix diagonal elements: (2n+1)/2 for n in {0,4,8,12}
    a0 = 0.5
    a4 = 4.5
    a8 = 8.5
    a12 = 12.5
    
    # Constraint: c0*psi0 + c4*psi4 + c8*psi8 + c12*psi12 = 0
    # => c12 = -(c0*psi0 + c4*psi4 + c8*psi8) / psi12
    
    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = -(c0 * psi0 + c4 * psi4 + c8 * psi8) / psi12
        num = a0 * c0**2 + a4 * c4**2 + a8 * c8**2 + a12 * c12**2
        den = c0**2 + c4**2 + c8**2 + c12**2
        if den < 1e-30:
            return 1e10
        return num / den
    
    def objective_grad(coeffs):
        c0, c4, c8 = coeffs
        c12 = -(c0 * psi0 + c4 * psi4 + c8 * psi8) / psi12
        num = a0 * c0**2 + a4 * c4**2 + a8 * c8**2 + a12 * c12**2
        den = c0**2 + c4**2 + c8**2 + c12**2
        if den < 1e-30:
            return np.zeros(3)
        
        # d(num)/dc0 = 2*a0*c0 + 2*a12*c12*(-psi0/psi12)
        dc12_dc0 = -psi0 / psi12
        dc12_dc4 = -psi4 / psi12
        dc12_dc8 = -psi8 / psi12
        
        dnum_dc0 = 2*a0*c0 + 2*a12*c12*dc12_dc0
        dnum_dc4 = 2*a4*c4 + 2*a12*c12*dc12_dc4
        dnum_dc8 = 2*a8*c8 + 2*a12*c12*dc12_dc8
        
        dden_dc0 = 2*c0 + 2*c12*dc12_dc0
        dden_dc4 = 2*c4 + 2*c12*dc12_dc4
        dden_dc8 = 2*c8 + 2*c12*dc12_dc8
        
        f = num / den
        df_dc0 = (dnum_dc0 * den - num * dden_dc0) / den**2
        df_dc4 = (dnum_dc4 * den - num * dden_dc4) / den**2
        df_dc8 = (dnum_dc8 * den - num * dden_dc8) / den**2
        
        return np.array([df_dc0, df_dc4, df_dc8])
    
    # Start from the prior state of the art
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Try multiple starting points
    start_points = [
        x0,
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, 1.0]),
        np.array([1.0, -1.0, 1.0]),
        np.array([1.0, 1.0, -1.0]),
        np.array([0.5, 0.5, 0.5]),
        np.array([-0.5, 0.5, 0.5]),
    ]
    
    for sp in start_points:
        if _time_left() < 10:
            break
        try:
            res = minimize(objective, sp, jac=objective_grad, method='BFGS',
                          options={'maxiter': 1000, 'gtol': 1e-12})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Polish with high precision
    if _time_left() > 5:
        try:
            res = minimize(objective, best_x, jac=objective_grad, method='BFGS',
                          options={'maxiter': 5000, 'gtol': 1e-15})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
