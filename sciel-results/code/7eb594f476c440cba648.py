"""Uncertainty inequality optimization via generalized eigenvalue formulation in Hermite basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

_DEADLINE = time.time() + 195  # 15s safety margin before 210s kill


def _precompute_I():
    """Precompute I[n] = integral x^(2n) e^(-x^2) dx for n=0..24."""
    I = np.zeros(25)
    I[0] = np.sqrt(np.pi)
    for n in range(1, 25):
        I[n] = (2.0 * n - 1.0) / 2.0 * I[n - 1]
    return I


_I = _precompute_I()


def _hermite_coeffs():
    """
    Return coefficient arrays for H0, H4, H8, H12 in the basis {1, x^2, x^4, x^6, x^8, x^10, x^12}.
    Each is a length-7 array where entry k is the coefficient of x^(2k).
    """
    # H0 = 1
    h0 = np.zeros(7)
    h0[0] = 1.0
    # H4 = x^4 - 6x^2 + 3
    h4 = np.zeros(7)
    h4[0] = 3.0
    h4[1] = -6.0
    h4[2] = 1.0
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    h8 = np.zeros(7)
    h8[0] = 105.0
    h8[1] = -420.0
    h8[2] = 210.0
    h8[3] = -28.0
    h8[4] = 1.0
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    h12 = np.zeros(7)
    h12[0] = 945.0
    h12[1] = -7560.0
    h12[2] = 13860.0
    h12[3] = -6930.0
    h12[4] = 1155.0
    h12[5] = -66.0
    h12[6] = 1.0
    return h0, h4, h8, h12


_H0, _H4, _H8, _H12 = _hermite_coeffs()


def _build_moment_matrices():
    """
    Build moment matrices M_k (k=0,1,2) for position and N_k for momentum.
    M_k[i,j] = integral x^(2k) H_i(x) H_j(x) e^(-x^2) dx
    N_k[i,j] = 2*pi * integral x^(2k) Q_i(x) Q_j(x) e^(-x^2) dx
    where Q_i is the Fourier-transformed Hermite (Q_k = (-1)^k H_k for even k).
    """
    # For position: P = sum a_i H_i, m_k = sum a_i a_j M_k[i,j]
    # For momentum: Q = sum a_i Q_i where Q_i = (-1)^i H_i (for even indices)
    # Since we only have even Hermite polynomials (0,4,8,12), Q_i = H_i for all (since (-1)^even = 1)
    # Wait: the Fourier transform of x^n e^(-x^2/2) involves i^n. For Hermite polynomials:
    # F[H_n(x) e^(-x^2/2)] = i^n H_n(k) e^(-k^2/2)
    # So for even n, i^n = (-1)^(n/2). For n=0: 1, n=4: -1, n=8: 1, n=12: -1
    # So Q_0 = H_0, Q_4 = -H_4, Q_8 = H_8, Q_12 = -H_12
    
    # Actually let me reconsider. The standard Hermite-Gaussian is H_n(x) * exp(-x^2/2).
    # Its Fourier transform is i^n H_n(k) exp(-k^2/2).
    # For n=0: i^0 = 1, so Q = H_0
    # For n=4: i^4 = 1, so Q = H_4
    # For n=8: i^8 = 1, so Q = H_8
    # For n=12: i^12 = 1, so Q = H_12
    # Wait, i^4 = (i^2)^2 = (-1)^2 = 1. i^8 = 1. i^12 = 1.
    # So all even Hermite polynomials have Fourier transform equal to themselves (up to sign convention).
    # 
    # But the original code used q = p * [1, -1, 1, -1, ...] which suggests alternating signs
    # for the x^(2k) terms, not for the Hermite indices. Let me re-examine.
    #
    # The original code defines P(x) = sum p_k x^(2k) where p_k are the coefficients of the
    # polynomial P = a0*H0 + a4*H4 + a8*H8 + a12*H12.
    # Then it computes m_k = integral x^(2k) |P|^2 e^(-x^2) dx
    # And n_k = 2*pi * integral x^(2k) |Q|^2 e^(-x^2) dx where Q_k = (-1)^k p_k
    #
    # This Q is the Fourier transform of P(x)e^(-x^2). The Fourier transform of x^n e^(-x^2)
    # involves derivatives and gives factors of i^n and (2pi)^{1/2} etc. The exact relation
    # for the full polynomial is more complex. The original code's approach of using
    # Q_k = (-1)^k p_k is an approximation or specific convention.
    #
    # Let me just follow the original code's convention: the momentum moments use
    # q_k = (-1)^k p_k as the polynomial coefficients.
    
    # For the moment matrix approach, I need to express everything in terms of the
    # Hermite basis coefficients (a0, a4, a8, a12) with the constraint a12 = f(a0,a4,a8).
    #
    # P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # = (a0 + 3*a4 + 105*a8 + 945*a12) * 1
    #   + (-6*a4 - 420*a8 - 7560*a12) * x^2
    #   + (a4 + 210*a8 + 13860*a12) * x^4
    #   + (-28*a8 - 6930*a12) * x^6
    #   + (a8 + 1155*a12) * x^8
    #   + (-66*a12) * x^10
    #   + (a12) * x^12
    #
    # With a12 = -(a0 + 3*a4 + 105*a8)/945:
    # p_0 = 0 (by construction)
    # p_1 = -6*a4 - 420*a8 - 7560*[-(a0+3*a4+105*a8)/945]
    #      = -6*a4 - 420*a8 + (7560/945)*(a0+3*a4+105*a8)
    #      = -6*a4 - 420*a8 + 8*(a0+3*a4+105*a8)
    #      = -6*a4 - 420*a8 + 8*a0 + 24*a4 + 840*a8
    #      = 8*a0 + 18*a4 + 420*a8
    #
    # p_2 = a4 + 210*a8 + 13860*[-(a0+3*a4+105*a8)/945]
    #      = a4 + 210*a8 - (13860/945)*(a0+3*a4+105*a8)
    #      = a4 + 210*a8 - 14.667*(a0+3*a4+105*a8)  -- let me use exact fractions
    #      13860/945 = 13860/945. 945*14 = 13230, 13860-13230=630, 630/945=2/3
    #      So 13860/945 = 14 + 2/3 = 44/3
    #      = a4 + 210*a8 - (44/3)*(a0+3*a4+105*a8)
    #      = a4 + 210*a8 - (44/3)*a0 - 44*a4 - (44/3)*105*a8
    #      = a4 + 210*a8 - (44/3)*a0 - 44*a4 - 1540*a8
    #      = -(44/3)*a0 + (1-44)*a4 + (210-1540)*a8
    #      = -(44/3)*a0 - 43*a4 - 1330*a8
    #
    # p_3 = -28*a8 - 6930*[-(a0+3*a4+105*a8)/945]
    #      = -28*a8 + (6930/945)*(a0+3*a4+105*a8)
    #      6930/945 = 7.333... = 22/3
    #      = -28*a8 + (22/3)*(a0+3*a4+105*a8)
    #      = -28*a8 + (22/3)*a0 + 22*a4 + 770*a8
    #      = (22/3)*a0 + 22*a4 + (770-28)*a8
    #      = (22/3)*a0 + 22*a4 + 742*a8
    #
    # p_4 = a8 + 1155*[-(a0+3*a4+105*a8)/945]
    #      = a8 - (1155/945)*(a0+3*a4+105*a8)
    #      1155/945 = 77/63 = 11/9
    #      = a8 - (11/9)*(a0+3*a4+105*a8)
    #      = a8 - (11/9)*a0 - 11*a4 - 121*a8
    #      = -(11/9)*a0 - 11*a4 + (1-121)*a8
    #      = -(11/9)*a0 - 11*a4 - 120*a8
    #
    # p_5 = -66*[-(a0+3*a4+105*a8)/945]
    #      = (66/945)*(a0+3*a4+105*a8)
    #      66/945 = 22/315
    #      = (22/315)*a0 + (22/105)*a4 + (22/3)*a8
    #
    # p_6 = -(a0+3*a4+105*a8)/945
    #      = -(1/945)*a0 - (3/945)*a4 - (105/945)*a8
    #      = -(1/945)*a0 - (1/315)*a4 - (1/9)*a8
    
    # So P(x) = p_1*x^2 + p_2*x^4 + p_3*x^6 + p_4*x^8 + p_5*x^10 + p_6*x^12
    # where each p_k is a linear function of (a0, a4, a8).
    
    # Let me define the transformation matrix T such that [p_1,...,p_6] = T * [a0,a4,a8]
    T = np.array([
        [8.0, 18.0, 420.0],
        [-44.0/3.0, -43.0, -1330.0],
        [22.0/3.0, 22.0, 742.0],
        [-11.0/9.0, -11.0, -120.0],
        [22.0/315.0, 22.0/105.0, 22.0/3.0],
        [-1.0/945.0, -1.0/315.0, -1.0/9.0]
    ])
    
    # Now m_k = sum_{i,j} p_i p_j I[k+i+j] where i,j index the x^(2i), x^(2j) terms
    # p_i for i=1..6 (p_0=0)
    # m_k = [p_1,...,p_6] * M_k * [p_1,...,p_6]^T where M_k[i,j] = I[k+i+j]
    
    # For momentum: q_k = (-1)^k p_k, so q = D * p where D = diag((-1)^1, (-1)^2, ..., (-1)^6)
    # q_k = (-1)^k p_k for k=1..6
    D = np.array([(-1.0)**k for k in range(1, 7)])
    
    # n_k = 2*pi * sum q_i q_j I[k+i+j] = 2*pi * [D*p] * M_k * [D*p]
    #     = 2*pi * p * (D M_k D) * p
    
    # So in terms of a = [a0,a4,a8]:
    # m_k = a^T (T^T M_k T) a
    # n_k = 2*pi * a^T (T^T D M_k D T) a
    
    # Define A_k = T^T M_k T for k=0,1,2
    # Define B_k = T^T D M_k D T for k=0,1,2
    
    # The uncertainty product bound is sqrt((m_2/m_0) * (n_2/n_0))
    # = sqrt((a^T A_2 a / a^T A_0 a) * (2*pi * a^T B_2 a / 2*pi * a^T B_0 a))
    # = sqrt((a^T A_2 a * a^T B_2 a) / (a^T A_0 a * a^T B_0 a))
    
    # This is not a simple Rayleigh quotient, but we can still optimize it.
    # However, the strategy suggests using a generalized eigenvalue problem.
    
    # Let me compute the matrices.
    
    # Build M_k for k=0,1,2 with indices 1..6 (corresponding to x^2, x^4, ..., x^12)
    # M_k[i,j] = I[k + (i+1) + (j+1)] where i,j are 0-indexed (0->x^2, 1->x^4, etc.)
    # Actually, if p has components indexed 0..5 corresponding to x^2, x^4, ..., x^12,
    # then the exponent for p_i is 2*(i+1), so:
    # m_k = sum_{i=0}^{5} sum_{j=0}^{5} p_i p_j I[k + 2(i+1) + 2(j+1)]
    #     = sum_{i,j} p_i p_j I[k + 2i + 2j + 4]
    
    M = []
    for k in range(3):
        Mk = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                idx = k + 2*(i+1) + 2*(j+1)
                if idx < 25:
                    Mk[i, j] = _I[idx]
        M.append(Mk)
    
    # A_k = T^T M_k T
    A = []
    for k in range(3):
        A.append(T.T @ M[k] @ T)
    
    # B_k = T^T D M_k D T
    B = []
    for k in range(3):
        D_mat = np.diag(D)
        B.append(T.T @ (D_mat @ M[k] @ D_mat) @ T)
    
    return A, B


_A, _B = _build_moment_matrices()


def _compute_bound_from_a(a):
    """
    Compute the uncertainty bound from coefficients a = [a0, a4, a8].
    Uses precomputed matrices A and B.
    """
    a0, a4, a8 = a
    a_vec = np.array([a0, a4, a8])
    
    m0 = a_vec @ _A[0] @ a_vec
    m2 = a_vec @ _A[2] @ a_vec
    n0 = a_vec @ _B[0] @ a_vec
    n2 = a_vec @ _B[2] @ a_vec
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    if m2 <= 0 or n2 <= 0:
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    bound = np.sqrt(vf * vF)
    
    if not np.isfinite(bound):
        return 1e18
    return bound


def _bound_and_grad(a):
    """Compute bound and gradient analytically."""
    a_vec = np.array(a)
    
    m0 = a_vec @ _A[0] @ a_vec
    m2 = a_vec @ _A[2] @ a_vec
    n0 = a_vec @ _B[0] @ a_vec
    n2 = a_vec @ _B[2] @ a_vec
    
    if m0 <= 0 or n0 <= 0 or m2 <= 0 or n2 <= 0:
        return 1e18, np.zeros(3)
    
    vf = m2 / m0
    vF = n2 / n0
    bound = np.sqrt(vf * vF)
    
    # Gradient of bound = 0.5 * (vf*vF)^(-1/2) * (vF * d(vf) + vf * d(vF))
    # d(vf)/da = (dm2/da * m0 - m2 * dm0/da) / m0^2
    # d(vF)/da = (dn2/da * n0 - n2 * dn0/da) / n0^2
    
    dm0 = 2.0 * (_A[0] @ a_vec)
    dm2 = 2.0 * (_A[2] @ a_vec)
    dn0 = 2.0 * (_B[0] @ a_vec)
    dn2 = 2.0 * (_B[2] @ a_vec)
    
    dvf = (dm2 * m0 - m2 * dm0) / (m0 * m0)
    dvF = (dn2 * n0 - n2 * dn0) / (n0 * n0)
    
    grad = 0.5 / bound * (vF * dvf + vf * dvF)
    
    return bound, grad


def _objective(a):
    return _compute_bound_from_a(a)


def _objective_grad(a):
    return _bound_and_grad(a)


def construct():
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    # Phase 1: Broad random sampling
    rng = np.random.default_rng(42)
    
    starts = []
    # Around the seed
    for _ in range(20):
        s = np.array([-1.13, 0.04, 0.000308642]) + rng.normal(0, 0.15, 3)
        starts.append(s)
    # Wider exploration
    for _ in range(40):
        s = np.array([rng.uniform(-5, 3), rng.uniform(-1, 1), rng.uniform(-0.05, 0.05)])
        starts.append(s)
    # Structured grid
    for a0 in np.linspace(-3, 1, 8):
        for a4 in np.linspace(-0.5, 0.5, 7):
            a8 = rng.uniform(-0.01, 0.01)
            starts.append(np.array([a0, a4, a8]))
    
    # Evaluate all starts
    for s in starts:
        if time.time() > _DEADLINE - 50:
            break
        try:
            val = _compute_bound_from_a(s)
            if val < best_val:
                best_val = val
                best_params = s.copy()
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B from best starts
    scored_starts = []
    for s in starts:
        try:
            val = _compute_bound_from_a(s)
            scored_starts.append((val, s))
        except Exception:
            pass
    scored_starts.sort(key=lambda x: x[0])
    
    top_starts = [s for _, s in scored_starts[:15]]
    top_starts.append(best_params.copy())
    
    for s in top_starts:
        if time.time() > _DEADLINE - 35:
            break
        try:
            res = minimize(_objective_grad, s, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead polish
    if time.time() < _DEADLINE - 20:
        try:
            nm_res = minimize(_objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 200000, 'xatol': 1e-18, 'fatol': 1e-20})
            if nm_res.fun < best_val and np.isfinite(nm_res.fun):
                best_val = nm_res.fun
                best_params = nm_res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final L-BFGS-B polish
    if time.time() < _DEADLINE - 10:
        try:
            fin_res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                             jac=True, options={'maxiter': 20000, 'ftol': 1e-17, 'gtol': 1e-16})
            if fin_res.fun < best_val and np.isfinite(fin_res.fun):
                best_val = fin_res.fun
                best_params = fin_res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Perturbation and re-optimization
    if time.time() < _DEADLINE - 5:
        for scale in [1e-2, 1e-3, 1e-4]:
            if time.time() > _DEADLINE - 3:
                break
            for _ in range(10):
                perturbed = best_params + rng.normal(0, scale, 3)
                try:
                    res = minimize(_objective_grad, perturbed, method='L-BFGS-B',
                                 jac=True, options={'maxiter': 10000, 'ftol': 1e-17, 'gtol': 1e-16})
                    if res.fun < best_val and np.isfinite(res.fun):
                        best_val = res.fun
                        best_params = res.x.copy()
                except Exception:
                    pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
