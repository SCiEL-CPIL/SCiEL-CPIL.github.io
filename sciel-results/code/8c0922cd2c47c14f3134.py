"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization + generalized eigenvalue polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200

# Precompute I_n = integral x^{2n} e^{-x^2} dx using recurrence
_I_n = np.zeros(28)  # need up to n=12 for products of degree-12 polynomials
_I_n[0] = np.sqrt(np.pi)
for _n in range(1, 28):
    _I_n[_n] = (2*_n - 1) / 2.0 * _I_n[_n-1]

# Precompute the polynomial coefficient matrix for P(x) given (a0, a4, a8, a12)
# P(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
# H0(x) = 1
# H4(x) = x^4 - 6x^2 + 3
# H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12(x) = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
#
# P(x) = sum_{k=0}^{6} p_k x^{2k}
# where p = M @ [a0, a4, a8, a12]
#
# p0 = 1*a0 + 3*a4 + 105*a8 + 945*a12
# p1 = 0*a0 + (-6)*a4 + (-420)*a8 + (-7560)*a12
# p2 = 0*a0 + 1*a4 + 210*a8 + 13860*a12
# p3 = 0*a0 + 0*a4 + (-28)*a8 + (-6930)*a12
# p4 = 0*a0 + 0*a4 + 1*a8 + 1155*a12
# p5 = 0*a0 + 0*a4 + 0*a8 + (-66)*a12
# p6 = 0*a0 + 0*a4 + 0*a8 + 1*a12

_M = np.array([
    [1.0,   3.0,   105.0,   945.0],
    [0.0,  -6.0,  -420.0, -7560.0],
    [0.0,   1.0,   210.0, 13860.0],
    [0.0,   0.0,  -28.0, -6930.0],
    [0.0,   0.0,    1.0,  1155.0],
    [0.0,   0.0,    0.0,  -66.0],
    [0.0,   0.0,    0.0,    1.0]
], dtype=np.float64)

# Sign vector for Fourier transform: q_k = p_k * (-1)^k
_sign_vec = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])


def _compute_moments(p):
    """Compute m0, m2, m4 from polynomial coefficients p (length 7)."""
    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        if pi == 0.0:
            continue
        for j in range(7):
            pp = pi * p[j]
            if pp == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 28:
                    m[k] += pp * _I_n[idx]
    return m


def _compute_fmoments(q):
    """Compute n0, n2, n4 from Fourier-space polynomial coefficients q (length 7)."""
    n = np.zeros(3)
    for i in range(7):
        qi = q[i]
        if qi == 0.0:
            continue
        for j in range(7):
            qq = qi * q[j]
            if qq == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 28:
                    n[k] += 2.0 * np.pi * qq * _I_n[idx]
    return n


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    coeffs4 = np.array([a0, a4, a8, a12])
    p = _M @ coeffs4
    
    m = _compute_moments(p)
    q = p * _sign_vec
    n = _compute_fmoments(q)
    
    m0, m2, m4 = m[0], m[1], m[2]
    n0, n2, n4 = n[0], n[1], n[2]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Compute gradient analytically
    # d(bound)/d(a_c) = (1/(2*bound)) * (vF * d(vf)/d(a_c) + vf * d(vF)/d(a_c))
    # d(vf)/d(a_c) = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # d(vF)/d(a_c) = (dn2/da_c * n0 - n2 * dn0/da_c) / n0^2
    
    # da12/da0 = -1/945, da12/da4 = -3/945, da12/da8 = -105/945
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    grad = np.zeros(3)
    for c in range(3):
        # dp_k/da_c = (column c of M) + (da12/da_c) * (column 3 of M)
        dp = _M[:, c] + da12[c] * _M[:, 3]
        
        # dm_k/da_c = 2*sum_i dp_i * sum_j p_j I[k+i+j]
        dm = np.zeros(3)
        for i in range(7):
            dpi = dp[i]
            if dpi == 0.0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 28:
                        s += p[j] * _I_n[idx]
                dm[k] += 2.0 * dpi * s
        
        dq = dp * _sign_vec
        
        dn = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if dqi == 0.0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 28:
                        s += q[j] * _I_n[idx]
                dn[k] += 2.0 * 2.0 * np.pi * dqi * s
        
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def _eigenvalue_polish(params):
    """
    Use the generalized eigenvalue problem to find stationary points of the
    uncertainty product in the 3D parameter space (with a12 determined by constraint).
    
    The variance ratio vf = m2/m0 where m_k = sum_{i,j} p_i p_j I[k+i+j].
    Similarly vF = n2/n0.
    
    We can write vf = (p^T A p) / (p^T B p) where A and B are 7x7 matrices:
    A[i,j] = I[1+i+j] (for m2)
    B[i,j] = I[0+i+j] (for m0)
    
    Similarly for vF with Q matrices.
    
    The product vf*vF is not a simple ratio of quadratic forms in p alone,
    but we can set up a 4x4 problem in the 4D coefficient space (a0,a4,a8,a12)
    with the constraint a0 + 3a4 + 105a8 + 945a12 = 0.
    
    Actually, let's use a different approach: parameterize the 3 free parameters
    and use the eigendecomposition of the Hessian at the current point to find
    a better direction, or simply use the generalized eigenvalue problem on the
    full 4D space with the constraint incorporated.
    
    For now, let's try a simpler approach: use scipy's eig to find stationary
    points of vf and vF separately, then combine.
    
    Actually, the most direct approach: the function bound^2 = vf * vF where
    vf = p^T A p / p^T B p and vF = q^T C q / q^T D q.
    
    Since q = S p where S = diag(sign_vec), and the constraint is linear in p,
    we can set up a 7x7 generalized eigenvalue problem for vf:
    A p = lambda B p
    and for vF:
    C' p = lambda D' p where C' = S C S, D' = S D S.
    
    The stationary points of the product are not simply the eigenvectors of either,
    but we can use the eigenvectors as starting points for local optimization.
    """
    # Build matrices A, B for vf
    A = np.zeros((7, 7))
    B = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            A[i, j] = _I_n[1 + i + j] if 1 + i + j < 28 else 0.0
            B[i, j] = _I_n[0 + i + j] if 0 + i + j < 28 else 0.0
    
    # For vF: q = S p, so n2 = q^T C q = p^T (S C S) p
    C = np.zeros((7, 7))
    D = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            C[i, j] = _I_n[1 + i + j] if 1 + i + j < 28 else 0.0
            D[i, j] = _I_n[0 + i + j] if 0 + i + j < 28 else 0.0
    
    C_p = C * np.outer(_sign_vec, _sign_vec)
    D_p = D * np.outer(_sign_vec, _sign_vec)
    
    # Solve generalized eigenvalue problems
    # For vf: A p = lambda B p
    try:
        from scipy.linalg import eig
        # Make B symmetric positive definite (it should be)
        # Use scipy.linalg.eig for generalized problem
        eigvals_vf, eigvecs_vf = eig(A, B)
        eigvals_vf = np.real(eigvals_vf)
        eigvecs_vf = np.real(eigvecs_vf)
    except Exception:
        return None
    
    try:
        eigvals_vF, eigvecs_vF = eig(C_p, D_p)
        eigvals_vF = np.real(eigvals_vF)
        eigvecs_vF = np.real(eigvecs_vF)
    except Exception:
        return None
    
    # Filter for positive eigenvalues
    valid_vf = np.where(eigvals_vf > 1e-10)[0]
    valid_vF = np.where(eigvals_vF > 1e-10)[0]
    
    if len(valid_vf) == 0 or len(valid_vF) == 0:
        return None
    
    # For each pair of eigenvectors, project onto the constraint space
    # and evaluate the bound
    best_val = np.inf
    best_params = None
    
    constraint = np.array([1.0, 3.0, 105.0, 945.0])
    # p = M @ coeffs4, constraint is constraint @ coeffs4 = 0
    
    # The constraint in terms of p: constraint @ inv(M) @ p = 0
    # But M is 7x4, so we can't invert. Instead, work in coeffs4 space.
    # coeffs4 = [a0, a4, a8, a12], constraint: a0 + 3a4 + 105a8 + 945a12 = 0
    # Free params: a0, a4, a8 (a12 determined)
    
    # For an eigenvector p, we need to find coeffs4 such that M @ coeffs4 ≈ p
    # and constraint @ coeffs4 = 0.
    # This is an underdetermined system (7 equations, 4 unknowns) so we use least squares.
    
    # Actually, let's just use the eigenvectors as hints and do a local search
    # from points near the eigenvector directions.
    
    # Project eigenvectors to the 3D free parameter space
    # coeffs4 = [a0, a4, a8, -(a0+3a4+105a8)/945]
    # p = M @ coeffs4
    
    # For a target p, solve M @ coeffs4 = p in least squares sense
    # with the constraint automatically satisfied by parameterization.
    
    # M_free = M[:, 0:3] + (constraint contribution from a12)
    # Actually, p = M[:, 0:3] @ [a0, a4, a8] + M[:, 3] @ a12
    # where a12 = -(a0 + 3a4 + 105a8)/945
    # So p = (M[:, 0:3] - M[:, 3] * [1, 3, 105]/945) @ [a0, a4, a8]
    
    M_free = _M[:, 0:3] - _M[:, 3] * np.array([1.0, 3.0, 105.0]) / 945.0
    
    # For each eigenvector, project to 3D space
    for idx in valid_vf:
        p_target = eigvecs_vf[:, idx]
        # Normalize
        p_target = p_target / np.linalg.norm(p_target)
        # Project: solve M_free @ a = p_target in least squares
        try:
            a_est, _, _, _ = np.linalg.lstsq(M_free, p_target, rcond=None)
        except Exception:
            continue
        
        # Evaluate
        try:
            val, _ = _bound_and_grad(a_est)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_params = a_est.copy()
        except Exception:
            pass
    
    for idx in valid_vF:
        p_target = eigvecs_vF[:, idx]
        p_target = p_target / np.linalg.norm(p_target)
        try:
            a_est, _, _, _ = np.linalg.lstsq(M_free, p_target, rcond=None)
        except Exception:
            continue
        
        try:
            val, _ = _bound_and_grad(a_est)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_params = a_est.copy()
        except Exception:
            pass
    
    if best_params is not None:
        return best_params
    return None


def construct():
    # Multiple starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
    ]
    
    rng = np.random.default_rng(123)
    for i in range(20):
        starts.append(rng.normal(0, 0.5, 3))
    
    # Add eigenvalue-based starting points
    try:
        eig_starts = _eigenvalue_polish(np.array([0.0, 0.0, 0.0]))
        if eig_starts is not None:
            starts.append(eig_starts)
    except Exception:
        pass
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 25:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish with Nelder-Mead from best
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish with L-BFGS-B
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Additional eigenvalue polish from the best point
    if best_params is not None and time.time() < _DEADLINE - 3:
        try:
            eig_result = _eigenvalue_polish(best_params)
            if eig_result is not None:
                val, _ = _bound_and_grad(eig_result)
                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best_params = eig_result.copy()
                    # Quick local polish
                    if time.time() < _DEADLINE - 1:
                        res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                                      jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
                        if res.fun < best_val and np.isfinite(res.fun):
                            best_val = res.fun
                            best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
