# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, gammaln
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 620

def _build_quadrature():
    """70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    cos_theta = x_gl
    phi = np.linspace(0, 2 * np.pi, 130, endpoint=False)
    dphi = 2.0 * np.pi / 130
    weights_1d = w_gl * dphi
    return cos_theta, phi, weights_1d

def _compute_harmonics_matrix(cos_theta, phi):
    """
    Compute the matrix X where X[i, j] = Y_j(ct_i, phi_j).
    j ranges from 0 to 20, corresponding to m = -10 to 10.
    """
    N_pts_ct = len(cos_theta)
    N_pts_phi = len(phi)
    X = np.zeros((N_pts_ct * N_pts_phi, 21))
    
    ct, ph = np.meshgrid(cos_theta, phi, indexing='ij')
    ct_flat = ct.flatten()
    ph_flat = ph.flatten()
    
    for k, m in enumerate(range(-10, 11)):
        abs_m = abs(m)
        log_N = 0.5 * (np.log(21.0 / (4.0 * np.pi)) + 
                       gammaln(10 - abs_m + 1) - gammaln(10 + abs_m + 1))
        N_m = np.exp(log_N)
        
        P = lpmv(abs_m, 10, ct_flat)
        
        if m == 0:
            Y = N_m * P
        elif m > 0:
            Y = np.sqrt(2.0) * N_m * P * np.cos(m * ph_flat)
        else:
            Y = np.sqrt(2.0) * N_m * P * np.sin(abs_m * ph_flat)
            
        X[:, k] = Y
        
    return X

def _build_spectral_tensors(X, w_flat):
    """
    Precompute the spectral tensors for L4 norm computation.
    T2[a,b] = sum_i w_i X[i,a] X[i,b]  (should be ~I for orthonormal basis)
    T4[a,b,c,d] = sum_i w_i X[i,a] X[i,b] X[i,c] X[i,d]
    
    Then ||f||_4^4 = sum_{a,b,c,d} T4[a,b,c,d] c_a c_b c_c c_d
    and ||f||_2^2 = sum_{a,b} T2[a,b] c_a c_b
    """
    n = 21
    # T2: 21x21
    T2 = X.T @ (w_flat[:, None] * X)
    
    # T4: 21^4 tensor - build via outer products
    # T4[a,b,c,d] = sum_i w_i * X[i,a]*X[i,b]*X[i,c]*X[i,d]
    # Build as: for each i, add w_i * outer(outer(X_i, X_i), outer(X_i, X_i))
    # This is 9100 iterations of 21^4 = ~17M ops each = too slow
    # Alternative: T4 = sum_i w_i * np.einsum('a,b,c,d->abcd', X[i], X[i], X[i], X[i])
    # Better: use the fact that T4[a,b,c,d] = sum_i w_i X[i,a]X[i,b] X[i,c]X[i,d]
    # Let M = X * sqrt(w)[:,None], then T4[a,b,c,d] = sum_i M[i,a]M[i,b] M[i,c]M[i,d]
    # = (M^T M) contracted... no, it's a 4th order tensor
    # Use: T4 = np.einsum('ia,ib,ic,id->abcd', M, M, M, M) where M = X*sqrt(w)
    
    sqrt_w = np.sqrt(w_flat)
    M = X * sqrt_w[:, None]  # 9100 x 21
    
    # T4[a,b,c,d] = sum_i M[i,a] M[i,b] M[i,c] M[i,d]
    # This is 21^4 = 194481 elements, each a sum over 9100 terms
    # Using einsum: ~9100 * 21^4 = ~1.77e9 flops, should be fast in numpy
    T4 = np.einsum('ia,ib,ic,id->abcd', M, M, M, M, optimize=True)
    
    return T2, T4

def _ratio_spectral(c, T2, T4):
    """
    Compute R = ||f||_4 / ||f||_2 using spectral tensors.
    ||f||_2^2 = c^T T2 c
    ||f||_4^4 = T4[a,b,c,d] c_a c_b c_c c_d
    """
    S2 = c @ T2 @ c
    if S2 < 1e-15:
        return 1e10
    
    # S4 = T4[a,b,c,d] c_a c_b c_c c_d
    # Efficient: first compute v[a,b] = sum_cd T4[a,b,c,d] c_c c_d, then S4 = sum_ab v[a,b] c_a c_b
    # Or: reshape T4 to (21*21, 21*21) and do matrix product
    T4_mat = T4.reshape(21*21, 21*21)
    c_mat = np.outer(c, c)  # 21x21, c[a]*c[b]
    c_mat_flat = c_mat.flatten()  # 441
    S4 = c_mat_flat @ (T4_mat @ c_mat_flat)
    
    if S4 < 1e-15:
        return 1e10
    
    N2 = np.sqrt(S2)
    N4 = S4 ** 0.25
    R = N4 / N2
    return R

def _ratio_and_grad_spectral(c, T2, T4):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient using spectral tensors.
    """
    S2 = c @ T2 @ c
    if S2 < 1e-15:
        return 1e10, np.zeros(21)
    
    T4_mat = T4.reshape(21*21, 21*21)
    c_mat = np.outer(c, c)
    c_mat_flat = c_mat.flatten()
    S4 = c_mat_flat @ (T4_mat @ c_mat_flat)
    
    if S4 < 1e-15:
        return 1e10, np.zeros(21)
    
    N2 = np.sqrt(S2)
    N4 = S4 ** 0.25
    R = N4 / N2
    
    # Gradient of S4 with respect to c:
    # S4 = T4[a,b,c,d] c_a c_b c_c c_d
    # dS4/dc_k = 4 * T4[a,b,c,k] c_a c_b c_c (by symmetry)
    # = 4 * sum_{a,b,c} T4[a,b,c,k] c_a c_b c_c
    # Let's compute: v_k = sum_{a,b,c} T4[a,b,c,k] c_a c_b c_c
    # = sum_{a,b} (sum_c T4[a,b,c,k] c_c) c_a c_b
    # = sum_{a,b} (T4[:, :, :, k] @ c) [a,b] * c[a] * c[b]
    # More directly: v_k = np.einsum('abc k, a, b, c -> k', T4, c, c, c)
    # But that's 21^4 * 21 = 4M ops, fine.
    
    # Actually: dS4/dc_k = 4 * np.einsum('abck,a,b,c->k', T4, c, c, c)
    dS4_dc = 4.0 * np.einsum('abck,a,b,c->k', T4, c, c, c, optimize=True)
    
    # dS2/dc = 2 * T2 @ c
    dS2_dc = 2.0 * (T2 @ c)
    
    # dR/dc = R * (dS4/(4*S4) - dS2/(2*S2))
    grad_R = R * (dS4_dc / (4.0 * S4) - dS2_dc / (2.0 * S2))
    
    return R, grad_R

def rudin_problem10() -> np.ndarray:
    """
    Minimize ||f||_4 / ||f||_2 for degree-10 real spherical harmonics.
    Returns 21 coefficients c_m for m = -10..10.
    """
    global _DEADLINE
    
    cos_theta, phi, w_1d = _build_quadrature()
    X = _compute_harmonics_matrix(cos_theta, phi)
    w_flat = np.tile(w_1d, 130)
    
    # Build spectral tensors
    T2, T4 = _build_spectral_tensors(X, w_flat)
    
    best_coeffs = None
    best_val = np.inf
    
    rng = np.random.default_rng(42)
    
    def _objective(c):
        return _ratio_spectral(c, T2, T4)
    
    def _objective_grad(c):
        return _ratio_and_grad_spectral(c, T2, T4)
    
    constraint = {'type': 'eq', 'fun': lambda c: np.dot(c, c) - 1.0, 'jac': lambda c: 2.0 * c}
    
    # Phase 1: Differential Evolution for global exploration
    # The ratio is scale-invariant, so we work on the sphere ||c||=1
    # DE doesn't handle constraints well, so we normalize after each eval
    def _de_objective(c_raw):
        c = c_raw / np.linalg.norm(c_raw)
        return _ratio_spectral(c, T2, T4)
    
    if time.time() < _DEADLINE - 300:
        try:
            bounds = [(-1.0, 1.0)] * 21
            de_result = differential_evolution(
                _de_objective,
                bounds,
                seed=42,
                maxiter=200,
                popsize=20,
                tol=1e-10,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=True,
                init='sobol',
                updating='deferred',
                workers=1
            )
            if de_result.success or de_result.fun < best_val:
                best_coeffs = de_result.x / np.linalg.norm(de_result.x)
                best_val = _ratio_spectral(best_coeffs, T2, T4)
        except Exception:
            pass
    
    # Phase 2: SLSQP local refinement from multiple starts
    starts = []
    
    if best_coeffs is not None:
        starts.append(best_coeffs.copy())
    
    # Key structural starts
    c0 = np.zeros(21); c0[10] = 1.0
    starts.append(c0)
    c10 = np.zeros(21); c10[20] = 1.0
    starts.append(c10)
    c_10 = np.zeros(21); c_10[0] = 1.0
    starts.append(c_10)
    c5 = np.zeros(21); c5[15] = 1.0
    starts.append(c5)
    c_5 = np.zeros(21); c_5[5] = 1.0
    starts.append(c_5)
    c4 = np.zeros(21); c4[14] = 1.0
    starts.append(c4)
    c_4 = np.zeros(21); c_4[6] = 1.0
    starts.append(c_4)
    
    # Random starts
    for _ in range(100):
        c = rng.standard_normal(21)
        c = c / np.linalg.norm(c)
        starts.append(c)
    
    for start in starts:
        if time.time() > _DEADLINE - 60:
            break
            
        try:
            res = minimize(
                lambda c: _ratio_and_grad_spectral(c, T2, T4)[0],
                start,
                jac=lambda c: _ratio_and_grad_spectral(c, T2, T4)[1],
                method='SLSQP',
                constraints=[constraint],
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            
            val = _ratio_spectral(res.x, T2, T4)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Trust-region refinement on the best solution
    if best_coeffs is not None and time.time() < _DEADLINE - 30:
        try:
            res = minimize(
                lambda c: _ratio_and_grad_spectral(c, T2, T4)[0],
                best_coeffs,
                jac=lambda c: _ratio_and_grad_spectral(c, T2, T4)[1],
                method='trust-constr',
                constraints=[{'type': 'eq', 'fun': lambda c: np.dot(c, c) - 1.0,
                              'jac': lambda c: 2.0 * c}],
                options={'maxiter': 5000, 'gtol': 1e-16, 'xtol': 1e-16, 'ctol': 1e-16}
            )
            val = _ratio_spectral(res.x, T2, T4)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Additional SLSQP polish with very tight tolerance
    if best_coeffs is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(
                lambda c: _ratio_and_grad_spectral(c, T2, T4)[0],
                best_coeffs,
                jac=lambda c: _ratio_and_grad_spectral(c, T2, T4)[1],
                method='SLSQP',
                constraints=[constraint],
                options={'maxiter': 10000, 'ftol': 1e-16}
            )
            val = _ratio_spectral(res.x, T2, T4)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: High-precision verification with 200-point Gauss-Legendre
    if best_coeffs is not None and time.time() < _DEADLINE - 10:
        try:
            x_gl_hp, w_gl_hp = np.polynomial.legendre.leggauss(200)
            ct_hp = x_gl_hp
            phi_hp = np.linspace(0, 2 * np.pi, 130, endpoint=False)
            dphi_hp = 2.0 * np.pi / 130
            w_flat_hp = np.tile(w_gl_hp * dphi_hp, 130)
            
            ct_m, ph_m = np.meshgrid(ct_hp, phi_hp, indexing='ij')
            ct_flat_hp = ct_m.flatten()
            ph_flat_hp = ph_m.flatten()
            
            X_hp = np.zeros((len(ct_flat_hp), 21))
            for k, m in enumerate(range(-10, 11)):
                abs_m = abs(m)
                log_N = 0.5 * (np.log(21.0 / (4.0 * np.pi)) +
                               gammaln(10 - abs_m + 1) - gammaln(10 + abs_m + 1))
                N_m = np.exp(log_N)
                P = lpmv(abs_m, 10, ct_flat_hp)
                if m == 0:
                    Y = N_m * P
                elif m > 0:
                    Y = np.sqrt(2.0) * N_m * P * np.cos(m * ph_flat_hp)
                else:
                    Y = np.sqrt(2.0) * N_m * P * np.sin(abs_m * ph_flat_hp)
                X_hp[:, k] = Y
            
            f_hp = X_hp @ best_coeffs
            S2_hp = np.sum(w_flat_hp * f_hp * f_hp)
            S4_hp = np.sum(w_flat_hp * f_hp**4)
            if S2_hp > 1e-10 and S4_hp > 1e-10:
                R_hp = (S4_hp ** 0.25) / np.sqrt(S2_hp)
                best_val = R_hp
        except Exception:
            pass
    
    if best_coeffs is None:
        best_coeffs = np.zeros(21)
        best_coeffs[10] = 1.0

    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs * (0.01 / max_abs)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs * (1e4 / max_abs)
        
    return best_coeffs

# EVOLVE-BLOCK-END
