"""Analytic uncertainty bound with exact gradient + multi-method optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = _T0 + 195  # 15s safety margin before 210s kill

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
_I_TABLE = np.zeros(14)
def _precompute_I():
    sp = math.sqrt(math.pi)
    I = np.zeros(14)
    for n in range(14):
        # (2n)! / (4^n n!) * sqrt(pi)
        num = 1.0
        for m in range(1, 2*n + 1):
            num *= m
        den = (4.0 ** n)
        for m in range(1, n + 1):
            den *= m
        I[n] = num / den * sp
    return I
_I_TABLE = _precompute_I()

# Precompute the coefficient matrix for p in terms of (a0, a4, a8, a12)
# p0 = 1*a0 + 3*a4 + 105*a8 + 945*a12
# p1 = 0*a0 + (-6)*a4 + (-420)*a8 + (-7560)*a12
# p2 = 0*a0 + 1*a4 + 210*a8 + 13860*a12
# p3 = 0*a0 + 0*a4 + (-28)*a8 + (-6930)*a12
# p4 = 0*a0 + 0*a4 + 1*a8 + 1155*a12
# p5 = 0*a0 + 0*a4 + 0*a8 + (-66)*a12
# p6 = 0*a0 + 0*a4 + 0*a8 + 1*a12
_P_COEFFS = np.array([
    [1.0, 3.0, 105.0, 945.0],
    [0.0, -6.0, -420.0, -7560.0],
    [0.0, 1.0, 210.0, 13860.0],
    [0.0, 0.0, -28.0, -6930.0],
    [0.0, 0.0, 1.0, 1155.0],
    [0.0, 0.0, 0.0, -66.0],
    [0.0, 0.0, 0.0, 1.0],
])

# a12 = -(a0 + 3*a4 + 105*a8) / 945
_A12_COEFFS = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

# Precompute M matrices for efficient moment computation
# m_k = sum_{i,j} p_i p_j I[k+i+j]
# We can write this as: for each k, m_k = p^T M_k p where (M_k)_{ij} = I[k+i+j]
_M_MATRICES = []
for k in range(3):
    Mk = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            idx = k + i + j
            if idx < 14:
                Mk[i, j] = _I_TABLE[idx]
    _M_MATRICES.append(Mk)

# Q coefficients: q_k = p_k * (-1)^k
_SIGNS = np.array([(-1.0)**k for k in range(7)])

def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = _A12_COEFFS[0]*a0 + _A12_COEFFS[1]*a4 + _A12_COEFFS[2]*a8
    
    # Compute p vector
    p = _P_COEFFS[0]*a0 + _P_COEFFS[1]*a4 + _P_COEFFS[2]*a8 + _P_COEFFS[3]*a12
    
    # Compute q vector (Fourier domain)
    q = p * _SIGNS
    
    # Compute moments using precomputed matrices
    m0 = p @ _M_MATRICES[0] @ p
    m2 = p @ _M_MATRICES[1] @ p
    n0 = 2.0*math.pi * (q @ _M_MATRICES[0] @ q)
    n2 = 2.0*math.pi * (q @ _M_MATRICES[1] @ q)
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0 or not np.isfinite(vf) or not np.isfinite(vF):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = math.sqrt(vf * vF)
    
    # Compute gradient analytically
    # d(bound)/d(a_c) = (1/(2*bound)) * (vF * d(vf)/d(a_c) + vf * d(vF)/d(a_c))
    # d(vf)/d(a_c) = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # d(vF)/d(a_c) = (dn2/da_c * n0 - n2 * dn0/da_c) / n0^2
    
    # dp/da_c for c = 0,1,2 (a0, a4, a8)
    # dp/da_c = P_COEFFS[:, c] + P_COEFFS[:, 3] * A12_COEFFS[c]
    DP = np.zeros((7, 3))
    for c in range(3):
        DP[:, c] = _P_COEFFS[:, c] + _P_COEFFS[:, 3] * _A12_COEFFS[c]
    
    grad = np.zeros(3)
    for c in range(3):
        dp = DP[:, c]
        dq = dp * _SIGNS
        
        # dm_k/da_c = 2 * dp^T M_k p
        dm0 = 2.0 * (dp @ _M_MATRICES[0] @ p)
        dm2 = 2.0 * (dp @ _M_MATRICES[1] @ p)
        
        # dn_k/da_c = 2 * 2*pi * dq^T M_k q
        dn0 = 2.0 * 2.0*math.pi * (dq @ _M_MATRICES[0] @ q)
        dn2 = 2.0 * 2.0*math.pi * (dq @ _M_MATRICES[1] @ q)
        
        dvm2 = (dm2*m0 - m2*dm0) / (m0*m0)
        dvn2 = (dn2*n0 - n2*dn0) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _eval_grid_2d(a0_vals, a4_vals, a8_val):
    """
    Vectorized evaluation of the objective on a 2D grid over (a0, a4) with fixed a8.
    Returns array of bound values with shape (len(a0_vals), len(a4_vals)).
    """
    n0_pts = len(a0_vals)
    n4_pts = len(a4_vals)
    
    # Build parameter matrix: (n0_pts * n4_pts, 3)
    A0, A4 = np.meshgrid(a0_vals, a4_vals, indexing='ij')
    A8 = np.full_like(A0, a8_val)
    params_flat = np.column_stack([A0.ravel(), A4.ravel(), A8.ravel()])
    
    # Compute a12 for all points
    a12_all = params_flat @ _A12_COEFFS  # shape (N,)
    
    # Compute p vectors: (N, 7)
    P_all = params_flat @ _P_COEFFS.T + np.outer(a12_all, _P_COEFFS[3])
    
    # Compute q vectors
    Q_all = P_all * _SIGNS
    
    # Compute moments: m_k = sum_i,j p_i p_j M_k[i,j]
    # For each point: m0 = p^T M0 p
    M0 = _M_MATRICES[0]
    M1 = _M_MATRICES[1]
    
    # m0[i] = P_all[i] @ M0 @ P_all[i]
    m0_all = np.einsum('ij,ij,jk,ik->i', P_all, P_all, M0, P_all)
    m2_all = np.einsum('ij,ij,jk,ik->i', P_all, P_all, M1, P_all)
    n0_all = 2.0*math.pi * np.einsum('ij,ij,jk,ik->i', Q_all, Q_all, M0, Q_all)
    n2_all = 2.0*math.pi * np.einsum('ij,ij,jk,ik->i', Q_all, Q_all, M1, Q_all)
    
    # Compute bounds
    valid = (m0_all > 0) & (n0_all > 0) & np.isfinite(m0_all) & np.isfinite(n0_all)
    vf = np.where(valid, m2_all / m0_all, 1e18)
    vF = np.where(valid, n2_all / n0_all, 1e18)
    
    valid2 = valid & (vf > 0) & (vF > 0) & np.isfinite(vf) & np.isfinite(vF)
    bounds = np.where(valid2, np.sqrt(vf * vF), 1e18)
    
    return bounds.reshape(n0_pts, n4_pts)


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    best_val = np.inf
    best_params = None
    
    # Phase 1: Coarser but faster grid search to identify promising regions
    # Use fewer points per dimension but more a8 values
    a8_candidates = [0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0008, 0.001, 0.002, 0.005]
    
    for a8_val in a8_candidates:
        if time.time() > _DEADLINE - 80:
            break
        # Coarser grid: 40x40 = 1600 evals per a8, fast
        a0_grid = np.linspace(-3.0, 1.0, 40)
        a4_grid = np.linspace(-0.2, 0.3, 40)
        
        try:
            bounds_2d = _eval_grid_2d(a0_grid, a4_grid, a8_val)
            idx_min = np.unravel_index(np.argmin(bounds_2d), bounds_2d.shape)
            if bounds_2d[idx_min] < best_val and np.isfinite(bounds_2d[idx_min]):
                best_val = bounds_2d[idx_min]
                best_params = np.array([a0_grid[idx_min[0]], a4_grid[idx_min[1]], a8_val])
        except Exception:
            pass
    
    # Phase 2: Adaptive refinement around grid minimum
    if best_params is not None and time.time() < _DEADLINE - 60:
        a0_c, a4_c, a8_c = best_params
        # Refine with finer grid around current best
        for scale in [0.1, 0.01, 0.001, 0.0001]:
            if time.time() > _DEADLINE - 45:
                break
            a0_fine = np.linspace(a0_c - 0.5*scale, a0_c + 0.5*scale, 30)
            a4_fine = np.linspace(a4_c - 0.5*scale, a4_c + 0.5*scale, 30)
            
            try:
                bounds_fine = _eval_grid_2d(a0_fine, a4_fine, a8_c)
                idx_min = np.unravel_index(np.argmin(bounds_fine), bounds_fine.shape)
                if bounds_fine[idx_min] < best_val and np.isfinite(bounds_fine[idx_min]):
                    best_val = bounds_fine[idx_min]
                    best_params = np.array([a0_fine[idx_min[0]], a4_fine[idx_min[1]], a8_c])
            except Exception:
                pass
    
    # Phase 3: Multi-start optimization with diverse methods
    if best_params is not None and time.time() < _DEADLINE - 30:
        # Generate diverse starting points
        starts = [best_params.copy()]
        rng = np.random.default_rng(42)
        
        # Add points from different scales
        for sigma in [0.01, 0.05, 0.1, 0.2, 0.5]:
            for _ in range(5):
                starts.append(best_params + rng.normal(0, sigma, 3))
        
        # Add some fixed candidate points based on known good regions
        fixed_starts = [
            [-1.13, 0.04, 0.0003],
            [-1.0, 0.05, 0.001],
            [-1.2, 0.03, 0.0005],
            [-0.9, 0.06, 0.0002],
            [-1.1, 0.02, 0.0008],
            [-1.15, 0.045, 0.0004],
            [-1.05, 0.035, 0.0006],
            [-1.25, 0.055, 0.0001],
        ]
        for fs in fixed_starts:
            starts.append(np.array(fs))
        
        # Run L-BFGS-B from each start
        for start in starts:
            if time.time() > _DEADLINE - 20:
                break
            try:
                res = minimize(_objective_grad, start, method='L-BFGS-B',
                              jac=True, options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-15})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    
    # Phase 4: Trust-region method (often better for smooth problems)
    if best_params is not None and time.time() < _DEADLINE - 15:
        tr_starts = [best_params.copy()]
        rng2 = np.random.default_rng(99)
        for i in range(5):
            tr_starts.append(best_params + rng2.normal(0, 0.02, 3))
        
        for start in tr_starts:
            if time.time() > _DEADLINE - 10:
                break
            try:
                res = minimize(_objective_grad, start, method='trust-ncg',
                              jac=True, options={'maxiter': 5000, 'gtol': 1e-12, 'xtol': 1e-14})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    
    # Phase 5: SLSQP polish
    if best_params is not None and time.time() < _DEADLINE - 8:
        try:
            res = minimize(_objective_grad, best_params, method='SLSQP',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 6: High-precision final polish with multiple methods
    if best_params is not None and time.time() < _DEADLINE - 3:
        # Final L-BFGS-B with very tight tolerances
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 50000, 'ftol': 1e-18, 'gtol': 1e-17})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        # One more trust-region pass
        if time.time() < _DEADLINE - 2:
            try:
                res = minimize(_objective_grad, best_params, method='trust-ncg',
                              jac=True, options={'maxiter': 10000, 'gtol': 1e-14, 'xtol': 1e-16})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
