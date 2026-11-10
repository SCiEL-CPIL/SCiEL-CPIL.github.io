# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Precompute quadrature nodes and weights
_N_GL = 70
_N_PHI = 130
_x_gl, _w_gl = np.polynomial.legendre.leggauss(_N_GL)
_theta_gl = np.arccos(_x_gl)
_phi_grid = 2 * np.pi * np.arange(_N_PHI) / _N_PHI
_dphi = 2 * np.pi / _N_PHI

# Precompute spherical harmonic values at all quadrature points
# Shape: (21, 70, 130) for m = -10..10
_k = 10
_N_COEFFS = 2 * _k + 1
_Y_precomputed = np.zeros((_N_COEFFS, _N_GL, _N_PHI))

for idx, m in enumerate(range(-_k, _k + 1)):
    abs_m = abs(m)
    P = lpmv(abs_m, _k, _x_gl)  # shape (70,)
    
    n_fact = factorial(_k - abs_m, exact=True)
    d_fact = factorial(_k + abs_m, exact=True)
    N_m = np.sqrt((2 * _k + 1) / (4 * np.pi) * (n_fact / d_fact))
    
    if m == 0:
        _Y_precomputed[idx] = N_m * P[:, None]
    elif m > 0:
        _Y_precomputed[idx] = np.sqrt(2) * N_m * P[:, None] * np.cos(m * _phi_grid)[None, :]
    else:
        _Y_precomputed[idx] = np.sqrt(2) * N_m * P[:, None] * np.sin(abs_m * _phi_grid)[None, :]

# Precompute weights for norm calculations
# Weight matrix: (70, 130) where each element is w_gl[i] * dphi
_W_matrix = _w_gl[:, None] * _dphi

def _compute_ratio(coeffs):
    """
    Compute R = ||f||_4 / ||f||_2 using precomputed quadrature.
    coeffs: length 21 vector c_m for m=-10..10
    Returns: ratio R (lower is better)
    """
    if np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs)):
        return np.inf
    
    # f = sum_m c_m * Y_m, computed via matrix multiplication
    # coeffs shape (21,), Y shape (21, 70, 130)
    # f shape (70, 130)
    f_val = np.tensordot(coeffs, _Y_precomputed, axes=(0, 0))
    
    f2 = f_val ** 2
    f4 = f_val ** 4
    
    norm2_sq = np.sum(f2 * _W_matrix)
    norm4_4 = np.sum(f4 * _W_matrix)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    
    return norm4 / norm2


def _compute_ratio_and_grad(coeffs):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient with respect to coeffs.
    Uses the chain rule:
      R = N4 / N2 where N4 = (sum w f^4)^(1/4), N2 = (sum w f^2)^(1/2)
      dR/dc = (1/N2) * dN4/dc - (N4/N2^2) * dN2/dc
      dN4/dc = (1/4) N4^(-3) * sum(w * 4 f^3 * Y) = N4^(-3) * sum(w * f^3 * Y)
      dN2/dc = (1/2) N2^(-1) * sum(w * 2 f * Y) = N2^(-1) * sum(w * f * Y)
    """
    if np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs)):
        return np.inf, np.zeros_like(coeffs)
    
    f_val = np.tensordot(coeffs, _Y_precomputed, axes=(0, 0))
    
    f2 = f_val ** 2
    f4 = f_val ** 4
    
    norm2_sq = np.sum(f2 * _W_matrix)
    norm4_4 = np.sum(f4 * _W_matrix)
    
    if norm2_sq <= 0:
        return np.inf, np.zeros_like(coeffs)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    
    R = norm4 / norm2
    
    # Gradient computation
    # dN4/dc_j = sum_i w_i * f_i^3 * Y_ij (for each j)
    # We can compute this as: Y.T @ (w * f^3) but shapes need care
    # _Y_precomputed has shape (21, 70, 130)
    # f_val has shape (70, 130)
    # w_matrix has shape (70, 130)
    
    # dN4_dc[j] = sum over all points of w * f^3 * Y_j
    # = einsum('ij,ijk->k', f_val**3 * _W_matrix, _Y_precomputed)
    # But let's be more explicit:
    
    # For each m (index j in 0..20):
    # dN4_dc[j] = sum_{i,k} w[i,k] * f[i,k]^3 * Y[j,i,k]
    
    weighted_f3 = (f_val ** 3) * _W_matrix  # shape (70, 130)
    dN4_dc = np.tensordot(_Y_precomputed, weighted_f3, axes=([1, 2], [0, 1]))  # shape (21,)
    
    weighted_f = f_val * _W_matrix  # shape (70, 130)
    dN2_dc = np.tensordot(_Y_precomputed, weighted_f, axes=([1, 2], [0, 1]))  # shape (21,)
    
    # dR/dc = (1/N2) * (1/N4^3) * dN4_dc - (N4/N2^2) * (1/N2) * dN2_dc
    #       = dN4_dc / (N2 * N4^3) - N4 * dN2_dc / (N2^3)
    
    grad = dN4_dc / (norm2 * norm4**3) - norm4 * dN2_dc / (norm2**3)
    
    return R, grad


def _compute_ratio_normalized(coeffs):
    """
    Normalized ratio computation that handles scale invariance better.
    We normalize coeffs to unit L2 norm before computing ratio.
    """
    norm_c = np.linalg.norm(coeffs)
    if norm_c < 1e-15:
        return np.inf
    c_norm = coeffs / norm_c
    return _compute_ratio(c_norm)


def rudin_problem10() -> np.ndarray:
    """
    Find the real coefficients (c_m) for m = -10..10 of a real degree-10 spherical harmonic
    that MINIMIZE the norm ratio ||f||_4 / ||f||_2.
    """
    start_time = time.time()
    deadline = 600  # seconds, with safety margin
    
    best_coeffs = None
    best_ratio = np.inf
    
    def _update_best(c, r):
        nonlocal best_coeffs, best_ratio
        if r < best_ratio and np.all(np.isfinite(c)):
            best_ratio = r
            best_coeffs = c.copy()
    
    # Initial guesses
    n_coeffs = _N_COEFFS
    
    # 1. All ones baseline
    c0 = np.ones(n_coeffs)
    r0 = _compute_ratio(c0)
    _update_best(c0, r0)
    
    # 2. Single mode candidates (each m individually)
    for m_idx in range(n_coeffs):
        c_single = np.zeros(n_coeffs)
        c_single[m_idx] = 1.0
        r_single = _compute_ratio(c_single)
        _update_best(c_single, r_single)
    
    # 3. Pairs of modes
    for i in range(n_coeffs):
        for j in range(i + 1, n_coeffs):
            c_pair = np.zeros(n_coeffs)
            c_pair[i] = 1.0
            c_pair[j] = 1.0
            r_pair = _compute_ratio(c_pair)
            _update_best(c_pair, r_pair)
    
    # 4. Sparse patterns
    rng = np.random.default_rng(42)
    for _ in range(50):
        if time.time() > deadline - 60:
            break
        n_nonzero = rng.integers(2, 8)
        indices = rng.choice(n_coeffs, size=n_nonzero, replace=False)
        c_sparse = np.zeros(n_coeffs)
        c_sparse[indices] = rng.normal(size=n_nonzero)
        r_sparse = _compute_ratio(c_sparse)
        _update_best(c_sparse, r_sparse)
    
    # 5. Structured patterns based on symmetry
    # Even m only
    c_even = np.zeros(n_coeffs)
    c_even[::2] = 1.0  # m = -10, -8, ..., 8, 10
    _update_best(c_even, _compute_ratio(c_even))
    
    # Odd m only
    c_odd = np.zeros(n_coeffs)
    c_odd[1::2] = 1.0  # m = -9, -7, ..., 7, 9
    _update_best(c_odd, _compute_ratio(c_odd))
    
    # Positive m only
    c_pos = np.zeros(n_coeffs)
    c_pos[10:] = 1.0  # m = 0..10
    _update_best(c_pos, _compute_ratio(c_pos))
    
    # Negative m only
    c_neg = np.zeros(n_coeffs)
    c_neg[:11] = 1.0  # m = -10..0
    _update_best(c_neg, _compute_ratio(c_neg))
    
    # 6. Collect candidates for optimization
    candidates = []
    if best_coeffs is not None:
        candidates.append(best_coeffs.copy())
    
    # Add diverse random normalized vectors
    for _ in range(30):
        if time.time() > deadline - 30:
            break
        c_rand = rng.normal(size=n_coeffs)
        c_rand /= np.linalg.norm(c_rand)
        candidates.append(c_rand)
    
    # Add mixed-mode structured candidates
    # m=0 with high-m modes
    for high_m in [5, 8, 10]:
        c_mixed = np.zeros(n_coeffs)
        c_mixed[10] = 1.0  # m=0
        c_mixed[10 + high_m] = rng.normal()  # m=+high_m
        c_mixed[10 - high_m] = rng.normal()  # m=-high_m
        c_mixed /= np.linalg.norm(c_mixed)
        candidates.append(c_mixed)
    
    # Sparse with specific patterns
    for _ in range(10):
        if time.time() > deadline - 30:
            break
        c_sparse = np.zeros(n_coeffs)
        n_nonzero = rng.integers(3, 7)
        indices = rng.choice(n_coeffs, size=n_nonzero, replace=False)
        c_sparse[indices] = rng.normal(size=n_nonzero)
        c_sparse /= np.linalg.norm(c_sparse)
        candidates.append(c_sparse)
    
    # Sort candidates by their ratio and take top few for optimization
    scored_candidates = []
    for c in candidates:
        r = _compute_ratio(c)
        scored_candidates.append((r, c))
    scored_candidates.sort(key=lambda x: x[0])
    
    # Define objective with gradient for SLSQP
    def _obj_with_grad(c):
        R, grad = _compute_ratio_and_grad(c)
        if np.isinf(R):
            return 1e10, np.zeros_like(c)
        return R, grad
    
    # Equality constraint: ||c||^2 = 1
    def _norm_constraint(c):
        return np.dot(c, c) - 1.0
    
    def _norm_constraint_grad(c):
        return 2.0 * c
    
    constraints = [{'type': 'eq', 'fun': _norm_constraint, 'jac': _norm_constraint_grad}]
    
    # Optimize from top 15 candidates using SLSQP with analytic gradient
    for i, (r_init, c_init) in enumerate(scored_candidates[:15]):
        if time.time() > deadline - 30:
            break
        
        try:
            res = minimize(
                _obj_with_grad,
                c_init,
                method='SLSQP',
                jac=True,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            c_opt = res.x
            # Normalize to ensure unit norm
            c_opt = c_opt / np.linalg.norm(c_opt)
            r_opt = _compute_ratio(c_opt)
            if np.isfinite(r_opt):
                _update_best(c_opt, r_opt)
        except Exception:
            pass
    
    # 7. Additional random restarts with SLSQP if time permits
    n_extra_restarts = 20
    for i in range(n_extra_restarts):
        if time.time() > deadline - 15:
            break
        
        c_start = rng.normal(size=n_coeffs)
        norm_c = np.linalg.norm(c_start)
        if norm_c < 1e-15:
            continue
        c_start_norm = c_start / norm_c
        
        try:
            res = minimize(
                _obj_with_grad,
                c_start_norm,
                method='SLSQP',
                jac=True,
                constraints=constraints,
                options={'maxiter': 150, 'ftol': 1e-14}
            )
            if res.fun < best_ratio:
                c_opt = res.x
                c_opt = c_opt / np.linalg.norm(c_opt)
                r_opt = _compute_ratio(c_opt)
                if np.isfinite(r_opt):
                    _update_best(c_opt, r_opt)
        except Exception:
            pass
    
    # 8. Final polish: refine best solution with tighter tolerances
    if best_coeffs is not None and time.time() < deadline - 10:
        try:
            res = minimize(
                _obj_with_grad,
                best_coeffs / np.linalg.norm(best_coeffs),
                method='SLSQP',
                jac=True,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-16}
            )
            c_opt = res.x
            c_opt = c_opt / np.linalg.norm(c_opt)
            r_opt = _compute_ratio(c_opt)
            if np.isfinite(r_opt):
                _update_best(c_opt, r_opt)
        except Exception:
            pass
    
    # Final validation
    if best_coeffs is None or np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        best_coeffs = np.ones(n_coeffs)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs == 0:
            best_coeffs = np.ones(n_coeffs)
        else:
            best_coeffs = best_coeffs / max_abs * 1.0
            if np.max(np.abs(best_coeffs)) < 1e-3:
                best_coeffs = np.ones(n_coeffs)
    
    return best_coeffs.astype(np.float64)


# EVOLVE-BLOCK-END
