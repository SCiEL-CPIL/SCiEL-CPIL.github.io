# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize, differential_evolution

K = 10
N_THETA = 70
N_PHI = 130

# Precompute quadrature weights and nodes
_x, _w = np.polynomial.legendre.leggauss(N_THETA)
_cos_theta = _x
_theta_weights = _w * 0.5

# phi grid
_phi = np.arange(N_PHI) * (2 * np.pi / N_PHI)

# Precompute associated Legendre functions P_l^m(cos theta) with Condon-Shortley phase
_P = {}
for m in range(K + 1):
    _P[m] = lpmv(m, K, _cos_theta)

# Normalization constants N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
def _norm_const(m):
    m = abs(m)
    val = 21.0 / (4.0 * np.pi) * float(factorial(10 - m) / factorial(10 + m))
    return np.sqrt(val)

# Precompute Y_m values on the grid
_basis = np.zeros((2 * K + 1, N_THETA, N_PHI), dtype=np.float64)
for idx, m in enumerate(range(-K, K + 1)):
    am = abs(m)
    norm = _norm_const(am)
    Pm = _P[am]
    if m == 0:
        _basis[idx, :, :] = norm * Pm[:, None]
    elif m > 0:
        cos_term = np.cos(m * _phi)
        _basis[idx, :, :] = np.sqrt(2.0) * norm * Pm[:, None] * cos_term[None, :]
    else:
        sin_term = np.sin(am * _phi)
        _basis[idx, :, :] = np.sqrt(2.0) * norm * Pm[:, None] * sin_term[None, :]

# Quadrature weights on the sphere
_w_phi = 2.0 * np.pi / N_PHI
_sphere_weights = np.outer(_theta_weights, np.full(N_PHI, _w_phi))
_sphere_weights_3d = _sphere_weights[None, :, :]

def _compute_R(coeffs):
    """Compute ||f||_4 / ||f||_2 for given coefficients."""
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.shape != (2 * K + 1,):
        return np.inf
    if not np.all(np.isfinite(coeffs)):
        return np.inf
    max_abs = np.max(np.abs(coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        return np.inf
    
    # f = sum c_m Y_m
    f = np.tensordot(coeffs, _basis, axes=([0], [0]))
    
    # L2 norm squared
    l2_sq = np.sum(f ** 2 * _sphere_weights)
    if l2_sq <= 0:
        return np.inf
    l2_norm = np.sqrt(l2_sq)
    
    # L4 norm
    l4_pow4 = np.sum(np.abs(f) ** 4 * _sphere_weights)
    if l4_pow4 <= 0:
        return np.inf
    l4_norm = l4_pow4 ** 0.25
    
    return l4_norm / l2_norm

def _objective(coeffs):
    """Objective to minimize: R = ||f||_4 / ||f||_2"""
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if not np.all(np.isfinite(coeffs)):
        return 1e10
    max_abs = np.max(np.abs(coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        return 1e10
    return _compute_R(coeffs)

def _initial_candidates():
    """Generate diverse initial candidates."""
    candidates = []
    
    # 1. All ones
    candidates.append(np.ones(2 * K + 1))
    
    # 2. Only m=0
    c = np.zeros(2 * K + 1)
    c[K] = 1.0
    candidates.append(c)
    
    # 3. Only highest |m|
    for sign in [1, -1]:
        c = np.zeros(2 * K + 1)
        idx = K + sign * K
        c[idx] = 1.0
        candidates.append(c)
    
    # 4. Alternating signs
    c = np.ones(2 * K + 1) * np.array([(-1)**i for i in range(2 * K + 1)])
    candidates.append(c)
    
    # 5. Symmetric (even m only)
    c = np.zeros(2 * K + 1)
    for m in range(-K, K + 1, 2):
        c[m + K] = 1.0
    candidates.append(c)
    
    # 6. Antisymmetric (odd m only)
    c = np.zeros(2 * K + 1)
    for m in range(-K + 1, K + 1, 2):
        c[m + K] = 1.0
    candidates.append(c)
    
    # 7. Random normal
    rng = np.random.default_rng(42)
    for _ in range(20):
        c = rng.normal(size=2 * K + 1)
        candidates.append(c)
    
    # 8. Sparse random
    for _ in range(10):
        c = np.zeros(2 * K + 1)
        idxs = rng.choice(2 * K + 1, size=rng.integers(3, 8), replace=False)
        c[idxs] = rng.normal(size=len(idxs))
        candidates.append(c)
    
    # 9. Cosine-weighted
    m_vals = np.arange(-K, K + 1)
    c = np.cos(m_vals * 0.3)
    candidates.append(c)
    
    # 10. Sine-weighted
    c = np.sin(m_vals * 0.3)
    candidates.append(c)
    
    # 11. Exponentially decaying from center
    dist = np.abs(m_vals - 0)
    c = np.exp(-0.2 * dist)
    candidates.append(c)
    
    # 12. Inverse of above
    c = 1.0 / (1.0 + 0.5 * dist)
    candidates.append(c)
    
    return candidates

def _local_optimize(start, deadline):
    """Local optimization using BFGS with safeguards."""
    x = np.array(start, dtype=np.float64).copy()
    # Normalize to avoid scale issues
    norm = np.linalg.norm(x)
    if norm > 0:
        x = x / norm * 1.0
    
    best_x = x.copy()
    best_f = _objective(x)
    
    # Try BFGS
    try:
        if time.time() < deadline - 5:
            res = minimize(_objective, x, method='BFGS', options={'maxiter': 200, 'gtol': 1e-6})
            if res.fun < best_f and np.all(np.isfinite(res.x)):
                best_x = res.x.copy()
                best_f = res.fun
    except Exception:
        pass
    
    return best_x, best_f

def rudin_problem10() -> np.ndarray:
    deadline = time.time() + 600
    
    # Phase 1: Generate and evaluate initial candidates
    candidates = _initial_candidates()
    best_coeffs = None
    best_R = np.inf
    
    for cand in candidates:
        if time.time() > deadline:
            break
        R = _compute_R(cand)
        if R < best_R:
            best_R = R
            best_coeffs = cand.copy()
    
    # Phase 2: Local optimization from multiple starts
    starts = []
    scored = [(_compute_R(c), c) for c in candidates]
    scored.sort(key=lambda x: x[0])
    for _, c in scored[:15]:
        starts.append(c)
    
    # Add some more diverse starts
    rng = np.random.default_rng(123)
    for _ in range(10):
        c = rng.normal(size=2 * K + 1)
        starts.append(c)
    
    for start in starts:
        if time.time() > deadline - 30:
            break
        opt_x, opt_R = _local_optimize(start, deadline)
        if opt_R < best_R:
            best_R = opt_R
            best_coeffs = opt_x.copy()
    
    # Phase 3: Random perturbations and re-optimize
    if best_coeffs is not None:
        for _ in range(20):
            if time.time() > deadline - 30:
                break
            perturbed = best_coeffs + rng.normal(0, 0.1, size=2 * K + 1)
            opt_x, opt_R = _local_optimize(perturbed, deadline)
            if opt_R < best_R:
                best_R = opt_R
                best_coeffs = opt_x.copy()
    
    # Phase 4: Differential evolution for global search
    if time.time() < deadline - 60:
        try:
            bounds = [(-10, 10)] * (2 * K + 1)
            de_result = differential_evolution(
                _objective,
                bounds,
                maxiter=100,
                popsize=15,
                seed=42,
                tol=1e-6,
                polish=True
            )
            if de_result.fun < best_R and np.all(np.isfinite(de_result.x)):
                best_R = de_result.fun
                best_coeffs = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 5: Fine-tuning with BFGS from the best solution
    if best_coeffs is not None and time.time() < deadline - 10:
        try:
            res = minimize(_objective, best_coeffs, method='BFGS', options={'maxiter': 500, 'gtol': 1e-8})
            if res.fun < best_R and np.all(np.isfinite(res.x)):
                best_R = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Validate final result
    if best_coeffs is None:
        best_coeffs = np.ones(2 * K + 1)
    
    best_coeffs = np.asarray(best_coeffs, dtype=np.float64)
    # Ensure validity
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(2 * K + 1)
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs / max_abs * 1e-2 if max_abs > 0 else np.ones(2 * K + 1)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs / max_abs * 1e5
    
    return best_coeffs
# EVOLVE-BLOCK-END
