# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize

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
    _P[m] = lpmv(m, K, _cos_theta)  # shape (N_THETA,)

# Normalization constants N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
def _norm_const(m):
    m = abs(m)
    val = 21.0 / (4.0 * np.pi) * float(factorial(10 - m) / factorial(10 + m))
    return np.sqrt(val)

# Build the full coefficient matrix: shape (2K+1, N_THETA, N_PHI)
_basis = np.zeros((2 * K + 1, N_THETA, N_PHI), dtype=np.float64)
for idx, m in enumerate(range(-K, K + 1)):
    am = abs(m)
    norm = _norm_const(am)
    Pm = _P[am]  # shape (N_THETA,)
    if m == 0:
        _basis[idx, :, :] = norm * Pm[:, None]
    elif m > 0:
        cos_term = np.cos(m * _phi)  # shape (N_PHI,)
        _basis[idx, :, :] = np.sqrt(2.0) * norm * Pm[:, None] * cos_term[None, :]
    else:
        sin_term = np.sin(am * _phi)  # shape (N_PHI,)
        _basis[idx, :, :] = np.sqrt(2.0) * norm * Pm[:, None] * sin_term[None, :]

# Quadrature weights on the sphere
_w_phi = 2.0 * np.pi / N_PHI
_sphere_weights = np.outer(_theta_weights, np.full(N_PHI, _w_phi))  # shape (N_THETA, N_PHI)
_sphere_weights_flat = _sphere_weights.ravel()  # shape (N_THETA * N_PHI,)

# Precompute basis as flat matrix: shape (N_POINTS, 2K+1)
# f(point) = B(point, :) @ c
_B_flat = _basis.transpose(1, 2, 0).reshape(-1, 2 * K + 1)  # shape (N_POINTS, 21)
N_POINTS = N_THETA * N_PHI

# For the SDP relaxation, we want to minimize ||f||_4^4 / ||f||_2^2 = (sum |f|^4 w) / (sum |f|^2 w)
# But R = ||f||_4 / ||f||_2, so R^4 = ||f||_4^4 / ||f||_2^4
# We minimize R^4 subject to ||f||_2^2 = 1 (scale invariance)
# Actually, let's minimize R = ||f||_4 / ||f||_2 directly using a different approach.

# Key insight: For a fixed direction of c (i.e., c/||c||), R is scale-invariant.
# So we can constrain ||c||_2 = 1 and minimize ||f||_4.
# But ||f||_4 = (sum |f|^4 w)^{1/4} is not convex in c.

# Alternative: Use the fact that we can parameterize c on the unit sphere and use
# a penalty method or direct optimization on the sphere.

# Better structural change: Use a **moment-based SDP relaxation**.
# Let z_i = f(p_i) for each quadrature point. Then z = B @ c.
# ||f||_2^2 = z^T W z where W = diag(w_i)
# ||f||_4^4 = sum w_i z_i^4
# We want to minimize (sum w_i z_i^4) / (z^T W z)^2 subject to z = B c, ||c||_2 = 1.

# This is a quartic optimization. Instead, let's use a different structural approach:
# **Exploit the angular structure** by separating the m modes and using the fact that
# the L4 norm couples different m modes through the phi dependence.

# Actually, a cleaner structural change: Use **alternating minimization** where we
# fix the magnitudes of the coefficients and optimize their phases, then fix phases
# and optimize magnitudes. This exploits the complex structure of the problem.

# But the coefficients are real. Let me think differently.

# New structural approach: **Use the eigenvalue problem formulation**.
# For degree-10 spherical harmonics, the "flattest" function (minimizing L4/L2 ratio)
# is related to the eigenvector of a certain operator. Specifically, if we consider
# the operator that maps a function to its L4 gradient, the minimizer satisfies
# a nonlinear eigenvalue problem. But this is hard to solve directly.

# Let me try a different structural change: **Use a grid-based approach with
# subpixel refinement**. Instead of optimizing in coefficient space, optimize in
# function space on the grid, then project back to coefficients.

# Actually, the cleanest structural change: **Use a randomized SVD / low-rank
# approximation approach**. The L4 norm is dominated by the peak values. We can
# identify the peak locations and use them to guide the optimization.

# Let me implement a **peak-tracking optimization**: 
# 1. Start with a candidate
# 2. Find the peak of |f| on the grid
# 3. Adjust coefficients to reduce the peak while maintaining L2 norm
# 4. Iterate

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
    
    f = _B_flat @ coeffs  # shape (N_POINTS,)
    
    l2_sq = np.sum(f ** 2 * _sphere_weights_flat)
    if l2_sq <= 0:
        return np.inf
    l2_norm = np.sqrt(l2_sq)
    
    l4_pow4 = np.sum(np.abs(f) ** 4 * _sphere_weights_flat)
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

def _peak_tracking_optimize(start, deadline, max_iters=200):
    """Optimize by tracking and reducing peaks."""
    x = np.array(start, dtype=np.float64).copy()
    # Normalize to unit L2 norm
    f = _B_flat @ x
    l2_sq = np.sum(f ** 2 * _sphere_weights_flat)
    if l2_sq > 0:
        x = x / np.sqrt(l2_sq)
    
    best_x = x.copy()
    best_R = _compute_R(x)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        f = _B_flat @ x
        # Find peak location
        peak_idx = np.argmax(np.abs(f))
        peak_val = f[peak_idx]
        
        # Compute gradient of R with respect to x
        # R = ||f||_4 / ||f||_2
        # dR/dx = (1/||f||_2) * d||f||_4/dx - (||f||_4 / ||f||_2^2) * d||f||_2/dx
        # This is complex. Instead, use finite differences for the gradient of R.
        
        # Alternative: Use the fact that we want to reduce the peak.
        # The gradient of |f|^4 at the peak points in the direction of f * B.T @ (w * f^3)
        # Let's use a simple approach: move x in the direction that reduces the peak.
        
        # Gradient of ||f||_4^4 = sum w_i f_i^4 with respect to x:
        # d/dx sum w_i f_i^4 = 4 * B.T @ (w * f^3)
        w = _sphere_weights_flat
        grad_l4_4 = 4.0 * (_B_flat.T @ (w * f ** 3))
        
        # Gradient of ||f||_2^2 = sum w_i f_i^2 with respect to x:
        # d/dx sum w_i f_i^2 = 2 * B.T @ (w * f)
        grad_l2_2 = 2.0 * (_B_flat.T @ (w * f))
        
        # R = ||f||_4 / ||f||_2
        # dR/dx = (1/||f||_2) * (1/4) * ||f||_4^{-3} * grad_l4_4 - (||f||_4 / ||f||_2^2) * (1/2) * ||f||_2^{-1} * grad_l2_2
        # Simplify:
        l4_4 = np.sum(w * f ** 4)
        l2_2 = np.sum(w * f ** 2)
        if l4_4 <= 0 or l2_2 <= 0:
            break
        l4 = l4_4 ** 0.25
        l2 = np.sqrt(l2_2)
        
        # dR/dx = (1/(4 * l2 * l4^3)) * grad_l4_4 - (l4 / (2 * l2^3)) * grad_l2_2
        grad_R = (1.0 / (4.0 * l2 * l4 ** 3)) * grad_l4_4 - (l4 / (2.0 * l2 ** 3)) * grad_l2_2
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad_R)
        if grad_norm < 1e-10:
            break
        
        # Adaptive step size
        lr = 0.1 / grad_norm
        # Limit step to avoid large changes
        lr = min(lr, 0.5 / np.linalg.norm(x) if np.linalg.norm(x) > 0 else 0.5)
        
        x_new = x - lr * grad_R
        
        # Project back to unit L2 norm
        f_new = _B_flat @ x_new
        l2_sq_new = np.sum(f_new ** 2 * _sphere_weights_flat)
        if l2_sq_new > 0:
            x_new = x_new / np.sqrt(l2_sq_new)
        
        R_new = _compute_R(x_new)
        if R_new < best_R:
            best_R = R_new
            best_x = x_new.copy()
        
        # If no improvement, reduce step size
        if R_new >= best_R:
            lr *= 0.5
            x_new = x - lr * grad_R
            f_new = _B_flat @ x_new
            l2_sq_new = np.sum(f_new ** 2 * _sphere_weights_flat)
            if l2_sq_new > 0:
                x_new = x_new / np.sqrt(l2_sq_new)
            R_new = _compute_R(x_new)
            if R_new < best_R:
                best_R = R_new
                best_x = x_new.copy()
        
        x = x_new
    
    return best_x, best_R

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
    for _ in range(30):
        c = rng.normal(size=2 * K + 1)
        candidates.append(c)
    
    # 8. Sparse random
    for _ in range(15):
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
    
    # 13. Bessel-like
    c = np.sinc(m_vals * 0.1)
    candidates.append(c)
    
    # 14. Gaussian in m
    c = np.exp(-0.1 * m_vals ** 2)
    candidates.append(c)
    
    return candidates

def rudin_problem10() -> np.ndarray:
    deadline = time.time() + 590  # Safety margin
    
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
    
    # Phase 2: Peak-tracking optimization from multiple starts
    scored = [(_compute_R(c), c) for c in candidates]
    scored.sort(key=lambda x: x[0])
    starts = [c for _, c in scored[:20]]
    
    # Add some more diverse starts
    rng = np.random.default_rng(123)
    for _ in range(15):
        c = rng.normal(size=2 * K + 1)
        starts.append(c)
    
    for start in starts:
        if time.time() > deadline - 30:
            break
        opt_x, opt_R = _peak_tracking_optimize(start, deadline, max_iters=150)
        if opt_R < best_R:
            best_R = opt_R
            best_coeffs = opt_x.copy()
    
    # Phase 3: Random perturbations and re-optimize
    if best_coeffs is not None:
        for _ in range(30):
            if time.time() > deadline - 30:
                break
            perturbed = best_coeffs + rng.normal(0, 0.1, size=2 * K + 1)
            opt_x, opt_R = _peak_tracking_optimize(perturbed, deadline, max_iters=100)
            if opt_R < best_R:
                best_R = opt_R
                best_coeffs = opt_x.copy()
    
    # Validate final result
    if best_coeffs is None:
        best_coeffs = np.ones(2 * K + 1)
    
    best_coeffs = np.asarray(best_coeffs, dtype=np.float64)
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(2 * K + 1)
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs / max_abs * 1e-2 if max_abs > 0 else np.ones(2 * K + 1)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs / max_abs * 1e5
    
    return best_coeffs
# EVOLVE-BLOCK-END
