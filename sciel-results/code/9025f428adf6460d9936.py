"""Rewritten program: minimise logarithmic energy of 150 points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 145.0  # seconds


def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum for accuracy."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _log_energy_fast(pts):
    """Vectorized log energy (less precise but much faster)."""
    sq = np.sum(pts * pts, axis=1)
    dist_sq = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    return float(np.sum(np.log(1.0 / d)))


def _log_energy_grad(pts):
    """Compute log energy and Riemannian gradient (tangent space projection included)."""
    sq = np.sum(pts * pts, axis=1)
    dist_sq = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq)

    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    energy = float(np.sum(np.log(1.0 / d)))

    diff = pts[:, None, :] - pts[None, :, :]
    inv_d2 = 1.0 / dist_sq
    np.fill_diagonal(inv_d2, 0.0)
    # grad[i] = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2
    grad = -np.sum(diff * inv_d2[:, :, None], axis=1)
    # Project gradient onto tangent space of sphere
    grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _exp_map_sphere(base, tangent):
    """
    Exponential map on the unit sphere.
    base: (N, 3) points on sphere
    tangent: (N, 3) tangent vectors (orthogonal to base)
    Returns new points on sphere.
    """
    theta = np.linalg.norm(tangent, axis=1, keepdims=True)
    # Avoid division by zero
    safe_theta = np.where(theta > 1e-15, theta, 1.0)
    sin_theta = np.sin(safe_theta)
    cos_theta = np.cos(safe_theta)
    # Handle very small theta: use Taylor expansion
    small = theta < 1e-8
    sin_theta_over_theta = np.ones_like(theta)
    sin_theta_over_theta[~small] = sin_theta[~small] / safe_theta[~small]
    # For small theta, sin(theta)/theta ≈ 1 - theta^2/6
    sin_theta_over_theta[small] = 1.0 - theta[small]**2 / 6.0
    
    result = cos_theta * base + sin_theta_over_theta * tangent
    # Normalize to ensure we're on the sphere
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return result / norms


def _riemannian_lbfgs(pts0, max_iter=5000, tol=1e-12, max_time=None, start_time=None, memory=20):
    """
    Riemannian L-BFGS on the sphere using exponential map for updates.
    This properly handles the manifold constraint.
    """
    if start_time is None:
        start_time = time.time()
    
    x = pts0.copy()
    f, g = _log_energy_grad(x)
    
    # L-BFGS history
    S = []  # tangent steps
    Y = []  # gradient differences
    m = memory
    
    for it in range(max_iter):
        if max_time is not None and (time.time() - start_time) > max_time:
            break
        
        # Compute search direction using L-BFGS two-loop recursion
        # p = -H * g where H is the inverse Hessian approximation
        p = -g.copy()
        
        # Two-loop recursion for L-BFGS
        alphas = []
        for s_k, y_k in zip(reversed(S), reversed(Y)):
            sy = np.dot(s_k, s_k)
            yk_yk = np.dot(y_k, y_k)
            if sy > 1e-30:
                beta = np.dot(y_k, s_k) / sy
                alphas.append(beta)
                p = p + beta * s_k
            else:
                alphas.append(0.0)
        
        # Scale by gamma (inverse of y^T s / s^T s for most recent pair)
        if S and Y:
            sy = np.dot(S[-1], S[-1])
            yk_yk = np.dot(Y[-1], Y[-1])
            if sy > 1e-30 and yk_yk > 1e-30:
                gamma = sy / yk_yk
            else:
                gamma = 1.0
            p = p * gamma
        else:
            gamma = 1.0
        
        for s_k, y_k, alpha in zip(reversed(S), reversed(Y), reversed(alphas)):
            sy = np.dot(s_k, s_k)
            if sy > 1e-30:
                rho = 1.0 / sy
                beta = np.dot(y_k, p)
                p = p + rho * (alpha - beta) * s_k
            # else: alpha was 0, so no update needed
        
        # Ensure p is in tangent space
        p = p - np.sum(p * x, axis=1, keepdims=True) * x
        
        # Line search with backtracking (Armijo condition)
        alpha_ls = 1.0
        c1 = 1e-4
        f_new = f
        g_new = g
        x_new = x
        for _ in range(40):
            x_trial = _exp_map_sphere(x, alpha_ls * p)
            f_trial, g_trial = _log_energy_grad(x_trial)
            if f_trial < f + c1 * alpha_ls * np.dot(g, p):
                f_new = f_trial
                g_new = g_trial
                x_new = x_trial
                break
            alpha_ls *= 0.5
        
        # Update L-BFGS history
        s_step = x_new - x  # approximate tangent step
        # Better: use the actual tangent direction
        s_step = alpha_ls * p
        y_step = g_new - g
        
        sy = np.dot(s_step, y_step)
        if sy > 1e-12:
            S.append(s_step)
            Y.append(y_step)
            if len(S) > m:
                S.pop(0)
                Y.pop(0)
        
        x = x_new
        f = f_new
        g = g_new
        
        # Convergence check
        if np.linalg.norm(g) < tol:
            break
    
    return x, f


def _fibonacci_start():
    """Golden-angle Fibonacci spiral on unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _halton_start(dim=3, n=N):
    """Generate Halton sequence points on the unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    pts = np.zeros((n, dim))
    bases = [2, 3, 5]
    for i in range(n):
        for j, base in enumerate(bases):
            pts[i, j] = halton(i + 1, base)
    
    # Map [0,1)^3 to sphere using spherical coordinates
    # Use a better mapping: stratified sampling
    # Convert to angles
    theta = 2 * np.pi * pts[:, 0]
    # Use sqrt for uniform distribution on sphere
    cos_phi = 1 - 2 * pts[:, 1]
    sin_phi = np.sqrt(np.maximum(0, 1 - cos_phi**2))
    
    x = sin_phi * np.cos(theta)
    y = sin_phi * np.sin(theta)
    z = cos_phi
    
    return np.column_stack([x, y, z])


def _random_start(seed):
    """Random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    return _project_to_sphere(pts)


def _optimize_from_start(pts0, max_time, start_time, max_iter=8000, tol=1e-13):
    """
    Optimize from a given start using Riemannian L-BFGS.
    """
    try:
        result, energy = _riemannian_lbfgs(pts0, max_iter=max_iter, tol=tol,
                                           max_time=max_time, start_time=start_time)
        return result, energy
    except Exception:
        return pts0, _log_energy_fast(pts0)


def _polish(pts, max_time=15.0):
    """
    High-precision polish: run Riemannian L-BFGS with very tight tolerances.
    """
    start_time = time.time()
    result, energy = _riemannian_lbfgs(pts, max_iter=15000, tol=1e-14,
                                       max_time=max_time, start_time=start_time,
                                       memory=30)
    elapsed = time.time() - start_time
    return result, energy, elapsed


def construct():
    global DEADLINE
    start_time = time.time()
    
    # Multiple diverse starting points
    starts = []
    
    # Fibonacci spiral
    pts_fib = _fibonacci_start()
    starts.append(pts_fib)
    
    # Halton sequence
    pts_halton = _halton_start()
    starts.append(pts_halton)
    
    # Random starts
    for seed in [1, 42, 123, 456]:
        pts_rand = _random_start(seed)
        starts.append(pts_rand)
    
    best_pts = starts[0].copy()
    best_energy = _log_energy_fast(starts[0])
    
    # Phase 1: Optimize from each diverse start
    for i, pts_start in enumerate(starts):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 10:
            break
        
        # Give more time to first few starts
        if i < 2:
            max_time = min(remaining - 5, 25.0)
        else:
            max_time = min(remaining - 5, 12.0)
        
        if max_time < 3:
            break
        
        pts_opt, e_opt = _optimize_from_start(pts_start, max_time, start_time,
                                               max_iter=6000 if i < 2 else 4000,
                                               tol=1e-13)
        
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 8:
            e_precise = _log_energy(pts_opt)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = pts_opt.copy()
    
    # Phase 2: Additional random restarts if time permits
    rng = np.random.RandomState(999)
    n_restarts = 20
    for i in range(n_restarts):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 12:
            break
        
        pts_rand = _random_start(1000 + i)
        max_time = min(remaining - 8, 6.0)
        if max_time < 2:
            break
        
        try:
            pts_new, e_new = _optimize_from_start(pts_rand, max_time, start_time,
                                                   max_iter=3000, tol=1e-12)
            elapsed = time.time() - start_time
            if elapsed < DEADLINE - 8:
                e_precise = _log_energy(pts_new)
                if e_precise < best_energy:
                    best_energy = e_precise
                    best_pts = pts_new.copy()
        except Exception:
            continue
    
    # Phase 3: Final high-precision polish on the best configuration
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 8:
        try:
            max_polish_time = min(remaining - 4, 18.0)
            pts_polished, e_polished, elapsed_polish = _polish(best_pts, max_time=max_polish_time)
            elapsed = time.time() - start_time
            if elapsed < DEADLINE - 3:
                e_final = _log_energy(pts_polished)
                if e_final < best_energy:
                    best_energy = e_final
                    best_pts = pts_polished.copy()
        except Exception:
            pass
    
    # Final verification
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 2:
        final_e = _log_energy(best_pts)
        if final_e < best_energy:
            best_energy = final_e
    
    # Ensure valid output
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-15) or np.any(~np.isfinite(best_pts)):
        return _fibonacci_start()
    
    return _project_to_sphere(best_pts)
