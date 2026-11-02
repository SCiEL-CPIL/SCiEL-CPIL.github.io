"""Minimal logarithmic energy of 150 points on the sphere - Basin Hopping + Riemannian CG."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150


def _energy_and_grad(pts):
    """Vectorized energy and gradient computation on the sphere."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    
    # Check for too-close points
    iu = np.triu_indices(n, k=1)
    d_upper = dists[iu]
    if np.any(d_upper < 1e-12):
        return float('inf'), np.zeros_like(pts)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    log_dists = np.log(dists)
    energy = -0.5 * np.sum(log_dists)
    
    # Gradient: grad_i E = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    weighted_sum = inv_d2 @ pts
    sum_inv = np.sum(inv_d2, axis=1)
    grad = weighted_sum - pts * sum_inv[:, np.newaxis]
    
    # Project to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    return energy, grad


def _energy_flat(x_flat):
    """Energy function for flat array input (for scipy minimize)."""
    pts = x_flat.reshape(N, 3)
    energy, _ = _energy_and_grad(pts)
    return energy


def _grad_flat(x_flat):
    """Gradient function for flat array input (for scipy minimize)."""
    pts = x_flat.reshape(N, 3)
    _, grad = _energy_and_grad(pts)
    return grad.ravel()


def _fibonacci_sphere(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _halton_sphere(n):
    """Generate n points on sphere using Halton sequence."""
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    points = []
    for i in range(1, n + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        p = np.array([x, y, z])
        norm = np.linalg.norm(p)
        if norm > 1e-15:
            points.append(p / norm)
        else:
            points.append(np.array([1.0, 0.0, 0.0]))
    return np.array(points)


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms


def _exp_map(pts, v):
    """Exponential map on sphere."""
    norms_v = np.linalg.norm(v, axis=1, keepdims=True)
    safe_norms = np.where(norms_v < 1e-15, 1.0, norms_v)
    cos_t = np.cos(norms_v)
    sin_t = np.sin(norms_v)
    return pts * cos_t + v * (sin_t / safe_norms)


def _riemannian_cg(pts0, max_iter=200, time_limit=None, start_time=None, tol=1e-12):
    """Riemannian Conjugate Gradient on the sphere."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    
    if not np.isfinite(energy):
        return pts, energy
    
    # Project initial gradient to tangent space
    def _project_tangent(p, v):
        dot = np.sum(p * v, axis=1)
        return v - dot[:, np.newaxis] * p
    
    d = _project_tangent(pts, -grad)
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        if np.linalg.norm(grad) < tol:
            break
        
        # Line search
        alpha = 1.0
        accepted = False
        for _ in range(30):
            new_pts = _project_to_sphere(pts + alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < energy + 1e-4 * alpha * np.sum(grad * d):
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            alpha = max(alpha, 1e-14)
            new_pts = _project_to_sphere(pts + alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            if new_energy < energy:
                accepted = True
            else:
                break
        
        if not accepted:
            break
        
        # Polak-Ribière update
        new_grad = _project_tangent(new_pts, new_grad)
        s = new_pts - pts
        y = new_grad - grad
        sy = np.sum(s * y)
        gy = np.sum(grad * y)
        
        if sy > 1e-16:
            beta = max(0.0, gy / sy)
        else:
            beta = 0.0
        
        d_new = _project_tangent(new_pts, -new_grad + beta * d)
        
        pts = new_pts
        grad = new_grad
        energy = new_energy
        d = d_new
    
    return pts, energy


def _basin_hopping(pts0, n_iter=100, step_size=0.1, time_limit=None, start_time=None, seed=42):
    """Basin hopping: random perturbation + Riemannian CG polish."""
    if start_time is None:
        start_time = time.time()
    
    rng = np.random.default_rng(seed)
    pts = pts0.copy()
    energy, _ = _energy_and_grad(pts)
    
    if not np.isfinite(energy):
        return pts, energy
    
    best_pts = pts.copy()
    best_energy = energy
    
    for it in range(n_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Random perturbation
        noise = rng.standard_normal((N, 3)) * step_size
        # Project noise to tangent space
        dot = np.sum(pts * noise, axis=1)
        noise = noise - dot[:, np.newaxis] * pts
        perturbed = _project_to_sphere(pts + noise)
        
        # Polish with Riemannian CG
        polished, polished_energy = _riemannian_cg(
            perturbed, max_iter=100,
            time_limit=time_limit, start_time=start_time, tol=1e-12
        )
        
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_pts = polished.copy()
            pts = polished.copy()
            # Reduce step size when improving
            step_size *= 0.95
        else:
            # Increase step size when not improving
            step_size = min(step_size * 1.05, 0.5)
    
    return best_pts, best_energy


def _lbfgs_scipy(pts0, time_limit=None, start_time=None):
    """L-BFGS using scipy.optimize.minimize on the sphere."""
    if start_time is None:
        start_time = time.time()
    
    x0 = pts0.ravel()
    
    # Use scipy L-BFGS-B with our gradient
    result = minimize(
        _energy_flat, x0,
        jac=_grad_flat,
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 1000}
    )
    
    pts = result.x.reshape(N, 3)
    pts = _project_to_sphere(pts)
    energy, _ = _energy_and_grad(pts)
    
    return pts, energy


def _final_polish(pts0, time_limit=None, start_time=None):
    """Final polish using Riemannian CG with tight tolerance."""
    if start_time is None:
        start_time = time.time()
    
    pts, energy = _riemannian_cg(pts0, max_iter=100, time_limit=time_limit, start_time=start_time, tol=1e-14)
    
    # One more round with even tighter tolerance
    pts, energy = _riemannian_cg(pts, max_iter=50, time_limit=time_limit, start_time=start_time, tol=1e-16)
    
    return pts, energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Basin Hopping with multiple restarts
    # Allocate ~65% of time to basin hopping
    phase1_deadline = start_time + (deadline - start_time) * 0.65
    
    # Different initial configurations
    inits = [
        _fibonacci_sphere(N),
        _halton_sphere(N),
    ]
    
    # Add random inits
    for seed in [42, 123, 456, 789, 1000, 2024]:
        rng = np.random.default_rng(seed)
        pts_rand = rng.standard_normal((N, 3))
        pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        inits.append(pts_rand)
    
    # Perturbed Fibonacci
    for seed, strength in [(7, 0.05), (99, 0.1), (1234, 0.03)]:
        rng = np.random.default_rng(seed)
        pts_fib = _fibonacci_sphere(N)
        pts_pert = pts_fib + rng.standard_normal((N, 3)) * strength
        pts_pert /= np.linalg.norm(pts_pert, axis=1, keepdims=True)
        inits.append(pts_pert)
    
    # Run basin hopping on each init
    for idx, init_pts in enumerate(inits):
        if time.time() > phase1_deadline:
            break
        
        remaining = phase1_deadline - time.time()
        num_remaining = len(inits) - idx
        time_alloc = min(remaining / num_remaining, 15.0)
        time_alloc = max(time_alloc, 5.0)
        
        # Use different step sizes for diversity
        step_size = 0.08 + 0.02 * (idx % 3)
        
        pts, energy = _basin_hopping(
            init_pts,
            n_iter=80,
            step_size=step_size,
            time_limit=time_alloc,
            start_time=time.time(),
            seed=idx * 100 + 42
        )
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 2: L-BFGS polish on best basin hopping result
    if best_pts is not None and time.time() < deadline - 25:
        remaining = deadline - time.time() - 20
        if remaining > 5:
            best_pts, best_energy = _lbfgs_scipy(
                best_pts,
                time_limit=remaining,
                start_time=time.time()
            )
    
    # Phase 3: Final Riemannian CG polish
    if best_pts is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        if remaining > 3:
            best_pts, best_energy = _final_polish(
                best_pts,
                time_limit=remaining,
                start_time=time.time()
            )
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    best_pts = best_pts / norms
    
    # Verify validity
    valid = True
    for i in range(N):
        if not np.all(np.isfinite(best_pts[i])):
            valid = False
            break
        if np.linalg.norm(best_pts[i]) < 1e-10:
            valid = False
            break
    
    if valid:
        min_dist = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_pts[i] - best_pts[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 1e-9:
            valid = False
    
    if not valid:
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
