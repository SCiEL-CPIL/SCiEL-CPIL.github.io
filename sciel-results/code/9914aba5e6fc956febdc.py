import numpy as np
import time
import math

N = 150

def _energy_and_grad(points):
    """Compute log energy and Riemannian gradient for points on unit sphere.
    
    Energy: E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Riemannian gradient: grad_i = sum_{j!=i} -(x_i - x_j)/|x_i - x_j|^2 - (x_i · grad_i) * x_i
    """
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu[0], iu[1]]))
    
    # Euclidean gradient: for each i, grad_i = sum_{j!=i} -(x_i - x_j)/|x_i - x_j|^2
    # coeff[i,j] = -1/|x_i - x_j|^2 for i != j
    coeff = np.zeros((n, n))
    mask = ~np.eye(n, dtype=bool)
    coeff[mask] = -1.0 / dist_sq[mask]
    # grad_i = sum_j coeff[i,j] * (x_i - x_j)
    # = sum_j coeff[i,j] * x_i - sum_j coeff[i,j] * x_j
    # = (sum_j coeff[i,j]) * x_i - (coeff @ points)[i]
    row_sum = np.sum(coeff, axis=1)
    eucl_grad = row_sum[:, np.newaxis] * points - (coeff @ points)
    
    # Riemannian gradient: project to tangent space
    # proj(v) = v - (v · x_i) * x_i
    dot = np.sum(eucl_grad * points, axis=1)
    riem_grad = eucl_grad - dot[:, np.newaxis] * points
    
    return energy, riem_grad


def _energy_only(points):
    """Fast energy computation."""
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu[0], iu[1]]))


def _fibonacci_spiral(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, seed=None):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _project_to_sphere(points):
    """Project points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms


def _retraction(points, grad, step):
    """Exponential map retraction on sphere: move along gradient direction."""
    # Move in direction of -grad, then re-normalize
    new_pts = points - step * grad
    return _project_to_sphere(new_pts)


def _riemannian_cg(points, max_iter=500, tol=1e-10, deadline=None):
    """Riemannian conjugate gradient on sphere."""
    n = points.shape[0]
    p = points.copy()
    e0, g0 = _energy_and_grad(p)
    d = -g0.copy()
    e_prev = e0
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Line search: backtracking
        alpha = 0.01
        for _ in range(50):
            new_p = _retraction(p, g0, alpha)
            new_e = _energy_only(new_p)
            if new_e < e0:
                break
            alpha *= 0.5
        else:
            break
        
        p = new_p
        e0 = new_e
        _, g0 = _energy_and_grad(p)
        
        # Check convergence
        if np.linalg.norm(g0) < tol:
            break
        
        # Polak-Ribiere beta
        d_old = d.copy()
        d = -g0
        dot_gd = np.sum(g0 * d_old)
        dot_gg = np.sum(g0 * g0)
        if dot_gg > 1e-30:
            beta = max(0.0, dot_gd / dot_gg)
            d = d + beta * d_old
        
        e_prev = e0
    
    return p, e0


def construct():
    deadline = time.time() + 140.0  # 10s safety margin before 150s kill
    
    # Initial: Fibonacci spiral
    points = _fibonacci_spiral(N)
    best_points = points.copy()
    best_energy = _energy_only(best_points)
    
    # Multiple restarts with different initializations
    rng = np.random.RandomState(42)
    restarts = [
        ("fib", _fibonacci_spiral(N)),
    ]
    # Add some random restarts
    for k in range(3):
        restarts.append((f"rand{k}", _random_points(N, seed=42 + k)))
    
    for name, init_pts in restarts:
        if time.time() > deadline - 30:
            break
        
        # Riemannian CG
        try:
            opt_pts, opt_e = _riemannian_cg(init_pts, max_iter=300, tol=1e-10, deadline=deadline - 10)
            opt_pts = _project_to_sphere(opt_pts)
            opt_e = _energy_only(opt_pts)
            if opt_e < best_energy:
                best_energy = opt_e
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Polish with more iterations on the best
    if time.time() < deadline - 15:
        try:
            opt_pts, opt_e = _riemannian_cg(best_points, max_iter=500, tol=1e-12, deadline=deadline - 5)
            opt_pts = _project_to_sphere(opt_pts)
            opt_e = _energy_only(opt_pts)
            if opt_e < best_energy:
                best_energy = opt_e
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Final projection and validation
    best_points = _project_to_sphere(best_points)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                best_points[j] += 1e-8 * rng.randn(3)
                best_points = _project_to_sphere(best_points)
    
    return best_points
