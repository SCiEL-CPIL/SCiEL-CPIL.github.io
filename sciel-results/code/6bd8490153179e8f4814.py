"""Improved logarithmic energy minimizer for 100 points on the unit sphere."""
import time
import math
import numpy as np
from scipy.spatial.distance import cdist

N = 100
DEADLINE = 140.0  # safety margin before 150s kill


def _energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    d = cdist(pts, pts)
    iu = np.triu_indices(N, 1)
    d = d[iu]
    d = np.maximum(d, 1e-15)
    return -math.fsum(np.log(d).tolist())


def _energy_grad(pts):
    """Compute energy and gradient on the sphere (projected)."""
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, 1.0)  # avoid log(0)
    
    # Energy
    iu = np.triu_indices(N, 1)
    d_iu = d[iu]
    d_iu = np.maximum(d_iu, 1e-15)
    e = -math.fsum(np.log(d_iu).tolist())
    
    # Gradient: grad_e = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Actually for log(1/d) = -log(d), dE/dx_i = -sum_j (1/d_ij) * d(d_ij)/d(x_i)
    # d(d_ij)/d(x_i) = (x_i - x_j)/d_ij
    # So dE/dx_i = -sum_{j!=i} (1/d_ij) * (x_i - x_j)/d_ij = -sum_{j!=i} (x_i - x_j)/d_ij^2
    # Wait, let me re-derive: E = sum_{i<j} -log(|x_i - x_j|)
    # dE/dx_i = -sum_{j>i} 1/|x_i-x_j| * (x_i-x_j)/|x_i-x_j| + sum_{j<i} 1/|x_i-x_j| * (x_j-x_i)/|x_j-x_i|
    #         = -sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
    #         = sum_{j!=i} (x_j - x_i)/|x_i - x_j|^2
    
    inv_d2 = np.zeros_like(d)
    np.fill_diagonal(inv_d2, 0.0)
    inv_d2 = 1.0 / (d * d)
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = sum_j (x_j - x_i) / d_ij^2
    # diff[i,j] = x_i - x_j, so (x_j - x_i) = -diff[i,j]
    grad = -np.sum(inv_d2[:, :, np.newaxis] * diff, axis=1)
    
    # Project gradient onto tangent space of sphere
    grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
    
    return e, grad


def _project(pts):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_spiral(offset=0.0):
    """Generate Fibonacci spiral points with optional offset."""
    i = np.arange(N) + 0.5 + offset
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _raster_search(start_pts, step_size=0.05, n_steps=50, time_budget=30.0):
    """Simple gradient descent with adaptive step size."""
    pts = start_pts.copy()
    e0, g0 = _energy_grad(pts)
    best_pts = pts.copy()
    best_e = e0
    t0 = time.time()
    
    for it in range(n_steps):
        if time.time() - t0 > time_budget:
            break
        
        # Adaptive step size via line search
        alpha = step_size
        for _ in range(10):
            new_pts = _project(pts - alpha * g0)
            new_e, new_g = _energy_grad(new_pts)
            if new_e < e0:
                pts = new_pts
                e0 = new_e
                g0 = new_g
                if e0 < best_e:
                    best_e = e0
                    best_pts = pts.copy()
                break
            alpha *= 0.5
        
        # Check convergence
        if np.linalg.norm(g0) < 1e-10:
            break
    
    return best_pts, best_e


def _simulated_annealing(pts, time_budget=40.0):
    """Simulated annealing on the sphere."""
    pts = pts.copy()
    e0 = _energy(pts)
    best_pts = pts.copy()
    best_e = e0
    t0 = time.time()
    
    T = 0.1
    T_min = 1e-6
    cooling = 0.995
    rng = np.random.default_rng(42)
    
    while time.time() - t0 < time_budget and T > T_min:
        # Random perturbation
        noise = rng.normal(0, 0.1, size=pts.shape)
        # Project noise to tangent space
        noise = noise - np.sum(noise * pts, axis=1, keepdims=True) * pts
        new_pts = _project(pts + noise)
        new_e = _energy(new_pts)
        
        delta = new_e - e0
        if delta < 0 or rng.random() < math.exp(-delta / T):
            pts = new_pts
            e0 = new_e
            if e0 < best_e:
                best_e = e0
                best_pts = pts.copy()
        
        T *= cooling
    
    return best_pts, best_e


def _polish(pts, time_budget=20.0):
    """High-precision local optimization."""
    pts = pts.copy()
    e0, g0 = _energy_grad(pts)
    best_pts = pts.copy()
    best_e = e0
    t0 = time.time()
    
    alpha = 0.01
    for it in range(2000):
        if time.time() - t0 > time_budget:
            break
        
        new_pts = _project(pts - alpha * g0)
        new_e, new_g = _energy_grad(new_pts)
        
        if new_e < e0:
            pts = new_pts
            e0 = new_e
            g0 = new_g
            if e0 < best_e:
                best_e = e0
                best_pts = pts.copy()
            alpha = min(alpha * 1.2, 0.1)
        else:
            alpha *= 0.5
        
        if np.linalg.norm(g0) < 1e-12:
            break
    
    return best_pts, best_e


def construct():
    t_start = time.time()
    
    # Generate multiple diverse starting points
    starts = []
    for offset in [0.0, 0.1, 0.2, 0.3, 0.5]:
        starts.append(_fibonacci_spiral(offset))
    
    # Add a few random starts
    rng = np.random.default_rng(123)
    for _ in range(3):
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        starts.append(pts)
    
    best_pts = starts[0].copy()
    best_e = _energy(best_pts)
    
    # Phase 1: Raster search from each start
    for start in starts:
        if time.time() - t_start > DEADLINE - 60:
            break
        pts, e = _raster_search(start, time_budget=15.0)
        if e < best_e:
            best_e = e
            best_pts = pts.copy()
    
    # Phase 2: Simulated annealing from best
    if time.time() - t_start < DEADLINE - 40:
        pts, e = _simulated_annealing(best_pts, time_budget=30.0)
        if e < best_e:
            best_e = e
            best_pts = pts.copy()
    
    # Phase 3: Final polish
    if time.time() - t_start < DEADLINE - 10:
        pts, e = _polish(best_pts, time_budget=15.0)
        if e < best_e:
            best_e = e
            best_pts = pts.copy()
    
    # Ensure validity
    best_pts = _project(best_pts)
    # Check no points are too close
    d = cdist(best_pts, best_pts)
    np.fill_diagonal(d, 1.0)
    if d.min() < 1e-9:
        # Fallback to Fibonacci spiral
        best_pts = _fibonacci_spiral()
    
    return best_pts
