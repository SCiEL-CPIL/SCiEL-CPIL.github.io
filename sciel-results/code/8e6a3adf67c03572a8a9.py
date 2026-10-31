"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 16
D = 2

# EVOLVE-BLOCK-START
def _score(points):
    """Compute (max_dist / min_dist)^2 for a set of points."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    min_d_sq = np.min(dists_sq)
    max_d_sq = np.max(dists_sq)
    if min_d_sq <= 0:
        return np.inf
    return max_d_sq / min_d_sq

def _soft_score(points, eps=1e-6):
    """Smooth approximation of the score for gradient-based optimization."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    
    min_d_sq = np.min(dists_sq)
    max_d_sq = np.max(dists_sq)
    
    # Soft max and min using log-sum-exp
    temp = 100.0
    soft_max = np.log(np.sum(np.exp(dists_sq / temp))) * temp
    soft_min = -np.log(np.sum(np.exp(-dists_sq / temp))) * temp
    
    if soft_min <= eps:
        return 1e10
    return soft_max / soft_min

def _generate_hex_seed(scale=1.0):
    """Generate a hexagonal seed configuration."""
    pts = [np.array([0.0, 0.0])]
    for ring in range(1, 4):
        for i in range(6 * ring):
            angle = i * 2 * np.pi / (6 * ring) + (ring % 2) * np.pi / (6 * ring)
            r = ring
            pts.append(np.array([r * np.cos(angle), r * np.sin(angle)]))
        if len(pts) >= N:
            break
    pts = np.array(pts[:N])
    # Center and scale
    pts = pts - np.mean(pts, axis=0)
    return pts * scale

def _generate_grid_seed(spacing=1.0):
    """Generate a 4x4 grid seed."""
    side = int(np.ceil(np.sqrt(N)))
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) < N:
                pts.append(np.array([i * spacing, j * spacing]))
    pts = np.array(pts[:N])
    pts = pts - np.mean(pts, axis=0)
    return pts

def _generate_spiral_seed(radius=2.0):
    """Generate a spiral seed."""
    angles = np.linspace(0, 4 * np.pi, N)
    r = np.linspace(0.1, radius, N)
    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    pts = pts - np.mean(pts, axis=0)
    return pts

def _direct_optimize(pts0, deadline, max_iter=2000):
    """Optimize all point positions directly using L-BFGS-B."""
    def objective(x):
        pts = x.reshape(N, 2)
        return _soft_score(pts)
    
    def gradient(x):
        pts = x.reshape(N, 2)
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=-1)
        np.fill_diagonal(dists_sq, np.inf)
        
        min_d_sq = np.min(dists_sq)
        max_d_sq = np.max(dists_sq)
        
        if min_d_sq <= 1e-8:
            return np.zeros_like(x)
        
        ratio = max_d_sq / min_d_sq
        
        # Gradient with respect to min distance (want to maximize min distance)
        min_idx = np.argmin(dists_sq)
        i, j = np.unravel_index(min_idx, (n, n))
        grad_min = np.zeros_like(pts)
        grad_min[i] = 2 * (pts[i] - pts[j])
        grad_min[j] = 2 * (pts[j] - pts[i])
        
        # Gradient with respect to max distance (want to minimize max distance)
        max_idx = np.argmax(dists_sq)
        i, j = np.unravel_index(max_idx, (n, n))
        grad_max = np.zeros_like(pts)
        grad_max[i] = 2 * (pts[i] - pts[j])
        grad_max[j] = 2 * (pts[j] - pts[i])
        
        # d(ratio)/d(min) = -max/min^2, d(ratio)/d(max) = 1/min
        grad = (-max_d_sq / min_d_sq**2) * grad_min + (1.0 / min_d_sq) * grad_max
        
        return grad.reshape(-1)
    
    x0 = pts0.flatten()
    
    res = minimize(
        objective,
        x0,
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    pts_opt = res.x.reshape(N, 2)
    # Center the result
    pts_opt = pts_opt - np.mean(pts_opt, axis=0)
    return pts_opt, _score(pts_opt)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin (150s limit)
    
    # Generate multiple diverse seeds
    seeds = [
        _generate_hex_seed(1.0),
        _generate_hex_seed(0.8),
        _generate_hex_seed(1.2),
        _generate_grid_seed(1.0),
        _generate_grid_seed(0.8),
        _generate_spiral_seed(2.0),
        _generate_spiral_seed(1.5),
    ]
    
    # Add some random perturbations
    rng = np.random.default_rng(42)
    for _ in range(4):
        base = seeds[rng.integers(0, len(seeds))]
        perturbed = base + rng.normal(0, 0.3, size=base.shape)
        seeds.append(perturbed)
    
    best_result = seeds[0]
    best_score = _score(best_result)
    
    # Try each seed with direct optimization
    for seed in seeds:
        if time.time() > deadline:
            break
        
        try:
            pts_opt, score = _direct_optimize(seed, deadline)
            if score < best_score:
                best_score = score
                best_result = pts_opt
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline:
            break
    
    # Final local refinement with Nelder-Mead if time permits
    if time.time() < deadline - 20:
        try:
            def obj_nm(x):
                return _score(x.reshape(N, 2))
            
            x0 = best_result.flatten()
            res = minimize(
                obj_nm,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            pts_nm = res.x.reshape(N, 2)
            score_nm = _score(pts_nm)
            if score_nm < best_score:
                best_score = score_nm
                best_result = pts_nm
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
