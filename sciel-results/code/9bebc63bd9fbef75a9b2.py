"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _normalize(v):
    v = np.asarray(v)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return v
        return v / norm
    else:
        norms = np.linalg.norm(v, axis=1)
        norms[norms < 1e-12] = 1.0
        return v / norms[:, np.newaxis]


# Pre-compute sample points for covering radius estimation
_N_SAMPLES = 20000
def _get_sample_points():
    i = np.arange(_N_SAMPLES) + 0.5
    z = 1.0 - 2.0 * i / _N_SAMPLES
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

_SAMPLES = _get_sample_points()


def _covering_radius(pts):
    """Compute covering radius using a dense sample grid on the sphere."""
    dots = _SAMPLES @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    max_dots = np.max(dots, axis=1)
    angles = np.arccos(max_dots)
    max_angle = np.max(angles)
    return max_angle * 180.0 / np.pi


def _local_optimize(pts, max_time=30.0, seed=None):
    """Local optimization to reduce covering radius using steepest descent."""
    if seed is not None:
        np.random.seed(seed)
    start_time = time.time()
    pts = _normalize(pts.copy())
    
    best_cr = _covering_radius(pts)
    best_pts = pts.copy()
    
    step_size = 0.1
    n_iter = 0
    no_improve_count = 0
    
    while time.time() - start_time < max_time:
        n_iter += 1
        
        # Compute gradient by finite differences in tangent space
        grad = np.zeros_like(pts)
        eps = 1e-4
        for i in range(N):
            for d in range(3):
                pts_pert = pts.copy()
                pts_pert[i, d] += eps
                pts_pert = _normalize(pts_pert)
                cr_pert = _covering_radius(pts_pert)
                grad[i, d] = (cr_pert - best_cr) / eps
        
        # Project gradient to tangent space
        for i in range(N):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        grad /= grad_norm
        
        # Try moving in negative gradient direction
        new_pts = pts - step_size * grad
        new_pts = _normalize(new_pts)
        
        cr_new = _covering_radius(new_pts)
        if cr_new < best_cr - 1e-10:
            best_cr = cr_new
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            no_improve_count = 0
            step_size = min(step_size * 1.05, 0.5)
        else:
            no_improve_count += 1
            step_size *= 0.85
            if step_size < 1e-5:
                break
        
        if no_improve_count > 50:
            break
    
    return best_pts, best_cr


def _try_move_point(pts, i, max_time_remaining, seed=None):
    """Try to improve by moving point i in various directions."""
    if seed is not None:
        np.random.seed(seed)
    start_time = time.time()
    best_cr = _covering_radius(pts)
    best_pts = pts.copy()
    
    for trial in range(30):
        if time.time() - start_time > max_time_remaining - 2.0:
            break
        
        direction = np.random.randn(3)
        norm_dir = np.linalg.norm(direction)
        if norm_dir < 1e-10:
            continue
        direction /= norm_dir
        
        new_pts = pts.copy()
        new_pt = new_pts[i] + 0.05 * direction
        new_pts[i] = _normalize(new_pt)
        
        cr = _covering_radius(new_pts)
        if cr < best_cr - 1e-10:
            best_cr = cr
            best_pts = new_pts.copy()
            pts = new_pts.copy()
    
    return best_pts, best_cr


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Phase 1: Gradient-based local optimization
    if time.time() < deadline - 60.0:
        remaining = deadline - time.time() - 10.0
        optimized_pts, cr = _local_optimize(pts, max_time=min(remaining, 50.0), seed=42)
        if cr < best_cr:
            best_pts = optimized_pts
            best_cr = cr
    
    # Phase 2: Random restarts with local optimization
    np.random.seed(42)
    for restart in range(10):
        if time.time() > deadline - 20.0:
            break
        
        # Random start
        rand_pts = np.random.randn(N, 3)
        rand_pts = _normalize(rand_pts)
        
        remaining = deadline - time.time() - 5.0
        if remaining < 5.0:
            break
        
        opt_pts, cr = _local_optimize(rand_pts, max_time=min(remaining, 15.0), seed=42+restart)
        if cr < best_cr:
            best_pts = opt_pts
            best_cr = cr
    
    # Phase 3: Fine-tune the best configuration
    if time.time() < deadline - 10.0:
        remaining = deadline - time.time() - 5.0
        if remaining > 5.0:
            fine_pts, cr = _local_optimize(best_pts, max_time=min(remaining, 20.0), seed=999)
            if cr < best_cr:
                best_pts = fine_pts
                best_cr = cr
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
