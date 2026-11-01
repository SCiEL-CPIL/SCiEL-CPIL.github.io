import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _repulsion_init(rng):
    """Initialize with repulsion-based spreading to avoid clustering."""
    pts = np.array([rng.normal(size=3) for _ in range(N)])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    # Simple repulsion iterations
    for _ in range(50):
        dots = pts @ pts.T
        np.fill_diagonal(dots, 0)
        # Repulsive force: push away from close neighbors
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i != j:
                    d = dots[i, j]
                    if d > 0.5:  # Only repel if too close
                        forces[i] -= d * (pts[i] - pts[j])
        pts += 0.01 * forces
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _compute_hull_radius(pts):
    """Compute the exact covering radius using the convex hull method.
    Returns (radius_in_radians, worst_facet_indices, worst_pole)."""
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.pi, None, None
            
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        ang_radii = np.arccos(cos_theta)
        
        worst_idx = np.argmax(ang_radii)
        worst_radius = ang_radii[worst_idx]
        worst_face = faces[valid][worst_idx]
        worst_pole = normals[worst_idx]
        
        return worst_radius, worst_face, worst_pole
    except Exception:
        return np.pi, None, None

def _simulated_annealing(pts, deadline, rng):
    """Simulated annealing directly on the hull metric."""
    best_pts = pts.copy()
    best_r, _, _ = _compute_hull_radius(pts)
    curr_r = best_r
    
    T = 0.05  # Initial temperature
    T_min = 0.001
    cooling_rate = 0.995
    step = 0.03
    
    no_improve = 0
    
    while time.time() < deadline and T > T_min:
        # Perturb a random subset of points
        n_perturb = rng.integers(1, 5)
        idx = rng.choice(N, size=n_perturb, replace=False)
        
        cand = pts.copy()
        for i in idx:
            # Random perturbation on the sphere
            noise = rng.normal(size=3) * step
            cand[i] = pts[i] + noise
            norm = np.linalg.norm(cand[i])
            if norm > _EPS:
                cand[i] /= norm
        
        cand_r, _, _ = _compute_hull_radius(cand)
        
        # Accept if better or with probability exp(-delta/T)
        if cand_r < curr_r:
            pts = cand
            curr_r = cand_r
            if curr_r < best_r:
                best_r = curr_r
                best_pts = pts.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            delta = cand_r - curr_r
            if delta < 0.1:  # Don't accept huge jumps
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    pts = cand
                    curr_r = cand_r
                no_improve += 1
        
        if no_improve > 50:
            step *= 0.9
            no_improve = 0
        
        T *= cooling_rate
        
    return best_pts, best_r

def construct():
    deadline = time.time() + 140.0  # Safety margin for 150s limit
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_r = np.pi
    
    # Multi-restart strategy
    n_restarts = 8
    time_per_restart = (deadline - time.time()) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Different initialization strategies
        if restart == 0:
            pts = _initial_fib()
        elif restart == 1:
            pts = _repulsion_init(rng)
        else:
            # Perturbed best so far
            if best_pts is not None:
                pts = best_pts.copy()
                noise = rng.normal(size=(N, 3)) * 0.1
                pts += noise
                pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            else:
                pts = _initial_fib()
        
        # Run SA on this restart
        pts, r = _simulated_annealing(pts, deadline - 3.0, rng)
        
        if r < best_r:
            best_r = r
            best_pts = pts.copy()
    
    # Final normalization
    if best_pts is not None:
        best_pts = np.array(best_pts, dtype=float)
        best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    else:
        best_pts = _initial_fib()
    
    # Validity check
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
