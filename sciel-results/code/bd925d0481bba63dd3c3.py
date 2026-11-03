"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

_START_TIME = time.time()
_DEADLINE = _START_TIME + 135.0  # safety margin before 150s kill

def _to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius(pts):
    """
    Compute the covering radius (in radians) of a set of points on the sphere.
    Uses convex hull facets: for each facet, the circumcenter on the sphere
    gives a candidate for the largest empty circle center.
    """
    pts = _to_sphere(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return _covering_radius_fallback(pts)
    
    max_radius = 0.0
    # Precompute all pairwise dot products for antipode check
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Circumcenter direction: cross((a-b), (a-c)) gives normal to triangle plane
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            continue
        
        x = cross / norm_cross
        
        # Angular distance from x to vertex a
        dot_a = np.dot(x, a)
        dot_a = np.clip(dot_a, -1.0, 1.0)
        r1 = np.arccos(dot_a)
        
        # The other direction
        dot_a_neg = -dot_a
        dot_a_neg = np.clip(dot_a_neg, -1.0, 1.0)
        r2 = np.arccos(dot_a_neg)
        
        # Take the larger circumradius (the one in the "empty" cap direction)
        candidate = max(r1, r2)
        if candidate > max_radius:
            max_radius = candidate
    
    # Check antipodes of each point
    for i in range(len(pts)):
        neg_pt = -pts[i]
        dots = pts @ neg_pt
        dots = np.clip(dots, -1.0, 1.0)
        min_dist = np.arccos(np.min(dots))
        if min_dist > max_radius:
            max_radius = min_dist
    
    return max_radius

def _covering_radius_fallback(pts):
    """Fallback covering radius computation via moderate sampling."""
    n_theta = 120
    n_phi = 240
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    max_radius = 0.0
    for theta in thetas:
        st = np.sin(theta)
        ct = np.cos(theta)
        y_batch = np.column_stack([
            st * np.cos(phis),
            st * np.sin(phis),
            np.full_like(phis, ct)
        ])
        dots = pts @ y_batch.T
        dots = np.clip(dots, -1.0, 1.0)
        min_dists = np.arccos(np.min(dots, axis=0))
        max_radius = max(max_radius, np.max(min_dists))
    return max_radius

def _objective(pts):
    """Compute the objective (covering radius in radians)."""
    return _covering_radius(pts)

def _local_optimize(pts, max_time=None):
    """
    Local optimization using a repulsion-based force model.
    Each point is pushed away from neighbors to maximize minimum spacing.
    """
    if max_time is None:
        max_time = 10.0
    
    pts = _to_sphere(pts.copy())
    current_obj = _objective(pts)
    best_pts = pts.copy()
    best_obj = current_obj
    
    start = time.time()
    step_size = 0.05
    n_iter = 0
    no_improve = 0
    
    while time.time() - start < max_time:
        # Compute pairwise distances
        dots = pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        
        # Force model: each point is repelled by all others, weighted by
        # how close they are relative to the target spacing
        forces = np.zeros_like(pts)
        for i in range(N):
            # Exclude self
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            d_i = dists[i, mask]
            pts_i = pts[mask]
            
            # Repulsion: stronger for closer points
            # Use inverse distance squared-like weighting
            weights = 1.0 / (d_i + 0.01) ** 2
            forces[i] = np.sum(weights[:, None] * (pts[i][None, :] - pts_i), axis=0)
        
        # Normalize forces to unit vectors
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms[force_norms == 0] = 1.0
        forces = forces / force_norms
        
        # Update points
        new_pts = pts + step_size * forces
        new_pts = _to_sphere(new_pts)
        
        new_obj = _objective(new_pts)
        
        if new_obj < current_obj:
            pts = new_pts
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_pts = pts.copy()
            no_improve = 0
            step_size = min(step_size * 1.02, 0.15)
        else:
            no_improve += 1
            step_size *= 0.95
            if no_improve > 50:
                break
        
        n_iter += 1
        if n_iter >= 200:
            break
    
    return best_pts, best_obj

def _simulated_annealing(pts, max_time):
    """
    Simulated annealing to find a good configuration.
    """
    pts = _to_sphere(pts.copy())
    current_obj = _objective(pts)
    best_pts = pts.copy()
    best_obj = current_obj
    
    start = time.time()
    temperature = 0.05
    cooling = 0.998
    step_size = 0.08
    
    while time.time() < max_time:
        # Propose a random perturbation
        noise = np.random.randn(N, 3) * step_size
        new_pts = _to_sphere(pts + noise)
        
        new_obj = _objective(new_pts)
        
        delta = new_obj - current_obj
        if delta < 0 or (temperature > 1e-6 and np.random.random() < np.exp(-delta / temperature)):
            pts = new_pts
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_pts = pts.copy()
        
        temperature *= cooling
        if temperature < 0.0005:
            temperature = 0.0005
            step_size = max(step_size * 0.9, 0.02)
    
    return best_pts, best_obj

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    # Start with Fibonacci spiral initial configuration
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _to_sphere(pts)
    
    best_pts = pts.copy()
    best_obj = _objective(pts)
    
    deadline = _DEADLINE
    remaining = deadline - time.time()
    
    # Phase 1: Local optimization from initial config
    if remaining > 5:
        pts, obj = _local_optimize(pts, max_time=min(remaining - 5, 25))
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
    
    # Phase 2: Simulated annealing with restarts
    remaining = deadline - time.time()
    if remaining > 10:
        np.random.seed(42)
        for restart in range(8):
            if time.time() > deadline - 10:
                break
            
            remaining = deadline - time.time()
            if remaining < 10:
                break
            
            # Perturb best solution
            perturbation = np.random.randn(N, 3) * 0.3
            perturbed = _to_sphere(best_pts + perturbation)
            pts, obj = _simulated_annealing(perturbed, max_time=min(remaining - 10, 15))
            
            if obj < best_obj:
                best_obj = obj
                best_pts = pts.copy()
            
            # Local polish from SA result
            remaining = deadline - time.time()
            if remaining > 8:
                pts, obj = _local_optimize(pts, max_time=min(remaining - 8, 10))
                if obj < best_obj:
                    best_obj = obj
                    best_pts = pts.copy()
    
    # Phase 3: Final high-precision polish
    remaining = deadline - time.time()
    if remaining > 5:
        pts, obj = _local_optimize(best_pts, max_time=min(remaining - 5, 15))
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
    
    # Ensure all points are on the unit sphere
    best_pts = _to_sphere(best_pts)
    
    # Validate: check for coincident points
    dots = best_pts @ best_pts.T
    np.fill_diagonal(dots, 0)
    if np.max(np.abs(dots)) > 0.9999:
        best_pts = _to_sphere(best_pts + np.random.randn(N, 3) * 0.01)
        best_pts = _to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
