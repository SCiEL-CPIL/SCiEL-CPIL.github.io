"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # safety margin before 150s kill

def _to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius(pts):
    """
    Compute the covering radius (in radians) of a set of points on the sphere.
    Uses convex hull facets to find the largest empty circle.
    """
    pts = _to_sphere(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return _covering_radius_fallback(pts)
    
    max_radius = 0.0
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            continue
        
        x = cross / norm_cross
        
        # Check both directions
        for sign in [1, -1]:
            xc = sign * x
            dot_a = np.clip(np.dot(xc, a), -1.0, 1.0)
            r = np.arccos(dot_a)
            
            # Verify this is a valid circumcenter: all other points should be
            # at distance >= r from xc (empty cap check)
            dots = np.clip(pts @ xc, -1.0, 1.0)
            dists = np.arccos(dots)
            min_other = np.min(dists)
            
            if min_other >= r - 1e-9:
                max_radius = max(max_radius, r)
    
    # Also check antipodes of each point
    for i in range(len(pts)):
        neg_pt = -pts[i]
        dots = np.clip(pts @ neg_pt, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dist = np.min(dists)
        max_radius = max(max_radius, min_dist)
    
    return max_radius

def _covering_radius_fallback(pts):
    """Fallback covering radius computation via dense sampling."""
    n_theta = 100
    n_phi = 200
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    max_radius = 0.0
    for theta in thetas:
        st = np.sin(theta)
        ct = np.cos(theta)
        for phi in phis:
            y = np.array([st*np.cos(phi), st*np.sin(phi), ct])
            dots = np.clip(pts @ y, -1.0, 1.0)
            min_dist = np.arccos(np.min(dots))
            if min_dist > max_radius:
                max_radius = min_dist
    return max_radius

def _fast_covering_radius_proxy(pts):
    """
    Fast proxy for the covering radius using vectorized circumcenter checks.
    Checks a subset of triples for speed.
    """
    pts = _to_sphere(pts)
    max_radius = 0.0
    
    # Check circumcenters of all triples (vectorized)
    # For N=50, there are C(50,3) = 19600 triples
    # We'll check all of them but vectorize the computation
    
    # Generate all triples
    idx_i, idx_j, idx_k = np.meshgrid(
        np.arange(N), np.arange(N), np.arange(N), indexing='ij'
    )
    mask = (idx_i < idx_j) & (idx_j < idx_k)
    i_idx = idx_i[mask]
    j_idx = idx_j[mask]
    k_idx = idx_k[mask]
    
    a = pts[i_idx]
    b = pts[j_idx]
    c = pts[k_idx]
    
    v1 = a - b
    v2 = a - c
    cross = np.cross(v1, v2)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x = cross / norms
    
    # Check both directions
    for sign in [1, -1]:
        xc = sign * x
        # Distance from xc to each of a, b, c
        dots_a = np.clip(np.sum(xc * a, axis=1), -1.0, 1.0)
        r = np.arccos(dots_a)
        
        # Check if all other points are at distance >= r
        # This is expensive, so we'll only check a subset
        # For speed, we'll just use the circumradius without full verification
        max_radius = max(max_radius, np.max(r))
    
    # Check antipodes
    for i in range(N):
        neg_pt = -pts[i]
        dots = np.clip(pts @ neg_pt, -1.0, 1.0)
        min_dist = np.arccos(np.min(dots))
        max_radius = max(max_radius, min_dist)
    
    return max_radius

def _repulsion_forces(pts):
    """
    Compute repulsion forces: each point is pushed away from its nearest neighbors.
    """
    dots = pts @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    np.fill_diagonal(dists, np.inf)
    
    forces = np.zeros_like(pts)
    for i in range(N):
        # Find the 5 nearest neighbors
        idx = np.argsort(dists[i])[:5]
        for j in idx:
            d = dists[i, j]
            f = 1.0 / (d + 0.1)
            forces[i] += f * (pts[i] - pts[j])
        
        # Project force to tangent plane
        forces[i] -= np.dot(forces[i], pts[i]) * pts[i]
    
    return forces

def _local_optimize(pts, max_time=30.0):
    """
    Local optimization using repulsion forces and the fast proxy.
    """
    pts = _to_sphere(pts.copy())
    start = time.time()
    
    step_size = 0.05
    best_pts = pts.copy()
    best_obj = _fast_covering_radius_proxy(pts)
    
    while time.time() - start < max_time:
        forces = _repulsion_forces(pts)
        
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms[force_norms == 0] = 1.0
        forces = forces / force_norms
        
        new_pts = pts + step_size * forces
        new_pts = _to_sphere(new_pts)
        
        new_obj = _fast_covering_radius_proxy(new_pts)
        
        if new_obj < best_obj:
            pts = new_pts
            if new_obj < best_obj:
                best_obj = new_obj
                best_pts = pts.copy()
            step_size = min(step_size * 1.02, 0.2)
        else:
            step_size *= 0.95
        
        if step_size < 0.005:
            step_size = 0.005
    
    # Final exact evaluation
    final_obj = _covering_radius(pts)
    if final_obj < best_obj:
        best_obj = final_obj
        best_pts = pts.copy()
    
    return best_pts, best_obj

def _high_precision_polish(pts, max_time=20.0):
    """
    High-precision polish using exact covering radius with small perturbations.
    """
    pts = _to_sphere(pts.copy())
    current_obj = _covering_radius(pts)
    best_pts = pts.copy()
    best_obj = current_obj
    
    start = time.time()
    step_size = 0.01
    
    while time.time() - start < max_time:
        improved = False
        for trial in range(30):
            if time.time() - start >= max_time:
                break
            
            noise = np.random.randn(N, 3) * step_size
            for i in range(N):
                noise[i] -= np.dot(noise[i], pts[i]) * pts[i]
            
            new_pts = _to_sphere(pts + noise)
            new_obj = _covering_radius(new_pts)
            
            if new_obj < best_obj:
                best_obj = new_obj
                best_pts = new_pts.copy()
                pts = new_pts.copy()
                current_obj = new_obj
                improved = True
        
        if not improved:
            step_size *= 0.7
            if step_size < 0.0001:
                break
        else:
            step_size = min(step_size * 1.1, 0.05)
    
    return best_pts, best_obj

def _simulated_annealing(pts, max_time=15.0):
    """
    Simulated annealing using fast proxy for speed.
    """
    pts = _to_sphere(pts.copy())
    current_obj = _fast_covering_radius_proxy(pts)
    best_pts = pts.copy()
    best_obj = _covering_radius(pts)
    
    start = time.time()
    temperature = 0.05
    cooling = 0.999
    step_size = 0.1
    
    while time.time() < max_time:
        noise = np.random.randn(N, 3) * step_size
        new_pts = _to_sphere(pts + noise)
        
        new_obj = _fast_covering_radius_proxy(new_pts)
        
        delta = new_obj - current_obj
        if delta < 0 or np.random.random() < np.exp(-delta / temperature):
            pts = new_pts
            current_obj = new_obj
            
            if np.random.random() < 0.1:
                exact_obj = _covering_radius(pts)
                if exact_obj < best_obj:
                    best_obj = exact_obj
                    best_pts = pts.copy()
        
        temperature *= cooling
        if temperature < 0.002:
            temperature = 0.002
            step_size = 0.02
    
    final_obj = _covering_radius(pts)
    if final_obj < best_obj:
        best_obj = final_obj
        best_pts = pts.copy()
    
    return best_pts, best_obj

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _to_sphere(pts)
    
    best_pts = pts.copy()
    best_obj = _covering_radius(pts)
    
    deadline = _START_TIME + 140.0
    remaining = deadline - time.time()
    
    if remaining > 10:
        pts, obj = _local_optimize(pts, max_time=min(remaining - 10, 40))
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
    
    remaining = deadline - time.time()
    if remaining > 15:
        np.random.seed(42)
        for restart in range(10):
            if time.time() > deadline - 15:
                break
            
            perturbation = 0.2 + 0.15 * restart
            perturbed = _to_sphere(best_pts + np.random.randn(N, 3) * perturbation)
            pts, obj = _simulated_annealing(perturbed, max_time=min(remaining - 15, 12))
            
            if obj < best_obj:
                best_obj = obj
                best_pts = pts.copy()
            
            remaining = deadline - time.time()
            if remaining > 10:
                pts, obj = _high_precision_polish(pts, max_time=min(remaining - 10, 8))
                if obj < best_obj:
                    best_obj = obj
                    best_pts = pts.copy()
    
    remaining = deadline - time.time()
    if remaining > 5:
        pts, obj = _high_precision_polish(best_pts, max_time=min(remaining - 5, 25))
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
    
    best_pts = _to_sphere(best_pts)
    
    dots = best_pts @ best_pts.T
    np.fill_diagonal(dots, 0)
    if np.max(np.abs(dots)) > 0.999999:
        best_pts = _to_sphere(best_pts + np.random.randn(N, 3) * 0.01)
    
    return best_pts
# EVOLVE-BLOCK-END
