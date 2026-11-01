"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _make_grid(n_theta=400, n_phi=800):
    """Create a dense spherical grid for evaluating covering radius."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    return grid

_GRID = _make_grid()

def _covering_radius_grid(pts, n_theta=180, n_phi=360):
    """Compute covering radius in degrees using a grid approximation."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    dots = grid @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = np.max(dots, axis=1)
    max_radius_rad = np.arccos(np.min(min_dots))
    return np.degrees(max_radius_rad)

def _covering_radius(pts):
    """Compute the exact covering radius using convex hull facets."""
    if len(pts) < 4:
        return _covering_radius_grid(pts, 360, 720)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return _covering_radius_grid(pts, 360, 720)
    
    max_radius = 0.0
    for simplex in hull.simplices:
        v1 = pts[simplex[0]]
        v2 = pts[simplex[1]]
        v3 = pts[simplex[2]]
        
        e1 = v2 - v1
        e2 = v3 - v1
        n = np.cross(e1, e2)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-15:
            continue
        
        d = np.dot(n, v1)
        dist = abs(d) / n_norm
        
        if dist > 1.0:
            dist = 1.0
        elif dist < -1.0:
            dist = -1.0
        
        angle = np.arccos(min(max(dist, -1.0), 1.0))
        if angle > max_radius:
            max_radius = angle
    
    return np.degrees(max_radius)

def _initial_fib():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _lloyd_iteration(pts, n_theta=360, n_phi=720):
    """One Lloyd iteration: move each point to the centroid of its Voronoi cell."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    dots = grid @ pts.T
    assignments = dots.argmax(axis=1)
    
    new_pts = pts.copy()
    for i in range(len(pts)):
        mask = assignments == i
        if np.sum(mask) > 0:
            centroid = grid[mask].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                new_pts[i] = centroid / norm
    
    return new_pts

def _refine_lloyd(pts, start_time, max_time=120.0, max_iters=30):
    """Lloyd-based refinement with annealing."""
    current = pts.copy()
    best = pts.copy()
    best_radius = _covering_radius_grid(current, 180, 360)
    
    for it in range(max_iters):
        if time.time() - start_time > max_time - 10.0:
            break
        
        new_pts = _lloyd_iteration(current)
        
        # Check for coincident points and nudge apart
        for i in range(N):
            for j in range(i + 1, N):
                dot = np.dot(new_pts[i], new_pts[j])
                if dot > 0.999999:
                    diff = new_pts[i] - new_pts[j]
                    norm = np.linalg.norm(diff)
                    if norm < 1e-10:
                        diff = np.array([1.0, 0.0, 0.0])
                        norm = 1.0
                    diff /= norm
                    new_pts[i] = new_pts[i] + 0.01 * diff
                    new_pts[j] = new_pts[j] - 0.01 * diff
                    new_pts[i] /= np.linalg.norm(new_pts[i])
                    new_pts[j] /= np.linalg.norm(new_pts[j])
        
        new_radius = _covering_radius_grid(new_pts, 180, 360)
        if new_radius < best_radius:
            current = new_pts
            best = new_pts.copy()
            best_radius = new_radius
        else:
            # Try blending
            alpha = 0.5
            blended = (1 - alpha) * current + alpha * new_pts
            norms = np.linalg.norm(blended, axis=1, keepdims=True)
            blended = blended / norms
            blended_radius = _covering_radius_grid(blended, 180, 360)
            if blended_radius < best_radius:
                current = blended
                best = blended.copy()
                best_radius = blended_radius
            else:
                break
    
    return best, best_radius

def _local_optimize(pts, start_time, max_time=120.0):
    """Local optimization using gradient-free moves with fast radius evaluation."""
    best_pts = pts.copy()
    best_radius = _covering_radius_grid(best_pts, 180, 360)
    
    step_size = 0.1
    
    while time.time() - start_time < max_time - 10.0:
        improved = False
        for idx in range(N):
            if time.time() - start_time > max_time - 10.0:
                break
            
            for attempt in range(3):
                delta = np.random.randn(3) * step_size
                new_pt = best_pts[idx] + delta
                norm = np.linalg.norm(new_pt)
                if norm < 1e-10:
                    continue
                new_pt = new_pt / norm
                
                new_pts = best_pts.copy()
                new_pts[idx] = new_pt
                
                new_radius = _covering_radius_grid(new_pts, 180, 360)
                if new_radius < best_radius - 1e-12:
                    best_pts = new_pts
                    best_radius = new_radius
                    improved = True
                    break
        
        if improved:
            step_size = max(step_size * 0.9, 0.001)
        else:
            step_size = max(step_size * 0.5, 0.001)
            if step_size <= 0.001:
                break
    
    return best_pts, best_radius

def _facet_optimize(pts, start_time, max_time=120.0):
    """Optimize by directly minimizing the worst facet circumradius."""
    best_pts = pts.copy()
    best_radius = _covering_radius(best_pts)
    
    for it in range(20):
        if time.time() - start_time > max_time - 10.0:
            break
        
        try:
            hull = ConvexHull(best_pts)
        except Exception:
            break
        
        # Find the worst facet
        worst_idx = -1
        worst_angle = 0.0
        for si, simplex in enumerate(hull.simplices):
            v1 = best_pts[simplex[0]]
            v2 = best_pts[simplex[1]]
            v3 = best_pts[simplex[2]]
            
            e1 = v2 - v1
            e2 = v3 - v1
            n = np.cross(e1, e2)
            n_norm = np.linalg.norm(n)
            
            if n_norm < 1e-15:
                continue
            
            d = np.dot(n, v1)
            dist = abs(d) / n_norm
            dist = min(max(dist, -1.0), 1.0)
            angle = np.arccos(dist)
            
            if angle > worst_angle:
                worst_angle = angle
                worst_idx = si
        
        if worst_idx < 0:
            break
        
        simplex = hull.simplices[worst_idx]
        # Move the points of the worst facet outward slightly
        for idx in simplex:
            direction = best_pts[idx] / np.linalg.norm(best_pts[idx])
            new_pt = best_pts[idx] + 0.01 * direction
            new_pt = new_pt / np.linalg.norm(new_pt)
            best_pts[idx] = new_pt
        
        new_radius = _covering_radius(best_pts)
        if new_radius < best_radius:
            best_radius = new_radius
        else:
            break
    
    return best_pts, best_radius

def construct():
    """Construct 130 points on the unit sphere to minimize the covering radius."""
    start_time = time.time()
    max_time = 120.0
    
    # Start with Fibonacci spiral
    pts = _initial_fib()
    best_pts = pts.copy()
    best_radius = _covering_radius(best_pts)
    
    # Phase 1: Lloyd refinement
    lloyd_pts, lloyd_radius = _refine_lloyd(pts, start_time, max_time)
    if lloyd_radius < best_radius:
        best_pts = lloyd_pts
        best_radius = lloyd_radius
    
    # Phase 2: Local optimization
    remaining = max_time - (time.time() - start_time)
    if remaining > 20.0:
        local_pts, local_radius = _local_optimize(best_pts, start_time, max_time)
        if local_radius < best_radius:
            best_pts = local_pts
            best_radius = local_radius
    
    # Phase 3: Facet-based optimization
    remaining = max_time - (time.time() - start_time)
    if remaining > 15.0:
        facet_pts, facet_radius = _facet_optimize(best_pts, start_time, max_time)
        if facet_radius < best_radius:
            best_pts = facet_pts
            best_radius = facet_radius
    
    # Phase 4: Final Lloyd refinement
    remaining = max_time - (time.time() - start_time)
    if remaining > 10.0:
        final_pts, final_radius = _refine_lloyd(best_pts, start_time, max_time, max_iters=10)
        if final_radius < best_radius:
            best_pts = final_pts
            best_radius = final_radius
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-10:
                best_pts[j] += np.random.randn(3) * 0.001
                best_pts[j] /= np.linalg.norm(best_pts[j])
    
    # Final normalization
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
