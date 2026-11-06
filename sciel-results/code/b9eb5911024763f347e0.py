"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _covering_radius(points):
    """Compute covering radius as max angular distance to nearest point."""
    # Use convex hull facets: covering radius = max circumradius of facets
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # fallback
    max_r = 0.0
    for simplex in hull.simplices:
        facet_pts = points[simplex]
        # Circumcenter of the facet (in 3D, on the plane)
        # For a triangle, circumcenter is equidistant from all 3 vertices
        a, b, c = facet_pts
        # Circumcenter formula
        d = 2.0 * (a[0]*(b[1]*c[2]-c[1]*b[2]) - b[0]*(a[1]*c[2]-c[1]*a[2]) + c[0]*(a[1]*b[2]-b[1]*a[2]))
        if abs(d) < 1e-12:
            continue
        ux = ((a[0]**2)*(b[1]*c[2]-c[1]*b[2]) - (b[0]**2)*(a[1]*c[2]-c[1]*a[2]) + (c[0]**2)*(a[1]*b[2]-b[1]*a[2])) / d
        uy = ((a[1]**2)*(c[0]*b[2]-b[0]*c[2]) - (b[1]**2)*(c[0]*a[2]-a[0]*c[2]) + (c[1]**2)*(b[0]*a[2]-a[0]*b[2])) / d
        uz = ((a[2]**2)*(b[0]*c[1]-c[0]*b[1]) - (b[2]**2)*(a[0]*c[1]-c[0]*a[1]) + (c[2]**2)*(a[0]*b[1]-b[0]*a[1])) / d
        center = np.array([ux, uy, uz])
        # Project to unit sphere
        cn = np.linalg.norm(center)
        if cn < 1e-12:
            continue
        center = center / cn
        # Angular distance from center to facet vertices
        dists = np.arccos(np.clip(np.abs(np.dot(center, facet_pts)), -1.0, 1.0))
        # The circumradius is the max angular distance
        r = np.max(dists)
        if r > max_r:
            max_r = r
    return max_r

def _repulsive_forces(points, k=1.0):
    """Compute repulsive forces between all point pairs."""
    n = len(points)
    forces = np.zeros_like(points)
    for i in range(n):
        for j in range(i+1, n):
            d = points[i] - points[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                # Random nudge
                d = np.random.randn(3)
                dist = np.linalg.norm(d)
            force = k * d / (dist**3)
            forces[i] += force
            forces[j] -= force
    return forces

def _project_to_sphere(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return points / norms

def _refine_facets(points, n_iters=50, step=0.01):
    """Targeted refinement: move points to reduce largest facet circumradius."""
    for _ in range(n_iters):
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        # Find facet with largest circumradius
        max_r = 0.0
        max_simplex = None
        for simplex in hull.simplices:
            facet_pts = points[simplex]
            a, b, c = facet_pts
            d = 2.0 * (a[0]*(b[1]*c[2]-c[1]*b[2]) - b[0]*(a[1]*c[2]-c[1]*a[2]) + c[0]*(a[1]*b[2]-b[1]*a[2]))
            if abs(d) < 1e-12:
                continue
            ux = ((a[0]**2)*(b[1]*c[2]-c[1]*b[2]) - (b[0]**2)*(a[1]*c[2]-c[1]*a[2]) + (c[0]**2)*(a[1]*b[2]-b[1]*a[2])) / d
            uy = ((a[1]**2)*(c[0]*b[2]-b[0]*c[2]) - (b[1]**2)*(c[0]*a[2]-a[0]*c[2]) + (c[1]**2)*(b[0]*a[2]-a[0]*b[2])) / d
            uz = ((a[2]**2)*(b[0]*c[1]-c[0]*b[1]) - (b[2]**2)*(a[0]*c[1]-c[0]*a[1]) + (c[2]**2)*(a[0]*b[1]-b[0]*a[1])) / d
            center = np.array([ux, uy, uz])
            cn = np.linalg.norm(center)
            if cn < 1e-12:
                continue
            center = center / cn
            dists = np.arccos(np.clip(np.abs(np.dot(center, facet_pts)), -1.0, 1.0))
            r = np.max(dists)
            if r > max_r:
                max_r = r
                max_simplex = simplex
        if max_simplex is None:
            break
        # Move points in the worst facet outward from center
        facet_pts = points[max_simplex]
        center = np.mean(facet_pts, axis=0)
        cn = np.linalg.norm(center)
        if cn > 1e-12:
            center = center / cn
        # Push each point in the facet away from the center direction
        for idx in max_simplex:
            v = points[idx]
            # Component along center
            proj = np.dot(v, center)
            if proj > 0:
                # Move slightly toward center to reduce circumradius
                points[idx] = v - step * center * (1.0 - proj)
                points[idx] = points[idx] / np.linalg.norm(points[idx])
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Phase 1: Repulsive forces to spread points
    k = 0.5
    for iteration in range(200):
        if time.time() > deadline - 30.0:
            break
        forces = _repulsive_forces(points, k)
        points = points + 0.001 * forces
        points = _project_to_sphere(points)
        
        current_radius = _covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # Decay repulsion strength
        k *= 0.98
    
    # Phase 2: Targeted facet refinement
    for _ in range(10):
        if time.time() > deadline - 15.0:
            break
        points = _refine_facets(best_points, n_iters=30, step=0.005)
        current_radius = _covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    # Phase 3: Fine-tuning with smaller steps
    for step in [0.002, 0.001, 0.0005]:
        if time.time() > deadline - 5.0:
            break
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            points = _refine_facets(best_points, n_iters=10, step=step)
            current_radius = _covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
    
    return best_points
# EVOLVE-BLOCK-END
