"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

def _compute_covering_radius(points):
    """Compute the exact covering radius in degrees using Voronoi vertices of convex hull."""
    try:
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        hull = ConvexHull(points)
        
        voronoi_vertices = []
        for face in hull.simplices:
            a = points[face[0]]
            b = points[face[1]]
            c = points[face[2]]
            
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            
            d = np.dot(n, a)
            t = d / (n_norm ** 2)
            o = t * n
            o_norm = np.linalg.norm(o)
            if o_norm < 1e-10:
                continue
            o = o / o_norm
            voronoi_vertices.append(o)
        
        if not voronoi_vertices:
            return 180.0
        
        voronoi_vertices = np.array(voronoi_vertices)
        
        max_dist = 0.0
        for p in points:
            dots = np.dot(voronoi_vertices, p)
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            min_dist = np.min(dists)
            if min_dist > max_dist:
                max_dist = min_dist
        
        return np.degrees(max_dist)
    except Exception:
        return 180.0


def _compute_facet_data(points):
    """
    Compute circumcenters and circumradii for each triangular facet of the convex hull.
    Returns:
        circumcenters: array of shape (n_facets, 3) - normalized circumcenter directions
        circumradii: array of shape (n_facets,) - angular circumradius for each facet
        face_vertices: list of (i, j, k) vertex indices for each facet
    """
    try:
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        hull = ConvexHull(points)
        
        circumcenters = []
        circumradii = []
        face_vertices = []
        
        for face in hull.simplices:
            i, j, k = face
            a = points[i]
            b = points[j]
            c = points[k]
            
            # Compute circumcenter of triangle abc on the sphere
            # The circumcenter is the point equidistant from a, b, c
            # For points on unit sphere, circumcenter = normalize(a + b + c - (a·b)*b/|a-b|^2 ...) 
            # Better: solve the linear system for the circumcenter
            # Circumcenter o satisfies: |o-a|^2 = |o-b|^2 = |o-c|^2
            # => 2*o·(a-b) = |a|^2 - |b|^2 = 0 (since all on unit sphere)
            # => o·(a-b) = 0 and o·(a-c) = 0
            # So o is perpendicular to both (a-b) and (a-c), i.e., o is parallel to (a-b)×(a-c)
            
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            
            o = n / n_norm
            
            # Verify this is on the correct side (outward normal)
            # The facet should have empty cap, so circumcenter should be in the direction
            # away from the origin through the facet
            # Check: the circumcenter should be such that the cap centered at o 
            # with angular radius arccos(o·a) contains no other points
            
            # Compute circumradius: angular distance from o to a (or b or c)
            dot_val = np.clip(np.dot(o, a), -1.0, 1.0)
            circumradius = np.arccos(dot_val)
            
            circumcenters.append(o)
            circumradii.append(circumradius)
            face_vertices.append((i, j, k))
        
        if not circumcenters:
            return None, None, None
        
        return np.array(circumcenters), np.array(circumradii), face_vertices
    except Exception:
        return None, None, None


def _facet_based_objective(flat_points):
    """
    Objective function for facet-based optimization.
    Returns the maximum circumradius (in radians) plus a penalty for non-uniformity.
    """
    try:
        points = flat_points.reshape(N, 3)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        circumcenters, circumradii, face_vertices = _compute_facet_data(points)
        if circumradii is None:
            return np.pi
        
        max_cr = np.max(circumradii)
        # Add a small penalty for variance to encourage uniformity
        var_penalty = 0.01 * np.var(circumradii)
        
        return max_cr + var_penalty
    except Exception:
        return np.pi


def _facet_gradient(flat_points):
    """
    Numerical gradient for the facet-based objective.
    """
    eps = 1e-6
    grad = np.zeros_like(flat_points)
    base_val = _facet_based_objective(flat_points)
    
    for idx in range(len(flat_points)):
        orig = flat_points[idx]
        flat_points[idx] = orig + eps
        val_plus = _facet_based_objective(flat_points)
        flat_points[idx] = orig - eps
        val_minus = _facet_based_objective(flat_points)
        flat_points[idx] = orig
        grad[idx] = (val_plus - val_minus) / (2 * eps)
    
    return grad


def _fibonacci_spiral(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        np.random.seed(seed)
    pts = np.random.randn(n, 3)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _facet_based_optimize(initial_points, max_iters=300, deadline=None):
    """
    Optimize using facet-based force balance.
    Iteratively compute facet circumradii and move points to equalize them.
    """
    points = initial_points.copy()
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        circumcenters, circumradii, face_vertices = _compute_facet_data(points)
        if circumradii is None:
            break
        
        current_max_cr = np.max(circumradii)
        current_radius = _compute_covering_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # Compute force on each vertex based on adjacent facet circumradii
        forces = np.zeros((N, 3))
        
        for idx, (i, j, k) in enumerate(face_vertices):
            cr = circumradii[idx]
            cc = circumcenters[idx]
            
            # The circumcenter direction indicates where the "worst point" is for this facet
            # Points should move toward the circumcenter if their facet has large circumradius
            # Weight by how much this facet's circumradius exceeds the mean
            weight = cr - np.mean(circumradii)
            
            # Move each vertex of this facet toward the circumcenter direction
            # The direction to move is the component of cc perpendicular to the point
            for v_idx in [i, j, k]:
                p = points[v_idx]
                # Direction toward circumcenter (tangent to sphere)
                move_dir = cc - np.dot(cc, p) * p
                move_dir_norm = np.linalg.norm(move_dir)
                if move_dir_norm > 1e-10:
                    move_dir = move_dir / move_dir_norm
                    forces[v_idx] += weight * move_dir
        
        # Normalize forces and apply with adaptive step size
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.maximum(force_norms, 1e-10)
        normalized_forces = forces / force_norms
        
        # Adaptive step size: smaller when close to optimum
        step_size = 0.05 * np.exp(-0.01 * iteration)
        
        points = points + step_size * normalized_forces
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return best_points, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    best_points = None
    best_radius = float('inf')
    
    # Try multiple initial configurations
    initial_configs = []
    
    # Fibonacci spiral
    initial_configs.append(_fibonacci_spiral(N))
    
    # Random configurations with different seeds
    for seed in [42, 123, 456, 789, 1000, 2000]:
        initial_configs.append(_random_points(N, seed=seed))
    
    # Icosahedral-inspired initial config
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [0, 1, phi_g], [0, -1, phi_g], [0, 1, -phi_g], [0, -1, -phi_g],
        [1, phi_g, 0], [-1, phi_g, 0], [1, -phi_g, 0], [-1, -phi_g, 0],
        [phi_g, 0, 1], [-phi_g, 0, 1], [phi_g, 0, -1], [-phi_g, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    ico_expanded = ico_verts.copy()
    remaining = N - len(ico_verts)
    if remaining > 0:
        extra = _random_points(remaining, seed=999)
        ico_expanded = np.vstack([ico_verts, extra])
    perm = np.random.permutation(N)
    ico_expanded = ico_expanded[perm]
    initial_configs.append(ico_expanded)
    
    # Run facet-based optimization from each initial config
    for init_idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 15:
            break
        
        # Check initial config
        init_radius = _compute_covering_radius(init_pts)
        if init_radius < best_radius:
            best_radius = init_radius
            best_points = init_pts.copy()
        
        # Facet-based optimization
        try:
            opt_pts, opt_radius = _facet_based_optimize(init_pts, max_iters=200, deadline=deadline - 10)
            if opt_radius < best_radius:
                best_radius = opt_radius
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Final polish: try small perturbations around best solution
    if best_points is not None and time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            noise = np.random.normal(0, 0.003, (N, 3))
            perturbed = best_points + noise
            perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
            
            try:
                final_pts, final_radius = _facet_based_optimize(perturbed, max_iters=100, deadline=deadline - 3)
                if final_radius < best_radius:
                    best_radius = final_radius
                    best_points = final_pts.copy()
            except Exception:
                pass
    
    if best_points is None:
        best_points = _fibonacci_spiral(N)
    
    return best_points
# EVOLVE-BLOCK-END
