"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _compute_covering_radius(points):
    """Compute the covering radius (in degrees) for points on unit sphere."""
    try:
        hull = ConvexHull(points)
        max_cos = 1.0
        for simplex in hull.simplices:
            verts = points[simplex]
            if len(verts) < 3:
                continue
            v1, v2, v3 = verts[0], verts[1], verts[2]
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            dist = np.abs(np.dot(normal, v1))
            dist = np.clip(dist, -1.0, 1.0)
            max_cos = min(max_cos, dist)
        
        if max_cos > 1.0:
            max_cos = 1.0
        if max_cos < -1.0:
            max_cos = -1.0
        
        radius_rad = np.arccos(max_cos)
        return np.degrees(radius_rad)
    except Exception:
        return 180.0


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points


def _compute_facet_data(points):
    """Compute facet normals, circumradii, and centers for the convex hull."""
    try:
        hull = ConvexHull(points)
        n_facets = len(hull.simplices)
        facet_normals = np.zeros((n_facets, 3))
        facet_circumradii = np.zeros(n_facets)
        facet_centers = np.zeros((n_facets, 3))
        
        for i, simplex in enumerate(hull.simplices):
            verts = points[simplex]
            v1, v2, v3 = verts[0], verts[1], verts[2]
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            facet_normals[i] = normal
            
            # Circumradius in angular space
            dist = np.abs(np.dot(normal, v1))
            dist = np.clip(dist, -1.0, 1.0)
            radius_rad = np.arccos(dist)
            facet_circumradii[i] = np.degrees(radius_rad)
            
            # Facet center (circumcenter on sphere)
            facet_centers[i] = normal
        
        return facet_normals, facet_circumradii, facet_centers, hull
    except Exception:
        return None, None, None, None


def _facet_force_optimize(points, max_iterations=500, lr_init=0.01, lr_decay=0.999, time_limit=100.0):
    """Optimize using facet-based forces that push points away from large facets."""
    start_time = time.time()
    best_points = points.copy()
    best_radius = _compute_covering_radius(best_points)
    
    lr = lr_init
    for it in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        facet_normals, facet_circumradii, facet_centers, hull = _compute_facet_data(points)
        if facet_normals is None:
            break
        
        # Compute forces: each point is pushed by the facets it belongs to
        # Force direction: away from the facet center (toward the point)
        # Force magnitude: proportional to the circumradius of the facet
        forces = np.zeros_like(points)
        
        for i, simplex in enumerate(hull.simplices):
            if facet_circumradii[i] <= 0:
                continue
            # Weight by how much larger this facet is compared to average
            avg_radius = np.mean(facet_circumradii)
            weight = max(0, (facet_circumradii[i] - avg_radius) / (avg_radius + 1e-10))
            
            # Push each vertex of this facet away from the facet center
            center = facet_centers[i]
            for idx in simplex:
                diff = points[idx] - center
                norm = np.linalg.norm(diff)
                if norm > 1e-10:
                    forces[idx] += weight * diff / norm
        
        # Add a small repulsion between all points to prevent coincidences
        for i in range(N):
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 0.1:
                    repulse = 0.1 / (dist * dist + 1e-6)
                    if dist > 1e-10:
                        forces[i] += repulse * diff / dist
                        forces[j] -= repulse * diff / dist
        
        # Normalize forces to be tangent to the sphere
        for i in range(N):
            normal = points[i]
            normal = normal / np.linalg.norm(normal)
            forces[i] -= np.dot(forces[i], normal) * normal
        
        # Apply forces with adaptive learning rate
        new_points = points.copy()
        for i in range(N):
            force_norm = np.linalg.norm(forces[i])
            if force_norm > 1e-10:
                new_points[i] = points[i] + lr * forces[i] / force_norm
                new_points[i] /= np.linalg.norm(new_points[i])
        
        # Check convergence
        if np.max(np.linalg.norm(new_points - points, axis=1)) < 1e-10:
            points = new_points
            break
        
        points = new_points
        lr *= lr_decay
        
        current_radius = _compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    return best_points, best_radius


def _nelder_mead_polish(points, time_limit=30.0):
    """Polish the configuration using Nelder-Mead on the covering radius."""
    start_time = time.time()
    best_points = points.copy()
    best_radius = _compute_covering_radius(best_points)
    
    # Flatten points to 1D
    x = points.flatten()
    
    # Create initial simplex
    alpha = 0.01
    simplex = [x.copy()]
    for i in range(len(x)):
        v = x.copy()
        v[i] += alpha
        simplex.append(v)
    
    def objective(flat_x):
        pts = flat_x.reshape(N, 3)
        # Renormalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
        return _compute_covering_radius(pts)
    
    n_iter = 0
    max_iter = 200
    while n_iter < max_iter:
        if time.time() - start_time > time_limit:
            break
        
        n_iter += 1
        # Evaluate all vertices
        fvals = [objective(v) for v in simplex]
        
        # Sort by objective
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if fvals[0] - fvals[-1] < 1e-8:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_reflect = centroid + (centroid - simplex[-1])
        f_reflect = objective(x_reflect)
        
        if f_reflect < fvals[0]:
            # Expansion
            x_expand = centroid + 2.0 * (centroid - simplex[-1])
            f_expand = objective(x_expand)
            if f_expand < f_reflect:
                simplex[-1] = x_expand
                fvals[-1] = f_expand
            else:
                simplex[-1] = x_reflect
                fvals[-1] = f_reflect
        elif f_reflect < fvals[-2]:
            simplex[-1] = x_reflect
            fvals[-1] = f_reflect
        else:
            # Contraction
            x_contract = centroid + 0.5 * (simplex[-1] - centroid)
            f_contract = objective(x_contract)
            if f_contract < fvals[-1]:
                simplex[-1] = x_contract
                fvals[-1] = f_contract
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                fvals[1:] = [objective(v) for v in simplex[1:]]
        
        # Update best
        min_f = min(fvals)
        if min_f < best_radius:
            best_radius = min_f
            best_points = simplex[np.argmin(fvals)].reshape(N, 3)
            best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    best_points = None
    best_radius = 180.0
    
    # Multiple restarts with different initializations
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Different initializations for each restart
        if restart == 0:
            points = _fibonacci_spiral()
        elif restart == 1:
            points = _fibonacci_spiral()
            points += np.random.randn(N, 3) * 0.05
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            points = np.random.randn(N, 3)
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            points = _fibonacci_spiral()
            points += np.random.randn(N, 3) * 0.1
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 4:
            # Random with some structure
            i = np.arange(N)
            z = 1.0 - 2.0 * i / N
            phi = np.random.uniform(0, 2 * np.pi, N)
            r = np.sqrt(1.0 - z * z)
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        elif restart == 5:
            points = _fibonacci_spiral()
            # Rotate
            theta = np.random.uniform(0, 2 * np.pi)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rot = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
            points = points @ rot.T
        elif restart == 6:
            points = np.random.randn(N, 3)
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        else:
            points = _fibonacci_spiral()
            points += np.random.randn(N, 3) * 0.02
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Phase 1: Facet-based optimization
        remaining_time = deadline - time.time()
        if remaining_time < 10:
            break
        
        facet_time = min(60.0, remaining_time * 0.6)
        points, radius = _facet_force_optimize(points, max_iterations=300, time_limit=facet_time)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
        
        # Phase 2: Nelder-Mead polish
        remaining_time = deadline - time.time()
        if remaining_time < 15:
            break
        
        nm_time = min(25.0, remaining_time * 0.5)
        points, radius = _nelder_mead_polish(points, time_limit=nm_time)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
        
        # Phase 3: Final facet-based refinement
        remaining_time = deadline - time.time()
        if remaining_time < 10:
            break
        
        facet_time = min(20.0, remaining_time * 0.8)
        points, radius = _facet_force_optimize(points, max_iterations=100, lr_init=0.005, time_limit=facet_time)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
    
    if best_points is None:
        best_points = _fibonacci_spiral()
    
    return best_points
# EVOLVE-BLOCK-END
