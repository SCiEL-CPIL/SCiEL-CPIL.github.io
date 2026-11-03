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
            if len(verts) == 3:
                v1, v2, v3 = verts
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
            else:
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

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points

def _compute_facet_data(points):
    """Compute facet normals, circumradii, and which points define each facet."""
    try:
        hull = ConvexHull(points)
        n_facets = len(hull.simplices)
        facet_normals = np.zeros((n_facets, 3))
        facet_circumradii = np.zeros(n_facets)
        facet_points = []  # list of point indices per facet
        
        for idx, simplex in enumerate(hull.simplices):
            verts = points[simplex]
            if len(verts) < 3:
                continue
            v1, v2, v3 = verts[0], verts[1], verts[2]
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                facet_normals[idx] = v1
                facet_circumradii[idx] = 1.0
                facet_points.append(list(simplex))
                continue
            normal = normal / norm
            # Ensure normal points outward
            if np.dot(normal, v1) < 0:
                normal = -normal
            facet_normals[idx] = normal
            dist = np.abs(np.dot(normal, v1))
            dist = np.clip(dist, 0.0, 1.0)
            circumradius_rad = np.arccos(dist)
            facet_circumradii[idx] = np.degrees(circumradius_rad)
            facet_points.append(list(simplex))
        
        return facet_normals, facet_circumradii, facet_points, hull
    except Exception:
        return None, None, None, None

def _facet_based_optimize(points, max_iters=300, lr=0.01, deadline=None):
    """Optimize points using facet-based force balance.
    
    Each point is moved based on the circumradii of facets it belongs to.
    Points in large-circumradius facets are pulled toward the facet center;
    points in small-circumradius facets are pushed away to equalize.
    """
    P = points.copy()
    best_P = P.copy()
    best_radius = _compute_covering_radius(P)
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute current facet data
        facet_normals, facet_circumradii, facet_points, hull = _compute_facet_data(P)
        if facet_normals is None:
            break
        
        # Track which facets each point belongs to
        point_facets = [[] for _ in range(N)]
        for idx, pts in enumerate(facet_points):
            for p_idx in pts:
                point_facets[p_idx].append(idx)
        
        # Compute forces on each point
        forces = np.zeros((N, 3))
        
        for i in range(N):
            if not point_facets[i]:
                continue
            
            p = P[i]
            force = np.zeros(3)
            
            for f_idx in point_facets[i]:
                facet_normal = facet_normals[f_idx]
                circumradius = facet_circumradii[f_idx]
                
                # Target: all facets should have similar circumradius
                # If this facet has larger than average circumradius, pull point toward facet center
                # If smaller, push away
                
                # Facet center direction (from origin)
                center_dir = facet_normal / (np.linalg.norm(facet_normal) + 1e-12)
                
                # Weight by how much this facet deviates from the mean
                # We want to equalize: points in large facets move toward center,
                # points in small facets move away
                
                # Compute a "pressure" based on circumradius relative to current max
                max_cr = np.max(facet_circumradii)
                if max_cr < 1e-6:
                    continue
                
                # Normalized deviation: positive if this facet is larger than average
                avg_cr = np.mean(facet_circumradii)
                deviation = circumradius - avg_cr
                
                # Force direction: toward facet center if deviation > 0, away if < 0
                # Scale by deviation
                force += deviation * center_dir
            
            # Also add a small repulsion from other nearby points to avoid coincident points
            # This helps escape degenerate configurations
            for j in range(N):
                if i == j:
                    continue
                d = np.dot(P[i], P[j])
                if d > 0.999:  # Very close points
                    force -= 0.1 * P[j]
            
            # Project force to tangent plane
            dot_fp = np.dot(force, p)
            force_tangent = force - dot_fp * p
            
            forces[i] = force_tangent
        
        # Update points
        P_new = P + lr * forces
        P_new = P_new / np.linalg.norm(P_new, axis=1, keepdims=True)
        
        # Check for improvements
        new_radius = _compute_covering_radius(P_new)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_P = P_new.copy()
            P = P_new
        else:
            # Adaptive learning rate: reduce if no improvement
            lr *= 0.95
        
        # Early stopping if convergence
        if lr < 1e-6:
            break
    
    return best_P, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Multiple restarts with different initializations
    best_points = None
    best_radius = 180.0
    
    # Initial configurations to try
    inits = [
        _fibonacci_spiral(N),
    ]
    
    # Add some perturbed versions
    rng = np.random.RandomState(42)
    for _ in range(3):
        perturbed = inits[0].copy()
        noise = rng.randn(N, 3) * 0.05
        perturbed = perturbed + noise
        perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
        inits.append(perturbed)
    
    for init_idx, init_points in enumerate(inits):
        if time.time() > deadline - 10:  # Need time for at least one more optimization
            break
        
        # Run facet-based optimization
        optimized, radius = _facet_based_optimize(
            init_points, 
            max_iters=200, 
            lr=0.02,
            deadline=deadline - 5
        )
        
        if radius < best_radius:
            best_radius = radius
            best_points = optimized.copy()
    
    # Final polish with smaller learning rate if time permits
    if best_points is not None and time.time() < deadline - 15:
        polished, radius = _facet_based_optimize(
            best_points,
            max_iters=100,
            lr=0.005,
            deadline=deadline - 3
        )
        if radius < best_radius:
            best_radius = radius
            best_points = polished.copy()
    
    if best_points is None:
        best_points = inits[0]
    
    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
