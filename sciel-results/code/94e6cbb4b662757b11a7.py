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


def _voronoi_centroid(points, idx):
    """Compute the centroid of the Voronoi cell for point idx on the sphere.
    
    The Voronoi cell on the sphere corresponds to the normal cone of the convex hull
    vertex. We approximate it by finding all hull facets containing this vertex,
    computing the circumcenter (in angular space) of each facet, and averaging.
    """
    try:
        hull = ConvexHull(points)
        # Find all facets containing this vertex
        facet_indices = []
        for i, simplex in enumerate(hull.simplices):
            if idx in simplex:
                facet_indices.append(i)
        
        if not facet_indices:
            return points[idx]
        
        # Collect circumcenters (in Cartesian space, then project to sphere)
        circumcenters = []
        for fi in facet_indices:
            verts = points[hull.simplices[fi]]
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
            # The circumcenter of the facet in angular terms is along the normal
            # projected onto the sphere
            cc = normal
            cc_norm = np.linalg.norm(cc)
            if cc_norm < 1e-12:
                continue
            cc = cc / cc_norm
            circumcenters.append(cc)
        
        if not circumcenters:
            return points[idx]
        
        # Average the circumcenters (spherical mean approximation)
        mean_cc = np.mean(circumcenters, axis=0)
        mean_norm = np.linalg.norm(mean_cc)
        if mean_norm < 1e-12:
            return points[idx]
        return mean_cc / mean_norm
    except Exception:
        return points[idx]


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(best_points)
    
    # Lloyd's algorithm: iteratively move each point to the centroid of its Voronoi cell
    # This equalizes cell areas and can find deeper local minima for covering radius
    
    # Multiple restarts with different initializations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Different initializations for each restart
        if restart == 0:
            # Fibonacci spiral
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        elif restart == 1:
            # Slightly perturbed Fibonacci
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + np.random.randn(N) * 0.05
            r = np.sqrt(1.0 - z * z)
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            # Random points
            points = np.random.randn(N, 3)
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            # Fibonacci with different golden ratio offset
            i = np.arange(N) + 0.3
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + 1.0
            r = np.sqrt(1.0 - z * z)
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        else:
            # More random
            points = np.random.randn(N, 3)
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        current_radius = _compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # Lloyd iterations
        n_iterations = 200
        for it in range(n_iterations):
            if time.time() > deadline:
                break
            
            # Compute new positions as Voronoi centroids
            new_points = np.zeros_like(points)
            for idx in range(N):
                new_points[idx] = _voronoi_centroid(points, idx)
            
            # Renormalize to unit sphere
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            new_points = new_points / norms
            
            # Check for convergence
            if np.max(np.linalg.norm(new_points - points, axis=1)) < 1e-8:
                points = new_points
                break
            
            points = new_points
            
            current_radius = _compute_covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
        
        # After Lloyd, do a few gradient steps to fine-tune
        lr = 0.005
        for fine_it in range(50):
            if time.time() > deadline:
                break
            
            new_points = points.copy()
            for idx in range(N):
                forces = np.zeros(3)
                for jdx in range(N):
                    if jdx == idx:
                        continue
                    diff = points[idx] - points[jdx]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-10:
                        forces += np.random.randn(3) * 0.01
                    else:
                        force_mag = 1.0 / (dist * dist + 0.01)
                        forces += force_mag * diff / dist
                
                force_norm = np.linalg.norm(forces)
                if force_norm > 1e-10:
                    new_points[idx] = points[idx] + lr * forces / force_norm
                    new_points[idx] /= np.linalg.norm(new_points[idx])
            
            points = new_points
            current_radius = _compute_covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
    
    return best_points
# EVOLVE-BLOCK-END
