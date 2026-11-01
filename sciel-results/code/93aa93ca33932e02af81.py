"""
Optimized 50-point spherical covering using differentiable Voronoi-based optimization.

Approach:
- Represent configuration as 50 points on the unit sphere.
- For each point, compute its Voronoi cell inradius (max distance to cell vertices).
- Cell vertices are circumcenters of Delaunay triangles (triples of neighbors).
- Minimize sum of squared inradii using gradient-based optimization.
- Use smooth approximations to handle non-differentiability at cell boundaries.
"""

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _normalize(v):
    """Normalize vectors to unit length."""
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return v / norms


def _cell_vertices(points, i, neighbors):
    """
    Compute Voronoi cell vertices for point i given its neighbors.
    
    For each triple of neighbors (j, k, l), the cell vertex is the intersection
    of three bisecting planes:
    - Plane between i and j: (p_i - p_j) · x = 0
    - Plane between i and k: (p_i - p_k) · x = 0
    - Plane between i and l: (p_i - p_l) · x = 0
    
    The solution is the cross product of two difference vectors, normalized.
    """
    pi = points[i]
    vertices = []
    n = len(neighbors)
    
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                pj = points[neighbors[a]]
                pk = points[neighbors[b]]
                pl = points[neighbors[c]]
                
                # Difference vectors defining the bisecting planes
                v1 = pi - pj
                v2 = pi - pk
                v3 = pi - pl
                
                # Intersection of planes v1·x=0, v2·x=0, v3·x=0
                # Solution: x = cross(v1, v2) (if not degenerate)
                x = np.cross(v1, v2)
                norm_x = np.linalg.norm(x)
                
                if norm_x < 1e-10:
                    # Degenerate: try other pairs
                    x = np.cross(v1, v3)
                    norm_x = np.linalg.norm(x)
                    if norm_x < 1e-10:
                        x = np.cross(v2, v3)
                        norm_x = np.linalg.norm(x)
                        if norm_x < 1e-10:
                            continue
                
                x = x / norm_x
                
                # Check if this vertex is actually in the Voronoi cell
                # It must satisfy: x·pi >= x·pj, x·pi >= x·pk, x·pi >= x·pl
                # And for all other neighbors m: x·pi >= x·pm
                dot_pi = np.dot(x, pi)
                valid = True
                for m in neighbors:
                    pm = points[m]
                    if dot_pi < np.dot(x, pm) - 1e-12:
                        valid = False
                        break
                
                if valid:
                    vertices.append(x)
    
    return vertices


def _voronoi_inradii(points, k_neighbors=12):
    """
    Compute the inradius (covering radius contribution) for each point.
    
    For each point i, find its k nearest neighbors. Compute all Voronoi cell
    vertices from triples of neighbors. The inradius is the maximum angular
    distance from i to any cell vertex.
    
    Uses soft-max for differentiability.
    """
    n = len(points)
    # Compute pairwise angular distances
    cos_dists = np.clip(points @ points.T, -1.0, 1.0)
    dists = np.arccos(cos_dists)
    np.fill_diagonal(dists, 0.0)
    
    inradii = np.zeros(n)
    for i in range(n):
        # Get k nearest neighbors (excluding self)
        idx = np.argsort(dists[i])[:k_neighbors]
        
        # Get cell vertices
        vertices = _cell_vertices(points, i, idx)
        
        if len(vertices) == 0:
            # Fallback: use max pairwise distance to neighbors
            inradii[i] = np.max(dists[i, idx])
        else:
            vertices = np.array(vertices)
            # Angular distance from i to each vertex
            cos_angles = np.clip(points[i] @ vertices.T, -1.0, 1.0)
            angles = np.arccos(cos_angles)
            
            # Use soft-max for differentiable max
            temp = 0.05
            inradii[i] = temp * np.log(np.sum(np.exp(angles / temp)))
    
    return inradii


def _objective(x):
    """
    Objective: sum of squared Voronoi inradii.
    This is differentiable and directly targets the covering radius.
    """
    points = _normalize(x.reshape(N, 3))
    inradii = _voronoi_inradii(points)
    return np.sum(inradii ** 2)


def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    Uses differentiable Voronoi inradius optimization with angular bisection planes.
    """
    start_time = time.time()
    max_time = 110  # Leave margin for safety
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    x0 = points.flatten().copy()
    best_x = x0.copy()
    best_obj = _objective(x0)
    
    # Phase 1: Coarse optimization with L-BFGS-B
    def obj_wrapper(x):
        return _objective(x)
    
    try:
        result = minimize(
            obj_wrapper,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 300, 'maxfun': 8000, 'ftol': 1e-10}
        )
        if result.fun < best_obj:
            best_x = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    if time.time() - start_time > max_time - 30:
        return _normalize(best_x.reshape(N, 3))
    
    # Phase 2: Local search with multiple restarts
    rng = np.random.RandomState(42)
    for restart in range(8):
        # Create perturbation
        scale = 0.15 * (1.0 - restart / 10.0)
        x_perturbed = best_x + rng.randn(N * 3) * scale
        try:
            result = minimize(
                obj_wrapper,
                x_perturbed,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-10}
            )
            if result.fun < best_obj:
                best_x = result.x
                best_obj = result.fun
        except Exception:
            pass
        
        if time.time() - start_time > max_time - 15:
            break
    
    # Phase 3: Final fine-tuning
    if time.time() - start_time < max_time - 5:
        try:
            result = minimize(
                obj_wrapper,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 15000, 'ftol': 1e-12}
            )
            if result.fun < best_obj:
                best_x = result.x
                best_obj = result.fun
        except Exception:
            pass
    
    points = _normalize(best_x.reshape(N, 3))
    
    # Validate: ensure points are on unit sphere and not degenerate
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-6):
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return points

# EVOLVE-BLOCK-END
