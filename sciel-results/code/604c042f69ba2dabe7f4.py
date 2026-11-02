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


def _spherical_circumradius(p1, p2, p3):
    """
    Compute the spherical circumradius of triangle (p1, p2, p3).
    The circumcenter c satisfies c·p1 = c·p2 = c·p3 = cos(R).
    c ∝ cross(p1 - p2, p1 - p3)
    R = arccos(c·p1)
    """
    v1 = p1 - p2
    v2 = p1 - p3
    c = np.cross(v1, v2)
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-12:
        d12 = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
        d13 = np.arccos(np.clip(np.dot(p1, p3), -1, 1))
        d23 = np.arccos(np.clip(np.dot(p2, p3), -1, 1))
        return max(d12, d13, d23) / 2.0
    c = c / norm_c
    cos_r = np.clip(np.dot(c, p1), -1.0, 1.0)
    R = np.arccos(cos_r)
    if R > np.pi / 2:
        c = -c
        cos_r = np.clip(np.dot(c, p1), -1.0, 1.0)
        R = np.arccos(cos_r)
    return R


def _compute_voronoi_inradii(points, k_neighbors=10):
    """
    Compute the inradius (max circumradius of cell vertices) for each point's Voronoi cell.
    Uses soft-max for differentiability.
    """
    n = len(points)
    cos_dists = np.clip(points @ points.T, -1.0, 1.0)
    dists = np.arccos(cos_dists)
    np.fill_diagonal(dists, 0.0)
    
    inradii = np.zeros(n)
    for i in range(n):
        idx = np.argsort(dists[i])[:k_neighbors]
        neighbors = list(idx)
        
        circumradii = []
        for a_idx in range(len(neighbors)):
            for b_idx in range(a_idx + 1, len(neighbors)):
                j = neighbors[a_idx]
                k = neighbors[b_idx]
                cr = _spherical_circumradius(points[i], points[j], points[k])
                circumradii.append(cr)
        
        if len(circumradii) == 0:
            inradii[i] = 0.0
        else:
            circumradii = np.array(circumradii)
            temp = 0.03
            inradii[i] = temp * np.log(np.sum(np.exp(circumradii / temp)))
    
    return inradii


def _objective(x):
    """Objective: sum of squared Voronoi inradii."""
    points = _normalize(x.reshape(N, 3))
    inradii = _compute_voronoi_inradii(points)
    return np.sum(inradii ** 2)


def _fibonacci_init():
    """Fibonacci spiral initialization."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    Uses differentiable Voronoi inradius optimization with multi-phase strategy.
    """
    start_time = time.time()
    max_time = 115
    
    rng = np.random.RandomState(12345)
    
    x0 = _fibonacci_init().flatten().copy()
    best_x = x0.copy()
    best_obj = _objective(x0)
    
    def obj_wrapper(x):
        return _objective(x)
    
    # Phase 1: Coarse optimization
    try:
        result = minimize(
            obj_wrapper, x0, method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 20000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.fun < best_obj:
            best_x = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    if time.time() - start_time > max_time - 40:
        return _normalize(best_x.reshape(N, 3))
    
    # Phase 2: Multiple restarts
    for restart in range(10):
        if time.time() - start_time > max_time - 20:
            break
        scale = 0.1 + 0.05 * restart
        x_perturbed = best_x + rng.randn(N * 3) * scale
        try:
            result = minimize(
                obj_wrapper, x_perturbed, method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < best_obj:
                best_x = result.x
                best_obj = result.fun
        except Exception:
            pass
    
    # Phase 3: Fine-tuning
    if time.time() - start_time < max_time - 10:
        try:
            result = minimize(
                obj_wrapper, best_x, method='L-BFGS-B',
                options={'maxiter': 1000, 'maxfun': 50000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.fun < best_obj:
                best_x = result.x
                best_obj = result.fun
        except Exception:
            pass
    
    # Phase 4: Final local search
    if time.time() - start_time < max_time - 5:
        for _ in range(5):
            if time.time() - start_time > max_time - 3:
                break
            x_perturbed = best_x + rng.randn(N * 3) * 0.01
            try:
                result = minimize(
                    obj_wrapper, x_perturbed, method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if result.fun < best_obj:
                    best_x = result.x
                    best_obj = result.fun
            except Exception:
                pass
    
    points = _normalize(best_x.reshape(N, 3))
    
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-6) or not np.all(np.isfinite(points)):
        points = _fibonacci_init()
    
    return points

# EVOLVE-BLOCK-END
