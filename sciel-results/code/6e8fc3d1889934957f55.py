"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _to_sphere(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius_fast(pts):
    """
    Compute the covering radius (in radians) via convex hull facet circumradii.
    Returns (radius, worst_circumcenter, worst_simplex_indices) or (pi, None, None) if invalid.
    """
    pts = _to_sphere(pts)
    
    # Check for coincident points
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    if np.min(d2[np.triu_indices(len(pts), k=1)]) < 1e-14:
        return np.pi, None, None
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, None, None
    
    if len(hull.simplices) == 0:
        return np.pi, None, None
    
    max_r = 0.0
    worst_c = None
    worst_simplex = None
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Compute normal to the facet
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            r = np.pi
            cc = np.zeros(3)
        else:
            n = n / nn
            best_r = np.pi
            best_c = n
            for s in (1.0, -1.0):
                p = s * n
                da = np.arccos(np.clip(np.dot(p, a), -1.0, 1.0))
                db = np.arccos(np.clip(np.dot(p, b), -1.0, 1.0))
                dc = np.arccos(np.clip(np.dot(p, c), -1.0, 1.0))
                if abs(da - db) < 1e-4 and abs(da - dc) < 1e-4:
                    if da < best_r:
                        best_r = da
                        best_c = p
            r = best_r
            cc = best_c
        
        if r > max_r:
            max_r = r
            worst_c = cc
            worst_simplex = simplex
    
    return max_r, worst_c, worst_simplex

def _lloyd_relax(pts, max_iter=100, deadline=None):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation).
    Moves each point to the centroid of its Voronoi cell (average of incident facet normals).
    """
    pts = _to_sphere(pts.copy())
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Accumulate normals for each vertex
        targets = np.zeros_like(pts)
        counts = np.zeros(len(pts))
        
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n = n / nn
            # Ensure normal points outward
            if np.dot(n, a) < 0:
                n = -n
            
            for idx in simplex:
                targets[idx] += n
                counts[idx] += 1
        
        # Update points
        improved = False
        for idx in range(len(pts)):
            if counts[idx] > 0:
                target = targets[idx] / np.linalg.norm(targets[idx])
                # Damped update to ensure stability
                new_pt = 0.5 * pts[idx] + 0.5 * target
                new_pt = _to_sphere(new_pt.reshape(1, 3))[0]
                if np.linalg.norm(new_pt - pts[idx]) > 1e-8:
                    pts[idx] = new_pt
                    improved = True
        
        if not improved:
            break
            
    return pts

def _greedy_refine(pts, max_iter=200, deadline=None):
    """
    Greedy refinement: iteratively shrinks the worst facet or moves the nearest point to the worst circumcenter.
    """
    pts = _to_sphere(pts.copy())
    best_cr, worst_c, worst_simplex = _covering_radius_fast(pts)
    
    if worst_c is None:
        return pts, best_cr

    step_size = 0.05
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
            
        if worst_c is None:
            break
            
        improved = False
        
        # Strategy 1: Move vertices of the worst facet towards the circumcenter
        if worst_simplex is not None:
            cand = pts.copy()
            for idx in worst_simplex:
                # Move towards circumcenter
                new_pt = cand[idx] + step_size * worst_c
                new_pt = _to_sphere(new_pt.reshape(1, 3))[0]
                cand[idx] = new_pt
            
            cr, wc, ws = _covering_radius_fast(cand)
            if cr < best_cr - 1e-9:
                pts = cand
                best_cr = cr
                worst_c = wc
                worst_simplex = ws
                step_size = min(step_size * 1.05, 0.2)
                improved = True
        
        # Strategy 2: Move the point nearest to the worst circumcenter towards it
        if not improved:
            d = np.sum((pts - worst_c[None, :]) ** 2, axis=1)
            k = int(np.argmin(d))
            
            cand = pts.copy()
            new_pt = cand[k] + step_size * worst_c
            new_pt = _to_sphere(new_pt.reshape(1, 3))[0]
            cand[k] = new_pt
            
            cr, wc, ws = _covering_radius_fast(cand)
            if cr < best_cr - 1e-9:
                pts = cand
                best_cr = cr
                worst_c = wc
                worst_simplex = ws
                step_size = min(step_size * 1.05, 0.2)
                improved = True
        
        if not improved:
            step_size *= 0.8
            if step_size < 1e-5:
                break
                
    return pts, best_cr

def _simulated_annealing(pts, max_iter=500, T0=0.1, T_end=0.001, deadline=None):
    """
    Simulated annealing with targeted perturbations.
    """
    current_pts = _to_sphere(pts.copy())
    current_val, worst_c, _ = _covering_radius_fast(current_pts)
    best_pts = current_pts.copy()
    best_val = current_val
    
    T = T0
    cooling = (T_end / T0) ** (1.0 / max_iter)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        T *= cooling
        
        # Perturb a random point
        idx = np.random.randint(len(current_pts))
        
        # Bias direction towards the worst circumcenter if available
        if worst_c is not None and np.random.random() < 0.5:
            direction = worst_c + 0.5 * np.random.randn(3)
        else:
            direction = np.random.randn(3)
            
        direction = direction / np.linalg.norm(direction)
        
        new_pts = current_pts.copy()
        new_pts[idx] = current_pts[idx] + 0.05 * direction
        new_pts[idx] = _to_sphere(new_pts[idx].reshape(1, 3))[0]
        
        new_val, wc, _ = _covering_radius_fast(new_pts)
        
        delta = new_val - current_val
        if delta < 0 or (T > 1e-6 and np.random.random() < np.exp(-delta / T)):
            current_pts = new_pts
            current_val = new_val
            worst_c = wc
            if current_val < best_val:
                best_pts = current_pts.copy()
                best_val = current_val
                
    return best_pts, best_val

def _fibonacci_spiral(n, offset=0.0):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=None):
    """Generate n random points on the sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    return _to_sphere(pts)

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    _deadline = time.time() + 140.0  # Safety margin
    
    np.random.seed(42)
    
    best_pts = None
    best_val = np.pi
    
    # Multiple starting configurations
    starts = [_fibonacci_spiral(N)]
    for seed in [1, 2, 3, 4, 5]:
        starts.append(_random_points(N, seed=seed))
    
    for st in starts:
        if time.time() > _deadline - 30.0:
            break
        
        # 1. Lloyd's Relaxation (Global balance)
        lloyd_pts = _lloyd_relax(st, max_iter=100, deadline=_deadline)
        
        # 2. Greedy Refinement (Local optimization of worst facets)
        ref_pts, ref_val = _greedy_refine(lloyd_pts, max_iter=300, deadline=_deadline)
        
        # 3. Simulated Annealing (Escape local minima)
        sa_pts, sa_val = _simulated_annealing(ref_pts, max_iter=500, T0=0.08, T_end=0.001, deadline=_deadline)
        
        # 4. Final Greedy Refinement
        final_pts, final_val = _greedy_refine(sa_pts, max_iter=200, deadline=_deadline)
        
        if final_val < best_val:
            best_val = final_val
            best_pts = final_pts.copy()
            
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        
    return _to_sphere(best_pts)
# EVOLVE-BLOCK-END
