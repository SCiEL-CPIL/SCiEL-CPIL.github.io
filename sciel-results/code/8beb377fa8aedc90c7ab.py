"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

# ---------- Core Geometry ----------

def _min_triangle_area_vec(points):
    """
    Optimized minimum triangle area calculation.
    """
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    for i in range(n - 2):
        xi, yi = x[i], y[i]
        for j in range(i + 1, n - 1):
            v1_x = x[j] - xi
            v1_y = y[j] - yi
            v2_x = x[j+1:] - xi
            v2_y = y[j+1:] - yi
            cross = v1_x * v2_y - v1_y * v2_x
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area

def _hull_area_fast(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Score is min_triangle_area / hull_area.
    """
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _objective_direct(flat_points):
    """
    Objective for minimization: HullArea / MinTriangleArea.
    Lower is better.
    """
    points = flat_points.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    
    min_area = _min_triangle_area_vec(points)
    if min_area < 1e-12:
        return 1e10
        
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
        
    return hull_area / min_area

def _normalize_unit_area(points):
    """Scale points so that convex hull area is 1.0."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area > 1e-10:
            scale = np.sqrt(1.0 / hull_area)
            return points * scale
    except Exception:
        pass
    # Fallback: scale by bounding box
    xs = points[:, 0]
    ys = points[:, 1]
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        scale = 1.0 / np.sqrt(w * h)
        return points * scale
    return points

# ---------- Initialization Strategies ----------

def _init_regular(n):
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return pts

def _init_mixed(n, outer_n=10, inner_scale=0.5):
    """
    Create a configuration with points on an outer ring and an inner ring.
    This often provides a better starting point than a single ring for maximizing min triangle area.
    """
    outer_n = min(outer_n, n)
    inner_n = n - outer_n
    
    # Outer ring
    angles_out = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    
    # Inner ring
    if inner_n > 0:
        angles_in = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
        inner = inner_scale * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
        pts = np.vstack([outer, inner])
    else:
        pts = outer
        
    return pts

def _init_lattice(n):
    """
    Generate points based on a deformed hexagonal lattice structure.
    """
    # Create a grid of points
    # For N=14, a 4x4 grid has 16 points, we can take 14.
    # Or a hexagonal packing.
    # Let's try a simple 4x4 grid with one removed, or 3x5 with one removed.
    # 4x4 grid:
    x = np.arange(4)
    y = np.arange(4)
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack([X.flatten(), Y.flatten()])
    
    # Center and scale
    pts -= pts.mean(axis=0)
    
    # Take first n points? No, better to take a compact subset.
    # Let's just use the first n points of the flattened grid, 
    # but ensure they are reasonably distributed.
    # Actually, a regular polygon is a safer "lattice-like" start for N=14 if we don't know the optimal lattice.
    # Let's stick to regular and mixed for reliability, but add a perturbed regular.
    return pts[:n] if len(pts) >= n else _init_regular(n)

def _generate_inits():
    inits = []
    
    # 1. Regular 14-gon
    inits.append(_init_regular(N))
    
    # 2. Mixed configurations (varying inner/outer split and scale)
    inits.append(_init_mixed(N, outer_n=10, inner_scale=0.5))
    inits.append(_init_mixed(N, outer_n=12, inner_scale=0.3))
    inits.append(_init_mixed(N, outer_n=14, inner_scale=0.0)) # Just regular
    
    # 3. Perturbed regular polygons
    rng = np.random.default_rng(42)
    base = _init_regular(N)
    for scale in [0.1, 0.2, 0.3]:
        noise = rng.normal(0, scale, size=base.shape)
        inits.append(base + noise)
        
    # 4. Perturbed mixed
    base_mixed = _init_mixed(N, outer_n=10, inner_scale=0.5)
    noise = rng.normal(0, 0.1, size=base_mixed.shape)
    inits.append(base_mixed + noise)

    return inits

# ---------- Main ----------

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_score = 0.0
    best_pts = None
    best_obj = float('inf')
    
    def _consider(points):
        nonlocal best_score, best_pts, best_obj
        if points is None or points.shape != (N, 2) or np.any(~np.isfinite(points)):
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_pts = points.copy()
            best_obj = 1.0 / s if s > 0 else float('inf')

    # Phase 1: Evaluate initial configurations
    inits = _generate_inits()
    for init in inits:
        _consider(init)
        
    # Phase 2: Global Search with Differential Evolution
    # Bounds: Points should stay within a reasonable box, e.g., [-2, 2]
    # Since the objective is scale invariant, we can restrict the search space.
    # If points spread out too much, hull area increases, min area increases, ratio stays same.
    # But numerically, very large/small numbers are bad.
    # Let's bound to [-2, 2].
    bounds = [(-2.0, 2.0)] * (N * 2)
    
    # DE is expensive. We run it if we have time.
    # max_iter and popsize need to be balanced.
    # For 28 dimensions, popsize=20, max_iter=100 is ~5600 function evals.
    # _objective_direct is fast (vectorized min area).
    
    if time.time() < deadline - 30.0:
        try:
            res = differential_evolution(
                _objective_direct, 
                bounds, 
                maxiter=150, 
                popsize=20, 
                tol=1e-8, 
                atol=1e-10,
                seed=42, 
                polish=False, # We do our own polish
                init='latinhypercube',
                update_interval=1,
                callback=None
            )
            if res.x is not None and res.fun < 1e9:
                cand = res.x.reshape(N, 2)
                _consider(cand)
        except Exception as e:
            pass

    # Phase 3: Local Refinement with Nelder-Mead
    # We run NM on the best point found so far, and potentially on the best initial points
    # if DE didn't improve much.
    
    candidates_for_nm = []
    if best_pts is not None:
        candidates_for_nm.append(best_pts)
        
    # Also add the best initial point if it's different from current best
    for init in inits[:3]: # Top 3 inits
        if best_pts is None or not np.allclose(init, best_pts, atol=1e-4):
            candidates_for_nm.append(init)

    for cand in candidates_for_nm:
        if time.time() > deadline - 20.0:
            break
            
        x0 = cand.flatten()
        try:
            # Nelder-Mead on the hard objective
            res = minimize(
                _objective_direct, 
                x0, 
                method='Nelder-Mead',
                options={
                    'maxiter': 3000, 
                    'xatol': 1e-8, 
                    'fatol': 1e-10, 
                    'adaptive': True
                }
            )
            if res.x is not None and res.fun < 1e9:
                cand_pts = res.x.reshape(N, 2)
                _consider(cand_pts)
        except Exception as e:
            pass

    # Phase 4: Final Polish
    # A very tight NM pass on the absolute best found
    if best_pts is not None and time.time() < deadline - 5.0:
        x0 = best_pts.flatten()
        try:
            res = minimize(
                _objective_direct, 
                x0, 
                method='Nelder-Mead',
                options={
                    'maxiter': 2000, 
                    'xatol': 1e-10, 
                    'fatol': 1e-12, 
                    'adaptive': True
                }
            )
            if res.x is not None and res.fun < 1e9:
                cand_pts = res.x.reshape(N, 2)
                _consider(cand_pts)
        except Exception as e:
            pass

    # Fallback
    if best_pts is None:
        best_pts = _init_regular(N)

    # Normalize to unit area
    final_points = _normalize_unit_area(best_pts)
    
    # Safety check
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-10:
            final_points = _normalize_unit_area(_init_regular(N))
    except Exception:
        final_points = _normalize_unit_area(_init_regular(N))

    return final_points
# EVOLVE-BLOCK-END
