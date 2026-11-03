import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute all triangle indices
indices = list(itertools.combinations(range(N), 3))
n_tri = len(indices)

def _bary_to_cart(bary):
    """Convert barycentric coordinates (u,v,w) with u+v+w=1 to cartesian."""
    # bary = (u, v, w) where p = u*V0 + v*V1 + w*V2
    # V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
    # x = v*1 + w*0.5, y = w*sqrt(3)/2
    u, v, w = bary[:, 0], bary[:, 1], bary[:, 2]
    x = v + 0.5 * w
    y = np.sqrt(3) / 2 * w
    return np.column_stack([x, y])

def _cart_to_bary(cart):
    """Convert cartesian points to barycentric coordinates."""
    x, y = cart[:, 0], cart[:, 1]
    w = y / (np.sqrt(3) / 2)
    v = x - 0.5 * w
    u = 1 - v - w
    return np.column_stack([u, v, w])

def _clamp_bary(bary):
    """Clamp barycentric coordinates to be non-negative and sum to 1."""
    bary = np.maximum(bary, 0)
    sums = bary.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    bary = bary / sums
    return bary

def _min_triangle_area_vectorized(points):
    """Compute the minimum triangle area among all C(N,3) triangles."""
    areas = np.zeros(n_tri)
    for idx, (i, j, k) in enumerate(indices):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        areas[idx] = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
    return areas.min()

def _objective_bary(x):
    """
    Objective in barycentric space: -min_area + penalty for invalid bary coords.
    x is flattened 3N barycentric coordinates.
    """
    bary = x.reshape(N, 3)
    # Clamp to valid barycentric
    bary_clamped = _clamp_bary(bary)
    # Penalty for deviation from valid barycentric
    penalty = np.sum((bary - bary_clamped) ** 2)
    
    points = _bary_to_cart(bary_clamped)
    min_area = _min_triangle_area_vectorized(points)
    
    return -min_area + 100.0 * penalty

def _make_structured_starts():
    """Generate diverse structured starting points in barycentric space."""
    starts = []
    
    # Start 1: vertices + midpoints + center + quarter points
    bary1 = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],  # vertices
        [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],  # midpoints
        [1/3, 1/3, 1/3],  # center
        [0.75, 0.25, 0], [0.25, 0.75, 0],  # quarter points on edge
        [0.75, 0, 0.25], [0.25, 0, 0.75],  # quarter points on another edge
    ])
    starts.append(bary1)
    
    # Start 2: More symmetric arrangement
    bary2 = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],  # vertices
        [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],  # midpoints
        [1/3, 1/3, 1/3],  # center
        [0.2, 0.4, 0.4], [0.4, 0.2, 0.4], [0.4, 0.4, 0.2],  # inner triangle
        [0.1, 0.45, 0.45],  # near vertex
    ])
    starts.append(bary2)
    
    # Start 3: Grid-like
    bary3 = np.array([
        [0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9],  # near vertices
        [0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5],  # inner
        [0.33, 0.33, 0.34],  # center
        [0.15, 0.45, 0.4], [0.4, 0.15, 0.45], [0.45, 0.4, 0.15],  # asymmetric
        [0.2, 0.2, 0.6],
    ])
    starts.append(bary3)
    
    # Random starts
    rng = np.random.RandomState(42)
    for _ in range(5):
        bary = rng.dirichlet(np.ones(N) * 3, size=N)
        starts.append(bary)
    
    return starts

def _local_optimize_lbfgs(x0, deadline):
    """Optimize using L-BFGS-B in barycentric space."""
    try:
        res = minimize(
            _objective_bary,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_nm(x0, deadline):
    """Optimize using Nelder-Mead in barycentric space."""
    try:
        res = minimize(
            _objective_bary,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0

def _refine_with_multiple_methods(x0, deadline):
    """Try multiple optimization methods and return the best result."""
    best_x = x0
    best_val = _objective_bary(x0)
    
    # L-BFGS-B
    x_lbfgs = _local_optimize_lbfgs(x0, deadline)
    val_lbfgs = _objective_bary(x_lbfgs)
    if val_lbfgs < best_val:
        best_val = val_lbfgs
        best_x = x_lbfgs
    
    # Nelder-Mead from L-BFGS result
    if time.time() < deadline - 5:
        x_nm = _local_optimize_nm(x_lbfgs, deadline)
        val_nm = _objective_bary(x_nm)
        if val_nm < best_val:
            best_val = val_nm
            best_x = x_nm
    
    # Nelder-Mead from original
    if time.time() < deadline - 5:
        x_nm2 = _local_optimize_nm(x0, deadline)
        val_nm2 = _objective_bary(x_nm2)
        if val_nm2 < best_val:
            best_val = val_nm2
            best_x = x_nm2
    
    return best_x

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin within 150s
    
    best_bary = None
    best_score = -float('inf')
    
    starts = _make_structured_starts()
    
    for i, bary_start in enumerate(starts):
        if time.time() > deadline - 10:
            break
        
        # Flatten for optimization
        x0 = bary_start.flatten()
        
        # Optimize
        x_opt = _refine_with_multiple_methods(x0, deadline)
        
        # Convert back and evaluate
        bary_opt = x_opt.reshape(N, 3)
        bary_opt = _clamp_bary(bary_opt)
        points_opt = _bary_to_cart(bary_opt)
        
        # Verify points are in triangle
        valid = True
        for p in points_opt:
            b = _cart_to_bary(p.reshape(1, 2))[0]
            if b.min() < -1e-6:
                valid = False
                break
        
        if valid:
            score = _min_triangle_area_vectorized(points_opt)
            if score > best_score:
                best_score = score
                best_bary = bary_opt.copy()
    
    # If we have time, do additional refinement from the best point
    if best_bary is not None and time.time() < deadline - 15:
        x_best = best_bary.flatten()
        x_final = _refine_with_multiple_methods(x_best, deadline)
        bary_final = x_final.reshape(N, 3)
        bary_final = _clamp_bary(bary_final)
        points_final = _bary_to_cart(bary_final)
        
        valid = True
        for p in points_final:
            b = _cart_to_bary(p.reshape(1, 2))[0]
            if b.min() < -1e-6:
                valid = False
                break
        
        if valid:
            score = _min_triangle_area_vectorized(points_final)
            if score > best_score:
                best_score = score
                best_bary = bary_final.copy()
    
    # Fallback
    if best_bary is None:
        bary_fallback = np.array([
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
            [1/3, 1/3, 1/3],
            [0.75, 0.25, 0], [0.25, 0.75, 0],
            [0.75, 0, 0.25], [0.25, 0, 0.75],
        ])
        best_bary = bary_fallback
    
    return _bary_to_cart(best_bary)
