"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Precompute all C(11,3) = 165 triples
_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_N_TRIPLES = _TRIPLES.shape[0]


def _min_triangle_area_fast(points):
    """Fully vectorized min triangle area for N=11 points."""
    p1 = points[_TRIPLES[:, 0]]
    p2 = points[_TRIPLES[:, 1]]
    p3 = points[_TRIPLES[:, 2]]
    d1 = p2 - p1
    d2 = p3 - p1
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _clip_to_triangle(points):
    """Clip points to be inside the equilateral triangle."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    
    y = np.clip(y, 0, H)
    x_min = y / S3
    x_max = 1 - y / S3
    x = np.clip(x, x_min, x_max)
    x = np.clip(x, 0, 1)
    
    points[:, 0] = x
    points[:, 1] = y
    return points


def _bary_to_cart(bary):
    """Convert barycentric coordinates (a, b, c) to Cartesian.
    point = b*(1,0) + c*(0.5, H)
    so x = b + 0.5*c, y = H*c
    """
    b = bary[:, 1]
    c = bary[:, 2]
    x = b + 0.5 * c
    y = H * c
    return np.column_stack([x, y])


def _cart_to_bary(points):
    """Convert Cartesian to barycentric coordinates."""
    x = points[:, 0]
    y = points[:, 1]
    c = y / H
    b = x - 0.5 * c
    a = 1 - b - c
    return np.column_stack([a, b, c])


def _structured_inits():
    """Generate diverse structured initial configurations."""
    inits = []
    
    # Symmetric configuration near corners and center
    base_bary = np.array([
        [0.02, 0.02, 0.96],
        [0.96, 0.02, 0.02],
        [0.02, 0.96, 0.02],
        [0.45, 0.02, 0.53],
        [0.02, 0.45, 0.53],
        [0.53, 0.45, 0.02],
        [0.33, 0.33, 0.34],
        [0.25, 0.55, 0.20],
        [0.55, 0.20, 0.25],
        [0.20, 0.25, 0.55],
        [0.45, 0.25, 0.30],
    ])
    inits.append(_bary_to_cart(base_bary))
    
    # Three rows
    row_bary = np.array([
        [0.05, 0.05, 0.90],
        [0.35, 0.05, 0.60],
        [0.65, 0.05, 0.30],
        [0.95, 0.05, 0.00],
        [0.15, 0.35, 0.50],
        [0.45, 0.35, 0.20],
        [0.75, 0.35, -0.10],
        [0.25, 0.65, 0.10],
        [0.55, 0.65, -0.20],
        [0.35, 0.55, 0.10],
        [0.65, 0.55, -0.20],
    ])
    row_bary[:, 2] = 1 - row_bary[:, 0] - row_bary[:, 1]
    inits.append(_bary_to_cart(row_bary))
    
    # Concentric pattern
    cent_bary = np.array([
        [0.33, 0.33, 0.34],
        [0.10, 0.40, 0.50],
        [0.50, 0.40, 0.10],
        [0.40, 0.10, 0.50],
        [0.90, 0.10, 0.00],
        [0.10, 0.90, 0.00],
        [0.50, 0.05, 0.45],
        [0.20, 0.55, 0.25],
        [0.55, 0.25, 0.20],
        [0.25, 0.20, 0.55],
        [0.45, 0.30, 0.25],
    ])
    inits.append(_bary_to_cart(cent_bary))
    
    # Grid-based
    grid_bary = np.array([
        [0.05, 0.05, 0.90],
        [0.30, 0.05, 0.65],
        [0.55, 0.05, 0.40],
        [0.80, 0.05, 0.15],
        [0.15, 0.30, 0.55],
        [0.40, 0.30, 0.30],
        [0.65, 0.30, 0.05],
        [0.25, 0.55, 0.20],
        [0.50, 0.55, -0.05],
        [0.35, 0.45, 0.20],
        [0.60, 0.45, -0.05],
    ])
    grid_bary[:, 2] = 1 - grid_bary[:, 0] - grid_bary[:, 1]
    inits.append(_bary_to_cart(grid_bary))
    
    # More symmetric patterns
    sym_bary = np.array([
        [0.03, 0.03, 0.94],
        [0.94, 0.03, 0.03],
        [0.03, 0.94, 0.03],
        [0.47, 0.03, 0.50],
        [0.03, 0.47, 0.50],
        [0.50, 0.47, 0.03],
        [0.33, 0.33, 0.34],
        [0.22, 0.53, 0.25],
        [0.53, 0.22, 0.25],
        [0.25, 0.22, 0.53],
        [0.42, 0.28, 0.30],
    ])
    inits.append(_bary_to_cart(sym_bary))
    
    # Hexagonal-like pattern
    hex_bary = np.array([
        [0.33, 0.33, 0.34],
        [0.15, 0.45, 0.40],
        [0.55, 0.45, 0.00],
        [0.45, 0.15, 0.40],
        [0.15, 0.15, 0.70],
        [0.70, 0.15, 0.15],
        [0.45, 0.70, -0.15],
        [0.25, 0.25, 0.50],
        [0.50, 0.25, 0.25],
        [0.25, 0.50, 0.25],
        [0.50, 0.50, -0.00],
    ])
    hex_bary[:, 2] = 1 - hex_bary[:, 0] - hex_bary[:, 1]
    inits.append(_bary_to_cart(hex_bary))
    
    return inits


def _objective_cartesian(flat):
    """Objective for Nelder-Mead: minimize negative min triangle area with penalty."""
    points = flat.reshape(N, 2)
    
    x = points[:, 0]
    y = points[:, 1]
    
    # Penalty for being outside the triangle
    eps = 1e-8
    penalty = 0.0
    penalty += np.sum(np.maximum(0, -y - eps)) * 100
    penalty += np.sum(np.maximum(0, y - S3 * x + eps)) * 100
    penalty += np.sum(np.maximum(0, y - S3 * (1 - x) + eps)) * 100
    
    min_area = _min_triangle_area_fast(points)
    return -min_area + penalty


def _objective_barycentric(flat):
    """Objective in barycentric space: minimize negative min triangle area."""
    bary = flat.reshape(N, 3)
    # Normalize to ensure they sum to 1
    bary = bary / bary.sum(axis=1, keepdims=True)
    # Clip small negatives
    bary = np.clip(bary, 0, None)
    bary = bary / bary.sum(axis=1, keepdims=True)
    
    points = _bary_to_cart(bary)
    min_area = _min_triangle_area_fast(points)
    return -min_area


def _refine_from(points, deadline, method='nelder'):
    """Refine a configuration using multi-start optimization."""
    best_points = points.copy()
    best_score = _min_triangle_area_fast(points)
    
    rng = np.random.RandomState(42)
    
    # Try multiple perturbations and refinements
    for attempt in range(6):
        if time.time() > deadline - 5:
            break
        
        if attempt == 0:
            start = best_points.copy()
        else:
            # Random perturbation of current best
            scale = 0.05 / (1 + 0.5 * attempt)
            perturbation = rng.randn(N, 2) * scale
            start = best_points + perturbation
            start = _clip_to_triangle(start)
        
        try:
            if method == 'nelder':
                res = minimize(
                    _objective_cartesian,
                    start.flatten(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
                if res.fun < 0:
                    candidate = res.x.reshape(N, 2)
                    candidate = _clip_to_triangle(candidate)
                    score = _min_triangle_area_fast(candidate)
                    if score > best_score:
                        best_score = score
                        best_points = candidate.copy()
                        
            elif method == 'bary':
                bary_start = _cart_to_bary(start)
                res = minimize(
                    _objective_barycentric,
                    bary_start.flatten(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
                if res.fun < 0:
                    bary_candidate = res.x.reshape(N, 3)
                    bary_candidate = np.clip(bary_candidate, 0, None)
                    bary_candidate = bary_candidate / bary_candidate.sum(axis=1, keepdims=True)
                    candidate = _bary_to_cart(bary_candidate)
                    score = _min_triangle_area_fast(candidate)
                    if score > best_score:
                        best_score = score
                        best_points = candidate.copy()
                        
        except Exception:
            continue
    
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_points = None
    best_score = -np.inf
    
    # Generate structured initial configurations
    inits = _structured_inits()
    
    # For each initial configuration, refine with multi-start Nelder-Mead
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        init_pts = _clip_to_triangle(init_pts)
        init_score = _min_triangle_area_fast(init_pts)
        
        # Refine this configuration
        refined_pts, refined_score = _refine_from(init_pts, deadline, method='nelder')
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()
    
    # If we have time, do additional random restarts
    if best_points is not None and time.time() < deadline - 15:
        rng = np.random.RandomState(123)
        for restart in range(8):
            if time.time() > deadline - 5:
                break
            
            # Random configuration
            bary = rng.dirichlet(np.ones(N), size=N)
            random_pts = _bary_to_cart(bary)
            random_pts = _clip_to_triangle(random_pts)
            
            refined_pts, refined_score = _refine_from(random_pts, deadline, method='nelder')
            
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_pts.copy()
    
    # Second pass with barycentric optimization if time permits
    if best_points is not None and time.time() < deadline - 10:
        refined_pts, refined_score = _refine_from(best_points, deadline, method='bary')
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()
    
    # Final fallback
    if best_points is None:
        best_points = _clip_to_triangle(inits[0])
    
    # Ensure all points are valid
    best_points = _clip_to_triangle(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
