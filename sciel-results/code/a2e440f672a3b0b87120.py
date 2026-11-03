"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _min_triangle_area(points):
    """
    Compute the minimum triangle area formed by any 3 points.
    points: (N, 2) array.
    Returns: min area (float).
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            d_ij = pj - pi
            diffs = points - pi
            crosses = d_ij[0] * diffs[:, 1] - d_ij[1] * diffs[:, 0]
            areas = 0.5 * np.abs(crosses)
            areas[i] = float('inf')
            areas[j] = float('inf')
            current_min = np.min(areas)
            if current_min < min_area:
                min_area = current_min
    return min_area


def _bary_to_cart(bary):
    """Convert barycentric coordinates (N,3) to cartesian (N,2)."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = np.sqrt(3) / 2 * bary[:, 2]
    return np.column_stack((x, y))


def _cart_to_bary(cart):
    """Convert cartesian (N,2) to barycentric (N,3)."""
    x = cart[:, 0]
    y = cart[:, 1]
    w2 = y / (np.sqrt(3) / 2)
    w1 = x - 0.5 * w2
    w0 = 1.0 - w1 - w2
    return np.column_stack((w0, w1, w2))


def _clamp_bary(bary):
    """Project barycentric coordinates to the simplex (sum=1, all >= 0)."""
    bary = np.maximum(bary, 0)
    s = bary.sum(axis=1, keepdims=True)
    s = np.maximum(s, 1e-15)
    bary = bary / s
    return bary


def _softmin_objective(bary_flat, beta=50.0):
    """
    Smooth approximation of -min_triangle_area using soft-min.
    bary_flat: flat array of length 3*N (barycentric coords).
    Returns: scalar to minimize.
    """
    bary = bary_flat.reshape(N, 3)
    bary = _clamp_bary(bary)
    pts = _bary_to_cart(bary)
    
    # Compute all triangle areas
    n = N
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                cross = (pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) - \
                        (pts[j, 1] - pts[i, 1]) * (pts[k, 0] - pts[i, 0])
                area = 0.5 * abs(cross)
                areas.append(area)
    areas = np.array(areas)
    
    # Soft-min: approximates min(areas) smoothly
    # soft_min = -logsumexp(-beta * areas) / beta
    # For numerical stability:
    m = np.min(areas)
    if m < 1e-15:
        return 1.0  # Penalty for degenerate triangles
    
    log_sum = np.log(np.sum(np.exp(-beta * (areas - m))))
    soft_min = m - np.log(log_sum) / beta
    
    return -soft_min


def _objective_bary(bary_flat, beta=50.0):
    """
    Objective in barycentric coordinates.
    Returns negative soft-min area (to maximize min area).
    """
    bary = bary_flat.reshape(N, 3)
    bary = _clamp_bary(bary)
    pts = _bary_to_cart(bary)
    
    # Compute all triangle areas
    n = N
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                cross = (pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) - \
                        (pts[j, 1] - pts[i, 1]) * (pts[k, 0] - pts[i, 0])
                area = 0.5 * abs(cross)
                areas.append(area)
    areas = np.array(areas)
    
    # Soft-min
    m = np.min(areas)
    if m < 1e-15:
        return 1.0
    
    log_sum = np.log(np.sum(np.exp(-beta * (areas - m))))
    soft_min = m - np.log(log_sum) / beta
    
    return -soft_min


def _make_initial_bary_grid():
    """Generate a structured grid of barycentric initial points."""
    # Use a 4x4x4 grid on the simplex, pick 11 well-spread points
    candidates = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if i + j + k == 3:
                    candidates.append([i/3.0, j/3.0, k/3.0])
    # That gives 10 points, add center
    candidates.append([1/3, 1/3, 1/3])
    return np.array(candidates)


def _make_initial_bary_random(seed=42):
    """Generate random barycentric initial points."""
    rng = np.random.default_rng(seed)
    bary = rng.random((N, 3))
    bary /= bary.sum(axis=1, keepdims=True)
    return bary


def _make_initial_bary_perturb(base_bary, sigma=0.1, seed=123):
    """Perturb base barycentric points."""
    rng = np.random.default_rng(seed)
    perturbed = base_bary + rng.normal(0, sigma, size=base_bary.shape)
    return _clamp_bary(perturbed)


def _nm_objective(bary_flat, beta=100.0):
    """Nelder-Mead objective with higher beta for sharper min approximation."""
    bary = bary_flat.reshape(N, 3)
    bary = _clamp_bary(bary)
    pts = _bary_to_cart(bary)
    
    n = N
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                cross = (pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) - \
                        (pts[j, 1] - pts[i, 1]) * (pts[k, 0] - pts[i, 0])
                area = 0.5 * abs(cross)
                areas.append(area)
    areas = np.array(areas)
    
    m = np.min(areas)
    if m < 1e-15:
        return 1.0
    
    log_sum = np.log(np.sum(np.exp(-beta * (areas - m))))
    soft_min = m - np.log(log_sum) / beta
    
    return -soft_min


def _nelder_mead_optimize(initial_bary, beta=100.0, maxiter=3000, deadline=None):
    """Run Nelder-Mead optimization on barycentric coordinates."""
    x0 = initial_bary.flatten()
    
    try:
        result = minimize(
            _nm_objective,
            x0,
            args=(beta,),
            method='Nelder-Mead',
            options={
                'maxiter': maxiter,
                'xatol': 1e-10,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        
        if result.fun < 0:
            bary_opt = _clamp_bary(result.x.reshape(N, 3))
            return bary_opt, -result.fun
        return initial_bary, 0.0
    except Exception:
        return initial_bary, 0.0


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_bary = None
    best_score = -float('inf')
    
    # Generate multiple initial barycentric configurations
    inits = []
    
    # Grid-based init
    inits.append(_make_initial_bary_grid())
    
    # Random inits with different seeds
    for seed in [42, 123, 456, 789, 1011]:
        inits.append(_make_initial_bary_random(seed=seed))
    
    # Symmetric structured init: 3 vertices + 3 edge midpoints + 5 inner
    bary_verts = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    bary_edges = np.array([
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])
    center = np.array([1/3, 1/3, 1/3])
    # 4 points around center in barycentric
    inner = [center]
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        # Small offset in barycentric space
        dx = 0.1 * np.cos(angle)
        dy = 0.1 * np.sin(angle)
        # Map to barycentric: perturb w1 and w2
        w1 = 1/3 + dx * 0.8
        w2 = 1/3 + dy * 0.8
        w0 = 1 - w1 - w2
        if w0 > 0 and w1 > 0 and w2 > 0:
            inner.append([w0, w1, w2])
        else:
            # Fallback: just use center with small perturbation
            inner.append([1/3 + 0.05*np.cos(angle), 1/3 + 0.05*np.sin(angle), 1/3 - 0.05*(np.cos(angle)+np.sin(angle))])
    inner = _clamp_bary(np.array(inner[:5]))
    
    symmetric_init = np.vstack([bary_verts, bary_edges, inner])
    inits.append(symmetric_init)
    
    # Evaluate all inits
    for init_bary in inits:
        init_bary = _clamp_bary(init_bary)
        pts = _bary_to_cart(init_bary)
        score = _min_triangle_area(pts)
        if score > best_score:
            best_score = score
            best_bary = init_bary.copy()
    
    # If no valid init, use random
    if best_bary is None:
        best_bary = _make_initial_bary_random(seed=42)
        best_score = _min_triangle_area(_bary_to_cart(best_bary))
    
    # Phase 1: L-BFGS-B with soft-min (beta=50) from multiple starts
    if time.time() < deadline - 30:
        starts = [best_bary]
        # Add perturbed versions
        for seed in [42, 123, 456]:
            starts.append(_make_initial_bary_perturb(best_bary, sigma=0.08, seed=seed))
        # Add some random starts
        for seed in [789, 1011, 1213]:
            starts.append(_make_initial_bary_random(seed=seed))
        
        for s_bary in starts:
            if time.time() > deadline - 30:
                break
            s_bary = _clamp_bary(s_bary)
            x0 = s_bary.flatten()
            
            try:
                result = minimize(
                    _objective_bary,
                    x0,
                    args=(50.0,),
                    method='L-BFGS-B',
                    options={
                        'maxiter': 2000,
                        'ftol': 1e-12,
                        'gtol': 1e-10
                    }
                )
                
                if result.fun < 0:
                    bary_opt = _clamp_bary(result.x.reshape(N, 3))
                    pts = _bary_to_cart(bary_opt)
                    score = _min_triangle_area(pts)
                    if score > best_score:
                        best_score = score
                        best_bary = bary_opt.copy()
            except Exception:
                pass
    
    # Phase 2: Nelder-Mead with higher beta (beta=100) from best and perturbations
    if time.time() < deadline - 15:
        nm_starts = [best_bary]
        for seed in [42, 123, 456, 789, 1011, 1213, 1415]:
            nm_starts.append(_make_initial_bary_perturb(best_bary, sigma=0.05, seed=seed))
        
        for s_bary in nm_starts:
            if time.time() > deadline - 15:
                break
            s_bary = _clamp_bary(s_bary)
            bary_opt, score = _nelder_mead_optimize(s_bary, beta=100.0, maxiter=2000, deadline=deadline)
            if score > best_score:
                best_score = score
                best_bary = bary_opt.copy()
    
    # Phase 3: Final polish with very high beta (beta=200)
    if time.time() < deadline - 5:
        bary_opt, score = _nelder_mead_optimize(best_bary, beta=200.0, maxiter=3000, deadline=deadline)
        if score > best_score:
            best_score = score
            best_bary = bary_opt.copy()
    
    # Convert to cartesian and return
    best_points = _bary_to_cart(best_bary)
    
    # Safety: verify all points are inside triangle
    tol = 1e-9
    x = best_points[:, 0]
    y = best_points[:, 1]
    if np.any(y < -tol) or np.any(y > np.sqrt(3) * x + tol) or \
       np.any(y > np.sqrt(3) * (1 - x) + tol) or \
       np.any(x < -tol) or np.any(x > 1 + tol):
        # Fallback: clamp
        for i in range(N):
            xi, yi = best_points[i]
            xi = max(0.0, min(1.0, xi))
            yi = max(0.0, min(np.sqrt(3)/2, yi))
            if yi > np.sqrt(3) * xi:
                yi = np.sqrt(3) * xi
            if yi > np.sqrt(3) * (1 - xi):
                yi = np.sqrt(3) * (1 - xi)
            best_points[i] = [xi, yi]
    
    return best_points
# EVOLVE-BLOCK-END
