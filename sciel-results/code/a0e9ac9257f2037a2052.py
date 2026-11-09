# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

def _triangle_areas(pts, tri_idx):
    """Compute all triangle areas vectorized."""
    a = pts[tri_idx[:, 0]]
    b = pts[tri_idx[:, 1]]
    c = pts[tri_idx[:, 2]]
    area = 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )
    return area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _get_tri_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx, dtype=int)

def _compute_score(pts, tri_idx):
    """Compute score given points and precomputed triangle indices."""
    if not np.all(np.isfinite(pts)):
        return 0.0
    areas = _triangle_areas(pts, tri_idx)
    min_area = areas.min()
    hull_area = _hull_area(pts)
    if hull_area <= 1e-9:
        return 0.0
    return min_area / hull_area

def _soft_min_objective(params, tri_idx, n, beta=30.0, hull_target=1.0, hull_weight=0.1):
    """
    Objective function for optimization.
    We maximize the minimum triangle area relative to the hull area.
    Since the score is scale-invariant, we penalize deviation from a target hull area
    to fix the scale, and maximize the soft-min of the areas.
    
    params: flattened (n, 2) coordinates or (n, 2) radial/angular params?
    Here we use free coordinates but initialized from a polygon structure.
    """
    pts = params.reshape(n, 2)
    
    # Check for degeneracy
    if not np.all(np.isfinite(pts)):
        return 1e6
        
    areas = _triangle_areas(pts, tri_idx)
    
    # Soft min of areas
    # To avoid overflow in exp, we shift
    shifted = -beta * areas
    max_shifted = shifted.max()
    # If all areas are 0, soft_min is 0
    if max_shifted < -1000:
        soft_min_val = 0.0
    else:
        soft_min_val = -(max_shifted + np.log(np.exp(shifted - max_shifted).sum())) / beta
    
    # Hull area
    hull_area = _hull_area(pts)
    if hull_area <= 1e-9:
        return 1e6
        
    # We want to maximize soft_min / hull_area.
    # This is equivalent to minimizing -soft_min / hull_area.
    # To make it scale invariant and stable, we can minimize:
    # - (soft_min / hull_area) + hull_weight * (hull_area - hull_target)^2
    # However, dividing by hull_area makes the gradient complex.
    # Alternative: Fix hull area to 1 via penalty, and maximize soft_min.
    # Let's stick to the ratio in the objective but regularize.
    
    score_approx = soft_min_val / hull_area
    penalty = (hull_area - hull_target) ** 2
    
    return -score_approx + hull_weight * penalty

def _regular_polygon_params(n, r=1.0):
    """Generate parameters for a regular polygon (flattened coords)."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    return pts.flatten()

def heilbronn_convex13() -> np.ndarray:
    n = 13
    tri_idx = _get_tri_indices(n)
    
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit
    
    # Initial candidates
    candidates = []
    
    # 1. Regular 13-gon
    reg_pts = _regular_polygon_params(n, r=1.0).reshape(n, 2)
    candidates.append(reg_pts)
    
    # 2. Perturbed regular 13-gon (random radial/angular jitter)
    np.random.seed(42)
    for _ in range(5):
        base = _regular_polygon_params(n, r=1.0).reshape(n, 2)
        # Add small noise
        noise = np.random.normal(0, 0.1, base.shape)
        candidates.append(base + noise)
        
    # 3. Triangular lattice subset (compact)
    # Generate a grid of triangular lattice points and pick the 13 closest to center
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    lattice_pts = []
    R = 3
    for i in range(-R, R+1):
        for j in range(-R, R+1):
            p = i * a1 + j * a2
            lattice_pts.append(p)
    lattice_pts = np.array(lattice_pts)
    lattice_pts = lattice_pts - lattice_pts.mean(axis=0)
    dists = np.linalg.norm(lattice_pts, axis=1)
    sorted_idx = np.argsort(dists)
    lat_pts = lattice_pts[sorted_idx[:n]]
    # Normalize to unit hull area approx
    h_area = _hull_area(lat_pts)
    if h_area > 0:
        lat_pts = lat_pts / np.sqrt(h_area)
    candidates.append(lat_pts)
    
    # Evaluate initial candidates
    best_score = -1.0
    best_pts = None
    for c in candidates:
        s = _compute_score(c, tri_idx)
        if s > best_score:
            best_score = s
            best_pts = c.copy()
            
    # Optimization Phase 1: Global Search (Differential Evolution)
    # Use a subset of the time for DE to explore the landscape
    if time.time() < deadline - 120:
        bounds = [(-2.0, 2.0)] * (n * 2)
        
        # Create a custom init population
        pop_size = 20
        init_pop = np.zeros((pop_size, n * 2))
        init_pop[0] = best_pts.flatten()
        for i in range(1, pop_size):
            # Mix of regular polygon and lattice
            if i % 2 == 0:
                init_pop[i] = _regular_polygon_params(n, r=1.0 + np.random.uniform(-0.2, 0.2))
            else:
                init_pop[i] = best_pts.flatten() + np.random.normal(0, 0.3, n * 2)
                
        try:
            # DE is slow, so we limit iterations
            result = differential_evolution(
                _soft_min_objective, 
                bounds, 
                args=(tri_idx, n),
                seed=42,
                popsize=pop_size,
                maxiter=30, # Keep low to ensure we have time for local refine
                tol=1e-4,
                init=init_pop,
                polish=False,
                workers=1 # Explicitly single thread to avoid overhead
            )
            
            if result.fun < 1e5: # Valid result
                de_pts = result.x.reshape(-1, 2)
                de_score = _compute_score(de_pts, tri_idx)
                if de_score > best_score:
                    best_score = de_score
                    best_pts = de_pts.copy()
        except Exception:
            pass

    # Optimization Phase 2: Local Refinement (L-BFGS-B with Soft Min)
    # Run multiple local optimizations from the best point and its perturbations
    local_starts = [best_pts]
    for _ in range(10):
        perturbed = best_pts + np.random.normal(0, 0.05, best_pts.shape)
        local_starts.append(perturbed)
        
    # Also add starts from the initial candidates if they were good
    for c in candidates[:3]:
        local_starts.append(c)
        
    for start_pts in local_starts:
        if time.time() > deadline - 20:
            break
            
        try:
            # L-BFGS-B minimizes the objective
            # We use a moderate beta for soft-min to balance smoothness and accuracy
            res = minimize(
                _soft_min_objective, 
                start_pts.flatten(), 
                args=(tri_idx, n, 30.0, 1.0, 0.1),
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < 1e5:
                res_pts = res.x.reshape(-1, 2)
                res_score = _compute_score(res_pts, tri_idx)
                if res_score > best_score:
                    best_score = res_score
                    best_pts = res_pts.copy()
        except Exception:
            pass

    # Optimization Phase 3: Coordinate Descent / Nudge
    # Explicitly nudge points to increase the minimum area
    if time.time() < deadline - 10:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            areas = _triangle_areas(best_pts, tri_idx)
            min_area = areas.min()
            # Find triangles with area close to min
            close_tris = np.where(areas < min_area * 1.05)[0]
            
            improved = False
            for tri_i in close_tris[:5]: # Limit to a few
                i, j, k = tri_idx[tri_i]
                # Calculate gradients for these specific points to push them apart
                # Simple heuristic: move points slightly away from each other
                p_i = best_pts[i]
                p_j = best_pts[j]
                p_k = best_pts[k]
                
                # Centroid
                centroid = (p_i + p_j + p_k) / 3.0
                
                # Move points away from centroid
                step = 0.005
                trial = best_pts.copy()
                trial[i] += (p_i - centroid) * step
                trial[j] += (p_j - centroid) * step
                trial[k] += (p_k - centroid) * step
                
                new_score = _compute_score(trial, tri_idx)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = trial
                    improved = True
            if not improved:
                # Try random small perturbations
                trial = best_pts + np.random.normal(0, 0.01, best_pts.shape)
                new_score = _compute_score(trial, tri_idx)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = trial
                else:
                    break # Stagnated

    # Normalize to unit hull area for consistency
    hull_area = _hull_area(best_pts)
    if hull_area > 1e-9:
        best_pts = best_pts / np.sqrt(hull_area)
        
    # Ensure finite
    if not np.all(np.isfinite(best_pts)):
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_pts

# EVOLVE-BLOCK-END
