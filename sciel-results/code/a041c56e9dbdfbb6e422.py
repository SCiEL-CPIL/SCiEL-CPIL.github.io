"""
Discrete Angular Grid Search + Robust Local Polish for Heilbronn Convex Regions n=14.

Strategy:
1. Generate a diverse set of initial convex configurations by perturbing the angles
   of a regular 14-gon on a discrete grid. This ensures we explore different
   "shapes" (e.g., slightly squashed, elongated, asymmetric) without getting stuck
   in the immediate basin of the regular polygon.
2. Evaluate these configurations exactly.
3. Select the top K candidates.
4. Polish each candidate using a local optimizer (Nelder-Mead) on a smoothed
   objective that penalizes small triangles heavily, but with adaptive parameters.
5. Return the best polished result.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

N = 14
_TRIPLE_INDICES = np.array(list(itertools.combinations(range(N), 3)))
_NUM_TRIPLES = _TRIPLE_INDICES.shape[0]

def _all_triangle_areas(points):
    """Vectorized computation of all triangle areas."""
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _min_triangle_area(points):
    return np.min(_all_triangle_areas(points))

def _convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Score = min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _generate_grid_candidates():
    """
    Generate a set of candidate point sets by perturbing the regular 14-gon
    angles on a discrete grid. We also vary the radius slightly to allow
    for non-circular convex hulls (e.g., ellipses).
    """
    candidates = []
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Grid for angular perturbations
    # We use a small grid to keep the number of candidates manageable
    # Step size 1 degree (pi/180) is too fine for 14 points (too many combos).
    # Instead, we use a structured perturbation: 
    # 1. Uniform rotation (doesn't change score due to normalization, but good for symmetry breaking in local opt? No, score is invariant).
    # 2. "Wave" perturbations: theta_i = base_i + A * sin(k * base_i + phi)
    # 3. Random discrete perturbations from a small set of values.
    
    # Let's use a combination of sinusoidal perturbations and random discrete steps.
    # This captures "near-regular" structures which are likely optimal.
    
    # Sinusoidal modes
    for mode in [2, 3, 4, 5]:
        for amp in [0.0, 0.05, 0.1, 0.15, 0.2]:
            for phase in np.linspace(0, 2*np.pi, 8, endpoint=False):
                angles = base_angles + amp * np.sin(mode * base_angles + phase)
                # Ensure angles are sorted (they are, for small amp)
                # Normalize angles
                angles = np.mod(angles, 2 * np.pi)
                sort_idx = np.argsort(angles)
                angles = angles[sort_idx]
                
                # Try a few radial patterns too
                for rad_amp in [0.0, 0.1, 0.2]:
                    radii = 1.0 + rad_amp * np.cos(mode * angles + phase)
                    x = radii * np.cos(angles)
                    y = radii * np.sin(angles)
                    pts = np.column_stack((x, y))
                    candidates.append(pts)
    
    # Add some random discrete perturbations
    np.random.seed(42)
    for _ in range(200):
        # Random angular offsets from {-0.1, -0.05, 0, 0.05, 0.1}
        d_thetas = np.random.choice([-0.1, -0.05, 0.0, 0.05, 0.1], size=N)
        angles = base_angles + d_thetas
        angles = np.mod(angles, 2 * np.pi)
        sort_idx = np.argsort(angles)
        angles = angles[sort_idx]
        
        # Random radial offsets
        d_radii = np.random.choice([-0.2, -0.1, 0.0, 0.1, 0.2], size=N)
        radii = 1.0 + d_radii
        
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        pts = np.column_stack((x, y))
        candidates.append(pts)
        
    return candidates

def _smoothed_objective(points_flat, alpha=50):
    """
    Objective for local optimization.
    Uses a power-sum approximation for the minimum area to make it differentiable
    and emphasize the smallest triangles.
    Minimize - (sum of areas^alpha)^(1/alpha) / hull_area
    """
    points = points_flat.reshape(N, 2)
    areas = _all_triangle_areas(points)
    # Clip to avoid 0^alpha issues if any area is 0 (degenerate)
    areas = np.maximum(areas, 1e-12)
    
    # Approximate min
    min_area_approx = np.sum(areas**alpha) ** (1.0/alpha)
    
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    
    return -min_area_approx / hull_area

def _polish_candidate(points, time_limit=2.0):
    """
    Polish a single candidate using Nelder-Mead on the smoothed objective.
    """
    start_time = time.time()
    initial_flat = points.flatten()
    
    # Nelder-Mead options
    # maxiter is not a direct time control, so we use a callback or just limit it.
    # Each iteration is ~14 evaluations. 
    # Let's estimate maxiter based on time.
    # Assume 1ms per eval (vectorized numpy is fast).
    # 2s / (14 * 1ms) ~ 140 iterations.
    max_iter = 100
    
    try:
        res = minimize(
            _smoothed_objective,
            initial_flat,
            method='Nelder-Mead',
            args=(50,), # alpha
            options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        if time.time() - start_time > time_limit:
            return points, _score(points)
            
        polished_pts = res.x.reshape(N, 2)
        polished_score = _score(polished_pts)
        
        # Compare with original
        orig_score = _score(points)
        if polished_score > orig_score:
            return polished_pts, polished_score
        else:
            return points, orig_score
    except Exception:
        return points, _score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Phase 1: Generate and score grid candidates
    candidates = _generate_grid_candidates()
    
    # Score all candidates
    scored_candidates = []
    for pts in candidates:
        score = _score(pts)
        scored_candidates.append((score, pts))
        
    # Sort by score
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_score = scored_candidates[0][0] if scored_candidates else 0.0
    best_points = scored_candidates[0][1] if scored_candidates else np.zeros((N, 2))
    
    # Phase 2: Polish top K candidates
    # How many can we polish?
    # Time spent on scoring: ~len(candidates) * 1ms = 0.5s
    # Time remaining: ~110s
    # Cost per polish: ~1-2s
    # We can polish ~50 candidates.
    K = 50
    
    top_k = scored_candidates[:K]
    
    for i, (score, pts) in enumerate(top_k):
        if time.time() > deadline - 5:
            break
            
        polished_pts, polished_score = _polish_candidate(pts, time_limit=1.5)
        
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts
            
    # Phase 3: Final check / slight random restarts if time remains
    # If we have a lot of time left, we might do a few random Cartesian starts
    # but the grid search is usually strong enough for n=14.
    
    return best_points

# EVOLVE-BLOCK-END
