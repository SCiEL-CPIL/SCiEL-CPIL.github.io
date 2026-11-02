"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time


def _compute_all_areas(pts):
    """
    Vectorized computation of all triangle areas.
    Returns an array of shape (286,).
    """
    n = len(pts)
    # Indices for all triples i < j < k
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    p_i = pts[i_idx]
    p_j = pts[j_idx]
    p_k = pts[k_idx]
    
    # Area = 0.5 * | (p_j - p_i) x (p_k - p_i) |
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    areas = _compute_all_areas(pts)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    return min_area / hull_area

def _penalty_objective(params, t_target, penalty_weight=1.0):
    """
    Objective function for penalty-based max-min optimization.
    1. Normalize points to unit hull area.
    2. Compute areas.
    3. Penalize areas below t_target.
    4. Add small regularization to prevent points from flying apart (though normalization handles scale).
    
    We use a quadratic penalty for violations: sum( (t_target - a)^2 * (a < t_target) )
    To make it differentiable, we use a smooth approximation:
    smooth_max(0, t - a) = 0.5 * (t - a + sqrt((t - a)^2 + eps^2))
    Then square it for the penalty.
    """
    pts = params.reshape(-1, 2)
    
    # Check for degenerate hull
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e10
        
    # Normalize to unit hull area
    scale = 1.0 / np.sqrt(ha)
    pts_norm = pts * scale
    
    areas = _compute_all_areas(pts_norm)
    
    # Calculate penalty for areas below t_target
    # diff = t_target - areas
    # We want to penalize where diff > 0.
    # Using smooth max: m = 0.5 * (diff + sqrt(diff^2 + eps^2))
    # Penalty = sum(m^2)
    diff = t_target - areas
    eps = 1e-4
    m = 0.5 * (diff + np.sqrt(diff**2 + eps**2))
    penalty = np.sum(m**2)
    
    # Regularization: keep points centered to avoid translation drift
    center = np.mean(pts, axis=0)
    reg = np.sum((pts - center)**2) * 1e-6
    
    return penalty_weight * penalty + reg

def _generate_initializations():
    """Generate diverse initial configurations."""
    inits = []
    
    # Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(pts)
    
    # 12-gon + center
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    center = np.array([[0.0, 0.0]])
    pts = np.vstack([pts, center])
    inits.append(pts)
    
    # Random perturbations
    np.random.seed(42)
    for _ in range(10):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = 1.0 + np.random.normal(0, 0.2, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts)
        
    # Random points in disk
    for _ in range(5):
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0.5, 1.0, N))
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts)
        
    return inits

def _optimize_with_penalty(pts_init, deadline):
    """
    Iteratively increase target t and minimize penalty.
    """
    pts = pts_init.copy()
    # Normalize initial points
    ha = _hull_area(pts)
    if ha > 1e-10:
        pts = pts / np.sqrt(ha)
        
    current_score = _score_points(pts)
    # Start target slightly above current min area (normalized)
    # Since pts are normalized to unit hull, min area is the score.
    t_target = current_score * 0.9
    
    best_pts = pts.copy()
    best_score = current_score
    
    # Iteratively increase t_target
    # We want to push t_target up towards the optimal value.
    # The optimal min area for 13 points is roughly around 0.02-0.03?
    # Regular 13-gon score is approx:
    # Area of regular n-gon with unit hull area...
    # Let's just try to increase it.
    
    num_steps = 50
    for i in range(num_steps):
        if time.time() > deadline - 5:
            break
            
        # Increase target
        # If we are stuck, we might not be able to increase much.
        # We increase t_target by a small factor or fixed amount.
        # Let's try to increase by 1% of current best score or fixed small amount.
        # Actually, let's just try to find a feasible solution for higher t.
        # If penalty is 0, we are feasible.
        
        # Try to optimize for current t_target
        try:
            result = minimize(
                lambda x: _penalty_objective(x, t_target),
                pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            candidate_pts = result.x.reshape(-1, 2)
            candidate_score = _score_points(candidate_pts)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts.copy()
                pts = candidate_pts.copy()
                
                # If we significantly improved, we can raise the target
                # Check if min area is close to t_target
                min_area_now = np.min(_compute_all_areas(candidate_pts / np.sqrt(_hull_area(candidate_pts))))
                if min_area_now > t_target * 0.95:
                    # We are feasible for this t, try higher
                    t_target = min_area_now * 1.02
                else:
                    # We are not fully feasible, maybe adjust t down or keep same
                    # If we are stuck, we might need to lower t to allow movement?
                    # No, penalty drives it up. If we can't reach t, we stay at best found.
                    # Let's just try a slightly higher t next time to see if we can break through
                    t_target = t_target * 1.01
                    
        except Exception:
            pass
            
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    best_score = 0.0
    best_pts = None

    inits = _generate_initializations()

    # Evaluate initializations
    for pts in inits:
        score = _score_points(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()

    # Optimize from top initializations
    if best_pts is not None:
        scored_inits = []
        for pts in inits:
            s = _score_points(pts)
            scored_inits.append((s, pts))
        scored_inits.sort(key=lambda x: -x[0])

        # Try top 5
        for idx in range(min(5, len(scored_inits))):
            if time.time() > deadline - 10:
                break
            init_pts = scored_inits[idx][1]
            try:
                opt_pts, opt_score = _optimize_with_penalty(init_pts, deadline)
                if opt_score > best_score:
                    best_score = opt_score
                    best_pts = opt_pts.copy()
            except Exception:
                pass

    # Final polish with Powell on the best result found
    if best_pts is not None and time.time() < deadline - 5:
        # Use a standard score-based polish to refine
        def _neg_score(params):
            pts = params.reshape(-1, 2)
            ha = _hull_area(pts)
            if ha < 1e-10: return 1e10
            pts_norm = pts / np.sqrt(ha)
            areas = _compute_all_areas(pts_norm)
            return -np.min(areas)
            
        try:
            result = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Powell',
                options={'maxiter': 100, 'xtol': 1e-8, 'ftol': 1e-10}
            )
            if result.fun is not None and result.fun < 1e5:
                candidate_pts = result.x.reshape(-1, 2)
                candidate_score = _score_points(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pts = candidate_pts.copy()
        except Exception:
            pass

    if best_pts is not None and len(best_pts) == N:
        return best_pts
    else:
        return inits[0]
# EVOLVE-BLOCK-END
