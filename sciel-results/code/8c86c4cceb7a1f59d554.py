"""Heilbronn convex regions: maximize min triangle area / hull area for 13 points."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 13
MAX_TIME = 140.0  # Safety margin before 150s kill

# Precompute triangle indices
_TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))
_TRI_I = _TRI_IDX[:, 0]
_TRI_J = _TRI_IDX[:, 1]
_TRI_K = _TRI_IDX[:, 2]

def _hull_area(pts):
    """Calculate the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _min_tri_area(pts):
    """Calculate the minimum area of any triangle formed by three points."""
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _score(pts):
    """Calculate the score: min triangle area / hull area."""
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 0.0
    mta = _min_tri_area(pts)
    return mta / ha

def _smooth_min(areas, tau):
    """Compute smooth minimum using log-sum-exp."""
    max_area = np.max(areas)
    log_sum_exp = np.log(np.sum(np.exp(-(areas - max_area) / tau)))
    return max_area - tau * log_sum_exp

def _is_convex(pts):
    """Check if points form a convex polygon (all cross products same sign)."""
    n = len(pts)
    if n < 3:
        return True
    signs = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        p2 = pts[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross != 0:
            signs.append(np.sign(cross))
    if not signs:
        return True
    return all(s == signs[0] for s in signs)

def _cartesian_objective(x, tau=0.01):
    """
    Objective function for Cartesian coordinates.
    x: flat array of 2*N coordinates
    Returns negative score (to minimize) with penalty for non-convex or degenerate configs.
    """
    pts = x.reshape(N, 2)
    
    # Check for degenerate points
    if np.any(np.abs(pts) < 1e-12):
        return 1e6
    
    # Compute hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 1e6
    
    if hull_area < 1e-8:
        return 1e6
    
    # Compute triangle areas
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    areas = 0.5 * np.abs(cross)
    
    if np.any(areas < 1e-12):
        return 1e6
    
    # Smooth min
    smooth_min = _smooth_min(areas, tau)
    score = smooth_min / hull_area
    
    # Convexity penalty: if not convex, add large penalty
    if not _is_convex(pts):
        return 1e4 + score
    
    return -score

def _generate_random_convex_points(rng, n=N):
    """
    Generate random points that are likely to form a convex polygon.
    Strategy: generate random angles, sort them, assign random radii.
    """
    angles = np.sort(rng.random(n) * 2 * np.pi)
    radii = 0.5 + 0.5 * rng.random(n)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def _refine_with_lbfgs(pts, deadline, max_iterations=300):
    """
    Refine a point configuration using L-BFGS-B in Cartesian space.
    Uses adaptive temperature for smooth min.
    """
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Start with larger tau for exploration, then refine
    temperatures = [0.05, 0.02, 0.01, 0.005, 0.002]
    
    for temp in temperatures:
        if time.time() > deadline - 3:
            break
        
        def obj(x):
            return _cartesian_objective(x, tau=temp)
        
        try:
            res = minimize(
                obj,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': max_iterations, 'ftol': 1e-9, 'gtol': 1e-7}
            )
            opt_pts = res.x.reshape(N, 2)
            
            # Only accept if convex and valid
            if _is_convex(opt_pts) and _hull_area(opt_pts) > 1e-8:
                score = _score(opt_pts)
                if score > best_score:
                    best_score = score
                    best_pts = opt_pts
        except Exception:
            continue
    
    return best_pts, best_score

def _custom_de_cartesian(deadline, pop_size=20, max_iter=50):
    """
    Custom differential evolution in Cartesian space with time monitoring.
    """
    rng = np.random.default_rng(42)
    
    # Initialize population
    population = []
    scores = []
    
    # First point: regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    regular_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    population.append(regular_pts.flatten())
    scores.append(-_score(regular_pts))
    
    # Other points: random convex-like configs
    for i in range(pop_size - 1):
        pts = _generate_random_convex_points(rng)
        if _is_convex(pts):
            population.append(pts.flatten())
            scores.append(-_cartesian_objective(pts.flatten(), tau=0.1))
        else:
            # Fallback: slightly perturbed regular polygon
            perturbation = rng.normal(0, 0.1, (N, 2))
            pts = regular_pts + perturbation
            population.append(pts.flatten())
            scores.append(-_cartesian_objective(pts.flatten(), tau=0.1))
    
    best_idx = np.argmin(scores)
    best_x = population[best_idx].copy()
    best_f = scores[best_idx]
    
    F = 0.8  # Mutation factor
    CR = 0.9  # Crossover rate
    
    for gen in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        for i in range(pop_size):
            if time.time() > deadline - 5:
                break
            
            # Select three distinct random individuals
            a, b, c = rng.choice(pop_size, 3, replace=False)
            
            # Mutation
            v = population[a] + F * (population[b] - population[c])
            
            # Crossover
            trial = np.empty_like(v)
            j_rand = rng.randint(0, len(v))
            for j in range(len(v)):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = population[i][j]
            
            # Evaluate
            trial_pts = trial.reshape(N, 2)
            if _is_convex(trial_pts):
                trial_f = _cartesian_objective(trial, tau=0.1)
            else:
                trial_f = 1e4
            
            # Selection
            if trial_f <= scores[i]:
                population[i] = trial
                scores[i] = trial_f
                if trial_f < best_f:
                    best_f = trial_f
                    best_x = trial.copy()
        
        # Periodic refinement of best
        if gen % 10 == 9 and time.time() < deadline - 10:
            best_pts = best_x.reshape(N, 2)
            refined_pts, refined_score = _refine_with_lbfgs(best_pts, deadline, max_iterations=100)
            if refined_score > -best_f:
                best_x = refined_pts.flatten()
                best_f = -refined_score
    
    return best_x.reshape(N, 2), -best_f

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    best_score = -np.inf
    best_pts = None
    
    # Phase 1: Custom DE in Cartesian space
    try:
        de_pts, de_score = _custom_de_cartesian(deadline, pop_size=20, max_iter=40)
        if de_score > best_score:
            best_score = de_score
            best_pts = de_pts
    except Exception:
        pass
    
    if time.time() > deadline - 25:
        if best_pts is None:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            best_pts = np.column_stack((np.cos(angles), np.sin(angles)))
            best_score = _score(best_pts)
        return best_pts
    
    # Phase 2: Multi-start L-BFGS refinement
    starts = []
    
    # Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    starts.append(np.column_stack((np.cos(angles), np.sin(angles))))
    
    # Best from DE
    if best_pts is not None:
        starts.append(best_pts.copy())
    
    # Random convex starts
    rng = np.random.default_rng(123)
    for seed in range(5):
        pts = _generate_random_convex_points(rng)
        if _is_convex(pts):
            starts.append(pts)
    
    # Slightly perturbed regular polygons
    for pert_scale in [0.05, 0.1, 0.15]:
        perturbation = rng.normal(0, pert_scale, (N, 2))
        pts = starts[0] + perturbation
        if _is_convex(pts):
            starts.append(pts)
    
    for start in starts:
        if time.time() > deadline - 8:
            break
        
        try:
            opt_pts, opt_score = _refine_with_lbfgs(start, deadline, max_iterations=200)
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts
        except Exception:
            continue
    
    # Phase 3: Final polish if time permits
    if time.time() < deadline - 5 and best_pts is not None:
        try:
            final_pts, final_score = _refine_with_lbfgs(best_pts, deadline, max_iterations=150)
            if final_score > best_score:
                best_score = final_score
                best_pts = final_pts
        except Exception:
            pass
    
    # Fallback
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_pts
