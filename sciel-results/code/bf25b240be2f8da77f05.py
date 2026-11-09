"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13
DEADLINE = 115.0  # safety margin before 120s kill

def _get_triangle_areas(points):
    """Compute all triangle areas vectorized."""
    n = len(points)
    if n < 3:
        return np.array([0.0])
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / hull area."""
    hull_a = _hull_area(points)
    if hull_a < 1e-12:
        return 0.0
    min_ta = np.min(_get_triangle_areas(points))
    return min_ta / hull_a

def _soft_min_objective(points, beta):
    """
    Objective for minimization: -soft_min(areas) / hull_area.
    soft_min(x) = -1/beta * log(mean(exp(-beta * x)))
    This is a smooth lower bound on min(x).
    As beta -> infinity, soft_min(x) -> min(x).
    """
    areas = _get_triangle_areas(points)
    hull_a = _hull_area(points)
    if hull_a < 1e-12:
        return 1e10 # Penalize degenerate hulls
    
    # Numerical stability for log-sum-exp
    # We want: -1/beta * log( sum(exp(-beta * areas)) / len(areas) )
    # = -1/beta * ( log(sum(exp(-beta * areas))) - log(len(areas)) )
    # Let M = min(areas). 
    # log(sum(exp(-beta * areas))) = -beta*M + log(sum(exp(-beta * (areas - M))))
    m = np.min(areas)
    # If beta is large, beta*m can be large, but we subtract it.
    # Actually, simpler:
    # soft_min = -1/beta * log( mean( exp(-beta * areas) ) )
    # To avoid underflow/overflow in exp, shift by min:
    # exp(-beta * areas) = exp(-beta * (areas - m)) * exp(-beta * m)
    # sum(exp(-beta * areas)) = exp(-beta * m) * sum(exp(-beta * (areas - m)))
    # log(sum(...)) = -beta * m + log(sum(exp(-beta * (areas - m))))
    # soft_min = -1/beta * ( -beta * m + log(sum(exp(-beta * (areas - m)))) - log(N_tris) )
    # soft_min = m - 1/beta * log(sum(exp(-beta * (areas - m)))) + 1/beta * log(N_tris)
    
    shifted = areas - m
    # exp(-beta * shifted) is <= 1, so no overflow.
    # Underflow is fine (becomes 0).
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    if sum_exp < 1e-300:
        return -m / hull_a # Fallback to hard min
    
    log_sum = np.log(sum_exp)
    n_tris = len(areas)
    log_mean = log_sum - np.log(n_tris)
    
    soft_min_val = m - (1.0 / beta) * log_mean
    
    return -soft_min_val / hull_a

def _initial_polygon(n):
    """Regular n-gon centered at origin, unit radius."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _initial_polygon_plus_center(n):
    """Regular (n-1)-gon plus center point."""
    if n <= 1:
        return np.zeros((n, 2))
    m = n - 1
    angles = np.linspace(0, 2 * np.pi, m, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    center = np.array([[0.0, 0.0]])
    return np.vstack([pts, center])

def _optimize_soft_min(start_points, deadline):
    """
    Optimize using L-BFGS-B on the soft-min objective with annealing.
    """
    best_points = start_points.copy()
    best_score = _score(best_points)
    
    # Annealing schedule for beta
    # Start smooth, end sharp
    betas = [10, 50, 100, 200, 500]
    
    current_points = start_points.copy()
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
            
        def objective(x):
            points = x.reshape(-1, 2)
            return _soft_min_objective(points, beta)
            
        x0 = current_points.flatten()
        bounds = [(-5, 5)] * (2 * N)
        
        try:
            # Use L-BFGS-B as it handles bounds and gradients well
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500}
            )
            
            cand = result.x.reshape(-1, 2)
            cand_score = _score(cand)
            
            if cand_score > best_score:
                best_score = cand_score
                best_points = cand.copy()
                current_points = cand.copy()
            else:
                # Even if score didn't improve in true metric, the soft-min
                # optimization might have moved us to a better basin.
                # Accept if soft-min objective improved significantly?
                # For simplicity, stick to best true score for 'best_points'
                # but use result for next iteration if it's finite.
                if np.isfinite(result.fun):
                     current_points = cand.copy()
                     
        except Exception:
            pass

    return best_points, best_score

def _optimize_nelder_mead(start_points, deadline):
    """Optimize using Nelder-Mead on the true (nonsmooth) objective."""
    best_points = start_points.copy()
    best_score = _score(best_points)

    def objective(x):
        points = x.reshape(-1, 2)
        return -_score(points)

    x0 = start_points.flatten()

    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        cand = result.x.reshape(-1, 2)
        cand_score = _score(cand)
        if cand_score > best_score:
            best_points = cand
            best_score = cand_score
    except Exception:
        pass

    return best_points, best_score

def construct():
    """Construct 13 points maximizing min triangle area / hull area."""
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_points = None
    best_score = 0.0

    # Generate diverse initial configurations
    initials = []
    initials.append(_initial_polygon(N))
    initials.append(_initial_polygon_plus_center(N))
    
    # Add a jittered version of the polygon to break symmetry
    rng = np.random.RandomState(42)
    initials.append(_initial_polygon(N) + rng.normal(0, 0.1, size=(N, 2)))

    # Phase 1: Soft-Min Annealing from each initial
    # This is the primary structural improvement
    for init_pts in initials:
        if time.time() > deadline - 30.0:
            break
        pts, score = _optimize_soft_min(init_pts, deadline)
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # Phase 2: Nelder-Mead Polish
    # Use the best result from Soft-Min as a starting point for local search
    # Nelder-Mead can handle the kinks in the true min function better locally
    if best_points is not None:
        if time.time() < deadline - 15.0:
            pts, score = _optimize_nelder_mead(best_points, deadline)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
        
        # Also try Nelder-Mead from the initial regular polygon in case
        # soft-min got stuck in a weird basin
        if time.time() < deadline - 10.0:
            pts, score = _optimize_nelder_mead(initials[0], deadline)
            if score > best_score:
                best_score = score
                best_points = pts.copy()

    # Phase 3: Final Soft-Min polish with high beta if time remains
    if best_points is not None and time.time() < deadline - 5.0:
        # High beta makes it very close to the true min, acting as a sharpener
        # We can just run one more iteration of the soft-min loop with high beta
        # Or simply rely on the previous phases.
        # Let's do a quick L-BFGS-B on the true gradient (subgradient) if available?
        # No, let's stick to the soft-min logic. 
        # Re-run soft min with just high beta from current best
        def objective_high_beta(x):
            points = x.reshape(-1, 2)
            return _soft_min_objective(points, 1000)
            
        x0 = best_points.flatten()
        try:
            result = minimize(
                objective_high_beta,
                x0,
                method='L-BFGS-B',
                bounds=[(-5, 5)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            cand = result.x.reshape(-1, 2)
            cand_score = _score(cand)
            if cand_score > best_score:
                best_score = cand_score
                best_points = cand.copy()
        except Exception:
            pass

    if best_points is None:
        return initials[0]

    return best_points
