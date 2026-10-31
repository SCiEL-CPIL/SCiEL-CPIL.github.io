import numpy as np
from scipy.spatial import ConvexHull
import time

N = 13

# EVOLVE-BLOCK-START

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = points.shape[0]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j, 0] - points[i, 0]) * (points[k, 1] - points[i, 1]) -
                                  (points[k, 0] - points[i, 0]) * (points[j, 1] - points[i, 1]))
                if area < min_area:
                    min_area = area
    return min_area


def _hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    ha = _hull_area(points)
    if ha < 1e-10:
        return 0.0
    return _min_triangle_area(points) / ha


def _soft_min(values, beta=50.0):
    """Smooth approximation of the minimum using log-sum-exp."""
    # For minimum, use -log-sum-exp of negative values
    neg_vals = -values
    max_neg = np.max(neg_vals)
    if not np.isfinite(max_neg):
        return 0.0
    s = np.sum(np.exp(neg_vals - max_neg))
    return -(max_neg + np.log(s)) / beta


def _soft_score(points, beta=50.0):
    """Smooth surrogate for the score using soft-min over triangle areas."""
    n = points.shape[0]
    areas = np.empty((n * (n - 1) * (n - 2)) // 6)
    idx = 0
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                areas[idx] = 0.5 * abs((points[j, 0] - points[i, 0]) * (points[k, 1] - points[i, 1]) -
                                        (points[k, 0] - points[i, 0]) * (points[j, 1] - points[i, 1]))
                idx += 1
    sm = _soft_min(areas, beta)
    ha = _hull_area(points)
    if ha < 1e-10:
        return 0.0
    return sm / ha


def _nelder_mead(points, max_iter=5000, tol=1e-10, time_limit=None, start_time=None):
    """Nelder-Mead optimization on the score function."""
    n_pts = points.shape[0]
    n_dim = n_pts * 2
    
    # Flatten points
    x0 = points.flatten().copy()
    
    # Initial simplex
    simplex = [x0.copy()]
    for i in range(n_dim):
        x_trial = x0.copy()
        x_trial[i] += 0.01 * (1.0 if x0[i] >= 0 else -1.0)
        simplex.append(x_trial)
    
    # Evaluate initial simplex
    fvals = [_score(np.array(x).reshape(n_pts, 2)) for x in simplex]
    
    alpha, gamma, rho = 1.0, 2.0, 0.5
    
    for it in range(max_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[-1] - fvals[0]) < tol * max(1.0, abs(fvals[0])):
            break
        
        x_worst = simplex[-1]
        x_centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_ref = x_centroid + alpha * (x_centroid - x_worst)
        f_ref = _score(np.array(x_ref).reshape(n_pts, 2))
        
        if f_ref < fvals[0]:
            # Expansion
            x_exp = x_centroid + gamma * (x_ref - x_centroid)
            f_exp = _score(np.array(x_exp).reshape(n_pts, 2))
            if f_exp < f_ref:
                simplex[-1] = x_exp
                fvals[-1] = f_exp
            else:
                simplex[-1] = x_ref
                fvals[-1] = f_ref
        elif f_ref < fvals[-2]:
            simplex[-1] = x_ref
            fvals[-1] = f_ref
        else:
            # Contraction
            x_con = x_centroid + rho * (x_worst - x_centroid)
            f_con = _score(np.array(x_con).reshape(n_pts, 2))
            if f_con < fvals[-1]:
                simplex[-1] = x_con
                fvals[-1] = f_con
            else:
                # Shrink
                for i in range(1, n_dim + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = _score(np.array(simplex[i]).reshape(n_pts, 2))
    
    # Return best point
    best_idx = np.argmin(fvals)
    return np.array(simplex[best_idx]).reshape(n_pts, 2), fvals[best_idx]


def _coordinate_descent(points, max_iter=2000, time_limit=None, start_time=None):
    """Coordinate descent polish on the hard objective."""
    n_pts = points.shape[0]
    best = points.copy()
    best_score = _score(best)
    
    step = 0.01
    for it in range(max_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        improved = False
        for i in range(n_pts):
            for dim in range(2):
                # Try small steps in each direction
                for direction in [-1, 1]:
                    trial = best.copy()
                    trial[i, dim] += direction * step
                    s = _score(trial)
                    if s > best_score:
                        best = trial
                        best_score = s
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Reduce step size
            step *= 0.5
            if step < 1e-10:
                break
        else:
            # Slightly increase step if we made progress
            step = min(step * 1.1, 0.05)
    
    return best, best_score


def _simulated_annealing(points, n_iter=50000, T0=0.1, T_end=0.001, time_limit=None, start_time=None):
    """Simulated annealing to escape local minima."""
    n_pts = points.shape[0]
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    T = T0
    cooling = (T_end / T0) ** (1.0 / max(n_iter, 1))
    
    for it in range(n_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Generate a random perturbation
        trial = current.copy()
        idx = np.random.randint(n_pts)
        dim = np.random.randint(2)
        trial[idx, dim] += np.random.normal(0, 0.05)
        
        trial_score = _score(trial)
        delta = trial_score - current_score
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current = trial
            current_score = trial_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        
        T *= cooling
    
    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin: finish well before 150s
    
    np.random.seed(42)
    
    # Start with a good initial configuration: regular polygon + center
    # For 13 points, place them on a circle with varying radii
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    # Use multiple rings for better distribution
    radii = np.ones(13)
    # Inner ring of 7 points, outer ring of 6 points
    radii[:7] = 0.5
    radii[7:] = 1.0
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Normalize to unit area
    ha = _hull_area(points)
    if ha > 1e-10:
        scale = 1.0 / np.sqrt(ha)
        points *= scale
    
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Phase 1: Nelder-Mead optimization
    remaining = deadline - time.time()
    if remaining > 10:
        nm_points, nm_score = _nelder_mead(points, max_iter=3000, tol=1e-10, 
                                             time_limit=remaining - 5, start_time=start_time)
        if nm_score > best_score:
            best_points = nm_points
            best_score = nm_score
    
    # Phase 2: Simulated annealing to explore
    remaining = deadline - time.time()
    if remaining > 15:
        sa_points, sa_score = _simulated_annealing(best_points, n_iter=30000, 
                                                    T0=0.05, T_end=0.0001,
                                                    time_limit=remaining - 10, start_time=start_time)
        if sa_score > best_score:
            best_points = sa_points
            best_score = sa_score
    
    # Phase 3: Another Nelder-Mead pass
    remaining = deadline - time.time()
    if remaining > 10:
        nm_points2, nm_score2 = _nelder_mead(best_points, max_iter=5000, tol=1e-12,
                                              time_limit=remaining - 5, start_time=start_time)
        if nm_score2 > best_score:
            best_points = nm_points2
            best_score = nm_score2
    
    # Phase 4: Coordinate descent polish
    remaining = deadline - time.time()
    if remaining > 5:
        cd_points, cd_score = _coordinate_descent(best_points, max_iter=3000,
                                                   time_limit=remaining - 2, start_time=start_time)
        if cd_score > best_score:
            best_points = cd_points
            best_score = cd_score
    
    # Final check: ensure we're within deadline
    if time.time() > deadline - 2:
        # Return best found so far
        pass
    
    return best_points

# EVOLVE-BLOCK-END
