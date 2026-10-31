"""Heilbronn triangles: maximize the minimum triangle area over 11 points in a unit-area equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11
TRI_AREA = (3 ** 0.5) / 4.0  # area of the reference equilateral triangle (side=1)

def _min_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the given points, normalized by TRI_AREA."""
    n = len(pts)
    min_a = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                a = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if a < min_a:
                    min_a = a
    return min_a / TRI_AREA

def _feasible(pts):
    """Check if all points are inside the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    for x, y in pts:
        if x < -1e-9 or y < -1e-9 or x > 1 + 1e-9:
            return False
        # y <= sqrt(3) * x  (left edge from (0,0) to (0.5, sqrt(3)/2))
        if y > (3 ** 0.5) * x + 1e-9:
            return False
        # y <= sqrt(3) * (1 - x)  (right edge from (1,0) to (0.5, sqrt(3)/2))
        if y > (3 ** 0.5) * (1 - x) + 1e-9:
            return False
    return True

def _penalty(pts):
    """Return 0 if feasible, else a large penalty."""
    if _feasible(pts):
        return 0.0
    return 1e6

def _objective_flat(flat):
    """Objective for flat parameter vector: negative min area + penalty."""
    pts = flat.reshape(N, 2)
    return -_min_area(pts) + _penalty(pts)

def _smooth_objective(flat, beta=50.0):
    """Smooth approximation of the negative min area using log-sum-exp of reciprocal areas."""
    pts = flat.reshape(N, 2)
    if not _feasible(pts):
        return 1e6
    # Compute all triangle areas
    areas = []
    for i in range(N - 2):
        for j in range(i + 1, N - 1):
            for k in range(j + 1, N):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                a = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                areas.append(max(a / TRI_AREA, 1e-10))
    areas = np.array(areas)
    # Smooth min via log-sum-exp of reciprocals: min(a_i) ≈ exp(-logsumexp(log(1/a_i)/beta)*beta)^{-1}... 
    # Simpler: use -logsumexp(-beta * a) / beta as a smooth approximation of min(a)
    # min(a) ≈ -logsumexp(-beta*a)/beta for large beta
    log_vals = -beta * areas
    max_log = np.max(log_vals)
    lse = max_log + np.log(np.sum(np.exp(log_vals - max_log)))
    smooth_min = -lse / beta
    return -smooth_min

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    best_score = 0.0
    best_pts = None
    
    # Strategy 1: Differential Evolution for global search
    # Bounds for each point: x in [0,1], y in [0, sqrt(3)/2]
    # But we need to ensure feasibility. DE with bounds won't guarantee feasibility.
    # Instead, use a transformation or just penalize.
    
    # Use a simpler approach: start with known good configurations and polish with L-BFGS-B
    
    # Known good starting points:
    # 1. Grid-like
    # 2. Random feasible points
    
    rng = np.random.RandomState(42)
    
    # Generate several random feasible initializations
    def random_feasible_point():
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1:
                x = u + 0.5 * v
                y = (3 ** 0.5) / 2 * v
                return np.array([x, y])
    
    # Try multiple random initializations
    n_starts = 20
    for s in range(n_starts):
        if time.time() > deadline - 5:
            break
        
        # Create a random feasible set of points
        pts = np.array([random_feasible_point() for _ in range(N)])
        flat = pts.flatten()
        
        # Use L-BFGS-B with smooth objective
        try:
            res = minimize(
                _smooth_objective,
                flat,
                method='L-BFGS-B',
                args=(50.0,),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception:
            continue
        
        if time.time() > deadline - 5:
            break
        
        candidate = res.x.reshape(N, 2)
        if _feasible(candidate):
            score = _min_area(candidate)
            if score > best_score:
                best_score = score
                best_pts = candidate.copy()
    
    # Strategy 2: Polish the best result with the true (non-smooth) objective using Nelder-Mead
    if best_pts is not None:
        if time.time() < deadline - 10:
            try:
                res = minimize(
                    _objective_flat,
                    best_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                candidate = res.x.reshape(N, 2)
                if _feasible(candidate):
                    score = _min_area(candidate)
                    if score > best_score:
                        best_score = score
                        best_pts = candidate.copy()
            except Exception:
                pass
    
    # Strategy 3: Try a structured grid-based initialization
    if time.time() < deadline - 15:
        # Create a hexagonal-like pattern
        # Divide the triangle into rows
        rows = [1, 2, 3, 3, 2]  # 11 points
        pts = []
        for row_idx, count in enumerate(rows):
            # y coordinate for this row
            y_frac = (row_idx + 1) / (len(rows) + 1)
            y = y_frac * (3 ** 0.5) / 2
            # x range at this height
            x_min = y / (3 ** 0.5)
            x_max = 1 - y / (3 ** 0.5)
            if x_max > x_min:
                for i in range(count):
                    x = x_min + (x_max - x_min) * (i + 0.5) / count
                    pts.append([x, y])
        
        if len(pts) == N:
            pts = np.array(pts)
            if _feasible(pts):
                score = _min_area(pts)
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
                
                # Polish this
                if time.time() < deadline - 10:
                    try:
                        res = minimize(
                            _smooth_objective,
                            pts.flatten(),
                            method='L-BFGS-B',
                            args=(50.0,),
                            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                        )
                        candidate = res.x.reshape(N, 2)
                        if _feasible(candidate):
                            score2 = _min_area(candidate)
                            if score2 > best_score:
                                best_score = score2
                                best_pts = candidate.copy()
                    except Exception:
                        pass
    
    # Final polish with true objective
    if best_pts is not None and time.time() < deadline - 5:
        try:
            res = minimize(
                _objective_flat,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            candidate = res.x.reshape(N, 2)
            if _feasible(candidate):
                score = _min_area(candidate)
                if score > best_score:
                    best_score = score
                    best_pts = candidate.copy()
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: return random feasible points
        best_pts = np.array([random_feasible_point() for _ in range(N)])
    
    return best_pts
