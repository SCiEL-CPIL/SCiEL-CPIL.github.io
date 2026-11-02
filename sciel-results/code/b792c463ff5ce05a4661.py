"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 13
DEADLINE = 140.0  # safety margin before 150s kill


def _min_tri_area(pts):
    """Compute minimum triangle area among all triples of points."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            v = pts[j] - pts[i]
            rel = pts - pts[i]
            cross = rel[:, 0] * v[1] - rel[:, 1] * v[0]
            areas = np.abs(cross) * 0.5
            areas[i] = np.inf
            areas[j] = np.inf
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        order = np.argsort(np.arctan2(pts[:, 1] - pts[:, 1].mean(),
                                       pts[:, 0] - pts[:, 0].mean()))
        p = pts[order]
        x, y = p[:, 0], p[:, 1]
        return 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    min_tri = _min_tri_area(pts)
    return min_tri / hull_a


def _all_triangle_areas(pts):
    """Return array of all triangle areas (n choose 3)."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                areas.append(0.5 * abs(cross))
    return np.array(areas)


def _soft_min_score(pts, temperature=0.01):
    """Smooth surrogate: -log-sum-exp of triangle areas / hull area."""
    areas = _all_triangle_areas(pts)
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    normalized = areas / hull_a
    # Soft min: -log(sum(exp(-a/t))) / n * t approximates min(a)
    # Use log-sum-exp trick for numerical stability
    min_val = np.min(normalized)
    shifted = normalized - min_val
    log_sum_exp = min_val + temperature * np.log(np.sum(np.exp(-shifted / temperature)))
    # This gives a smooth approximation of min
    return -log_sum_exp


def _make_symmetric_hex7(params):
    """
    Build 13 points: 7 outer (regular heptagon) + 6 inner (regular hexagon).
    params = [r_outer, r_inner, angle_outer, angle_inner, cx, cy]
    """
    r_out = params[0]
    r_in = params[1]
    ang_out = params[2]
    ang_in = params[3]
    cx = params[4]
    cy = params[5]
    
    angles_out = np.linspace(0, 2 * np.pi, 7, endpoint=False) + ang_out
    pts_out = np.column_stack([
        cx + r_out * np.cos(angles_out),
        cy + r_out * np.sin(angles_out)
    ])
    
    angles_in = np.linspace(0, 2 * np.pi, 6, endpoint=False) + ang_in
    pts_in = np.column_stack([
        cx + r_in * np.cos(angles_in),
        cy + r_in * np.sin(angles_in)
    ])
    
    return np.vstack([pts_out, pts_in])


def _score_symmetric_hex7(params):
    pts = _make_symmetric_hex7(params)
    return -_score(pts)


def _make_symmetric_6_6_1(params):
    """
    Build 13 points: 6 outer (hexagon) + 6 middle (hexagon) + 1 center.
    params = [r_outer, r_mid, angle_outer, angle_mid, cx, cy]
    """
    r_out = params[0]
    r_mid = params[1]
    ang_out = params[2]
    ang_mid = params[3]
    cx = params[4]
    cy = params[5]
    
    angles_out = np.linspace(0, 2 * np.pi, 6, endpoint=False) + ang_out
    pts_out = np.column_stack([
        cx + r_out * np.cos(angles_out),
        cy + r_out * np.sin(angles_out)
    ])
    
    angles_mid = np.linspace(0, 2 * np.pi, 6, endpoint=False) + ang_mid
    pts_mid = np.column_stack([
        cx + r_mid * np.cos(angles_mid),
        cy + r_mid * np.sin(angles_mid)
    ])
    
    center = np.array([[cx, cy]])
    
    return np.vstack([pts_out, pts_mid, center])


def _score_symmetric_6_6_1(params):
    pts = _make_symmetric_6_6_1(params)
    return -_score(pts)


def _make_symmetric_12_1(params):
    """
    Build 13 points: 12 outer (regular dodecagon) + 1 center.
    params = [r_outer, angle_outer, cx, cy]
    """
    r_out = params[0]
    ang_out = params[1]
    cx = params[2]
    cy = params[3]
    
    angles_out = np.linspace(0, 2 * np.pi, 12, endpoint=False) + ang_out
    pts_out = np.column_stack([
        cx + r_out * np.cos(angles_out),
        cy + r_out * np.sin(angles_out)
    ])
    
    center = np.array([[cx, cy]])
    
    return np.vstack([pts_out, center])


def _score_symmetric_12_1(params):
    pts = _make_symmetric_12_1(params)
    return -_score(pts)


def _grid_search_symmetric(make_fn, score_fn, bounds, grid_sizes, deadline_check=None, t_start=None):
    """
    Grid search over symmetric parameter space.
    bounds: list of (low, high) for each parameter
    grid_sizes: list of number of grid points for each parameter
    """
    best_score = -np.inf
    best_params = None
    
    # Generate grid
    grids = [np.linspace(bounds[i][0], bounds[i][1], grid_sizes[i]) for i in range(len(bounds))]
    
    # Use itertools.product for grid search
    import itertools
    for combo in itertools.product(*grids):
        if deadline_check is not None and t_start is not None:
            if time.time() - t_start > deadline_check:
                break
        params = np.array(combo)
        s = -score_fn(params)  # score_fn returns negative score
        if s > best_score:
            best_score = s
            best_params = params.copy()
    
    return best_score, best_params


def construct():
    t_start = time.time()
    
    best_score = 0.0
    best_pts = None
    
    # Strategy 1: Grid search over symmetric configurations
    # Hexagon (6) + Heptagon (7)
    bounds_hex7 = [(0.5, 1.5), (0.1, 1.0), (0, 2 * np.pi), (0, 2 * np.pi),
                   (-0.2, 0.2), (-0.2, 0.2)]
    grid_sizes_hex7 = [8, 8, 6, 6, 3, 3]
    
    try:
        s, params = _grid_search_symmetric(
            _make_symmetric_hex7, _score_symmetric_hex7,
            bounds_hex7, grid_sizes_hex7,
            deadline_check=DEADLINE, t_start=t_start
        )
        if s > best_score:
            best_score = s
            best_pts = _make_symmetric_hex7(params).copy()
    except Exception:
        pass
    
    # Hexagon (6) + Hexagon (6) + Center (1)
    bounds_661 = [(0.5, 1.5), (0.1, 1.0), (0, 2 * np.pi), (0, 2 * np.pi),
                  (-0.2, 0.2), (-0.2, 0.2)]
    grid_sizes_661 = [8, 8, 6, 6, 3, 3]
    
    try:
        s, params = _grid_search_symmetric(
            _make_symmetric_6_6_1, _score_symmetric_6_6_1,
            bounds_661, grid_sizes_661,
            deadline_check=DEADLINE, t_start=t_start
        )
        if s > best_score:
            best_score = s
            best_pts = _make_symmetric_6_6_1(params).copy()
    except Exception:
        pass
    
    # Dodecagon (12) + Center (1)
    bounds_121 = [(0.5, 1.5), (0, 2 * np.pi), (-0.2, 0.2), (-0.2, 0.2)]
    grid_sizes_121 = [10, 8, 3, 3]
    
    try:
        s, params = _grid_search_symmetric(
            _make_symmetric_12_1, _score_symmetric_12_1,
            bounds_121, grid_sizes_121,
            deadline_check=DEADLINE, t_start=t_start
        )
        if s > best_score:
            best_score = s
            best_pts = _make_symmetric_12_1(params).copy()
    except Exception:
        pass
    
    # Strategy 2: Differential evolution on symmetric parameters
    if best_pts is not None and time.time() - t_start < DEADLINE - 30:
        # Refine hex7
        try:
            result = differential_evolution(
                _score_symmetric_hex7, bounds_hex7,
                seed=42, maxiter=50, popsize=12,
                tol=1e-6, mutation=(0.3, 1.0), recombination=0.8,
                polish=True
            )
            pts = _make_symmetric_hex7(result.x)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            pass
        
        # Refine 661
        if time.time() - t_start < DEADLINE - 20:
            try:
                result = differential_evolution(
                    _score_symmetric_6_6_1, bounds_661,
                    seed=42, maxiter=50, popsize=12,
                    tol=1e-6, mutation=(0.3, 1.0), recombination=0.8,
                    polish=True
                )
                pts = _make_symmetric_6_6_1(result.x)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
            except Exception:
                pass
    
    # Strategy 3: Full 26D Nelder-Mead refinement from best symmetric solution
    if best_pts is not None and time.time() - t_start < DEADLINE - 15:
        try:
            def neg_score_flat(x):
                pts = x.reshape(-1, 2)
                return -_score(pts)
            
            x0 = best_pts.flatten()
            
            res = minimize(
                neg_score_flat, x0,
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if time.time() - t_start < DEADLINE:
                pts = res.x.reshape(-1, 2)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
        except Exception:
            pass
    
    # Strategy 4: Multi-start Nelder-Mead from diverse symmetric configs
    if time.time() - t_start < DEADLINE - 10:
        def neg_score_flat(x):
            pts = x.reshape(-1, 2)
            return -_score(pts)
        
        np.random.seed(123)
        starts = []
        
        # Regular 13-gon
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        starts.append(np.column_stack([np.cos(angles), np.sin(angles)]).flatten())
        
        # Various hex+hep configs
        for r_h in [0.3, 0.5, 0.7]:
            for r_he in [0.4, 0.6, 0.8]:
                for a_h in [0, np.pi/6]:
                    for a_he in [0, np.pi/7]:
                        p = _make_symmetric_hex7([r_h, r_he, a_h, a_he, 0, 0])
                        starts.append(p.flatten())
        
        # Random perturbations of best
        if best_pts is not None:
            for _ in range(3):
                perturbed = best_pts + np.random.randn(13, 2) * 0.05
                starts.append(perturbed.flatten())
        
        for x0 in starts:
            if time.time() - t_start > DEADLINE:
                break
            try:
                res = minimize(
                    neg_score_flat, x0,
                    method='Nelder-Mead',
                    options={'maxiter': 400, 'xatol': 1e-7, 'fatol': 1e-9}
                )
                pts = res.x.reshape(-1, 2)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
            except Exception:
                continue
    
    # Final fallback
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts


# Quick self-test
if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
    print(f"Points:\n{pts}")
