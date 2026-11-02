"""Circle packing in perimeter-4 rectangle: Delaunay contact-graph + LP radii solver."""
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import linprog, minimize

N = 21

def _delaunay_edges(points):
    """Return list of (i, j) edges from Delaunay triangulation of 2D points."""
    if len(points) < 3:
        return [(i, j) for i in range(len(points)) for j in range(i+1, len(points))]
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for k in range(3):
                a, b = simplex[k], simplex[(k+1) % 3]
                if a > b:
                    a, b = b, a
                edges.add((a, b))
        return list(edges)
    except Exception:
        # Fallback: all pairs (expensive but safe)
        return [(i, j) for i in range(len(points)) for j in range(i+1, len(points))]

def _solve_lp_radii(points, w, h):
    """
    Solve LP: maximize sum(r_i)
    subject to:
      r_i >= 0
      r_i <= dist(i, j) / 2  for all Delaunay edges (i,j)
      r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= h - y_i  (wall constraints)
    """
    n = len(points)
    xs = points[:, 0]
    ys = points[:, 1]
    
    # Collect all constraints: r_i + r_j <= d_ij for edges, r_i <= wall_dist
    # Variables: r_0, ..., r_{n-1}
    # Maximize sum(r) => minimize -sum(r)
    c = -np.ones(n)
    
    A_ub = []
    b_ub = []
    
    edges = _delaunay_edges(points)
    for (i, j) in edges:
        d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
        # r_i + r_j <= d
        row = np.zeros(n)
        row[i] = 1.0
        row[j] = 1.0
        A_ub.append(row)
        b_ub.append(d)
    
    # Wall constraints: r_i <= min(x_i, w-x_i, y_i, h-y_i)
    for i in range(n):
        wall = min(xs[i], w - xs[i], ys[i], h - ys[i])
        if wall < 0:
            wall = 0.0
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(wall)
    
    if not A_ub:
        return np.zeros(n)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    
    return np.zeros(n)

def _make_grid_init(w, h, n_circles=N):
    """Create a grid-like initial placement."""
    cols = int(np.ceil(np.sqrt(n_circles * w / h))) if h > 0 else 5
    rows = int(np.ceil(n_circles / cols)) if cols > 0 else 5
    cols = max(1, cols)
    rows = max(1, rows)
    
    points = np.zeros((n_circles, 2))
    for i in range(n_circles):
        col = i % cols
        row = i // cols
        points[i, 0] = (col + 0.5) * w / cols
        points[i, 1] = (row + 0.5) * h / rows
    return points

def _evaluate(points, w):
    """Compute sum of radii for given points and width."""
    h = 2.0 - w
    if w <= 0 or h <= 0:
        return -np.inf
    # Check all points inside
    xs, ys = points[:, 0], points[:, 1]
    if np.any(xs < 0) or np.any(xs > w) or np.any(ys < 0) or np.any(ys > h):
        return -np.inf
    radii = _solve_lp_radii(points, w, h)
    return np.sum(radii)

def construct():
    """Pack 21 circles with disjoint interiors in a rectangle of perimeter 4,
    maximizing the sum of radii.
    Returns array of shape (n, 3): [x, y, r] for each circle.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    best_score = -np.inf
    best_result = None
    
    def try_solution(points, w):
        nonlocal best_score, best_result
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return
        # Check bounds
        xs, ys = points[:, 0], points[:, 1]
        if np.any(xs < -1e-9) or np.any(xs > w + 1e-9) or np.any(ys < -1e-9) or np.any(ys > h + 1e-9):
            return
        radii = _solve_lp_radii(points, w, h)
        if np.any(radii < -1e-12):
            return
        radii = np.maximum(radii, 0.0)
        score = np.sum(radii)
        
        # Validate: check all pairwise distances >= r_i + r_j
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - 1e-7:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and score > best_score:
            best_score = score
            result = np.zeros((N, 3))
            result[:, 0] = xs
            result[:, 1] = ys
            result[:, 2] = radii
            best_result = result
    
    # Phase 1: Try various aspect ratios with grid initialization
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.7, 2.0, 0.5, 1.3, 0.9, 1.1, 1.4, 0.85, 1.15, 1.6, 0.65, 1.8, 0.6]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        w = ar
        h = 2.0 - w
        if w <= 0 or h <= 0:
            continue
        
        points = _make_grid_init(w, h)
        try_solution(points, w)
        
        # Phase 2: Refine positions via SLSQP-like iterative local search
        # Use a simple gradient-free local improvement: nudge points to maximize min-distance
        # We'll use scipy minimize to maximize a proxy: sum of radii from LP
        # But LP is not smooth, so instead we optimize a smooth proxy:
        # minimize -sum(min_dist_to_neighbors) as a proxy for larger radii
        
        if time.time() > deadline - 10:
            break
        
        # Smooth proxy: for each point, its "available radius" is half the min distance to any other point
        # and also limited by walls. We maximize the sum of these.
        def proxy_objective(params):
            w_val = params[0]
            h_val = 2.0 - w_val
            if w_val <= 0.01 or h_val <= 0.01:
                return 1e10
            pts = params[1:].reshape(-1, 2)
            # Penalty for out-of-bounds
            pen = 0.0
            for i in range(N):
                if pts[i, 0] < 0: pen += pts[i, 0]**2 * 100
                if pts[i, 0] > w_val: pen += (pts[i, 0] - w_val)**2 * 100
                if pts[i, 1] < 0: pen += pts[i, 1]**2 * 100
                if pts[i, 1] > h_val: pen += (pts[i, 1] - h_val)**2 * 100
            # Sum of min-distances to other points (proxy for radius)
            total = 0.0
            for i in range(N):
                min_d = min(w_val - pts[i,0], pts[i,0], h_val - pts[i,1], pts[i,1])  # wall limit
                for j in range(N):
                    if i == j: continue
                    d = np.sqrt((pts[i,0]-pts[j,0])**2 + (pts[i,1]-pts[j,1])**2)
                    if d < min_d:
                        min_d = d
                total += min_d
            return -total + pen
        
        x0 = np.zeros(1 + N*2)
        x0[0] = w
        x0[1:] = points.flatten()
        
        try:
            res = minimize(proxy_objective, x0, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < 1e8:  # reasonable
                w_opt = res.x[0]
                pts_opt = res.x[1:].reshape(-1, 2)
                # Clamp to bounds
                h_opt = 2.0 - w_opt
                pts_opt[:, 0] = np.clip(pts_opt[:, 0], 0, w_opt)
                pts_opt[:, 1] = np.clip(pts_opt[:, 1], 0, h_opt)
                try_solution(pts_opt, w_opt)
        except Exception:
            pass
    
    # Phase 3: Random restarts with perturbation
    if time.time() < deadline - 15:
        rng = np.random.default_rng(42)
        for restart in range(10):
            if time.time() > deadline - 10:
                break
            ar = rng.uniform(0.5, 1.8)
            w = ar
            h = 2.0 - w
            if w <= 0.1 or h <= 0.1:
                continue
            points = _make_grid_init(w, h)
            # Perturb
            points += rng.normal(0, 0.02, size=points.shape)
            points[:, 0] = np.clip(points[:, 0], 0, w)
            points[:, 1] = np.clip(points[:, 1], 0, h)
            try_solution(points, w)
            
            # Quick local refinement
            if time.time() > deadline - 10:
                break
            
            def proxy_objective(params):
                w_val = params[0]
                h_val = 2.0 - w_val
                if w_val <= 0.01 or h_val <= 0.01:
                    return 1e10
                pts = params[1:].reshape(-1, 2)
                pen = 0.0
                for i in range(N):
                    if pts[i, 0] < 0: pen += pts[i, 0]**2 * 100
                    if pts[i, 0] > w_val: pen += (pts[i, 0] - w_val)**2 * 100
                    if pts[i, 1] < 0: pen += pts[i, 1]**2 * 100
                    if pts[i, 1] > h_val: pen += (pts[i, 1] - h_val)**2 * 100
                total = 0.0
                for i in range(N):
                    min_d = min(w_val - pts[i,0], pts[i,0], h_val - pts[i,1], pts[i,1])
                    for j in range(N):
                        if i == j: continue
                        d = np.sqrt((pts[i,0]-pts[j,0])**2 + (pts[i,1]-pts[j,1])**2)
                        if d < min_d:
                            min_d = d
                    total += min_d
                return -total + pen
            
            x0 = np.zeros(1 + N*2)
            x0[0] = w
            x0[1:] = points.flatten()
            try:
                res = minimize(proxy_objective, x0, method='L-BFGS-B',
                              options={'maxiter': 1500, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.fun < 1e8:
                    w_opt = res.x[0]
                    pts_opt = res.x[1:].reshape(-1, 2)
                    h_opt = 2.0 - w_opt
                    pts_opt[:, 0] = np.clip(pts_opt[:, 0], 0, w_opt)
                    pts_opt[:, 1] = np.clip(pts_opt[:, 1], 0, h_opt)
                    try_solution(pts_opt, w_opt)
            except Exception:
                pass
    
    # Phase 4: If we have a best result, do a final LP polish
    if best_result is not None and time.time() < deadline - 5:
        xs = best_result[:, 0].copy()
        ys = best_result[:, 1].copy()
        rs = best_result[:, 2].copy()
        # Estimate w and h from bounding box
        w_est = np.max(xs + rs) - np.min(xs - rs)
        h_est = np.max(ys + rs) - np.min(ys - rs)
        total_dim = w_est + h_est
        if total_dim > 0:
            w_final = w_est / total_dim * 2.0
        else:
            w_final = 1.0
        h_final = 2.0 - w_final
        
        # Rescale positions to fit in [0, w_final] x [0, h_final]
        x_min = np.min(xs - rs)
        y_min = np.min(ys - rs)
        x_max = np.max(xs + rs)
        y_max = np.max(ys + rs)
        x_span = x_max - x_min
        y_span = y_max - y_min
        if x_span > 0 and y_span > 0:
            scale_x = w_final / x_span
            scale_y = h_final / y_span
            scale = min(scale_x, scale_y)
            xs = (xs - x_min) * scale
            ys = (ys - y_min) * scale
            rs = rs * scale
            # Center
            xs = (xs + (w_final - np.max(xs + rs) + np.min(xs - rs)) / 2)
            ys = (ys + (h_final - np.max(ys + rs) + np.min(ys - rs)) / 2)
        
        points_final = np.column_stack([xs, ys])
        try_solution(points_final, w_final)
    
    # Fallback if nothing worked
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                r = 0.5 / side - 0.005
                cells.append([(i+0.5)/side, (j+0.5)/side, r])
        best_result = np.array(cells[:N])
    
    return best_result
