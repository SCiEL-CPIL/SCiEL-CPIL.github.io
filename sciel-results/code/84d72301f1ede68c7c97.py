"""Circle packing in unit square for N=26 circles."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def _validate_and_score(circles):
    """Check validity and compute score. circles: (N,3) array [x,y,r]."""
    if circles.shape != (N, 3):
        return None
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    if np.any(r <= 0):
        return None
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9) or np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return None
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx * dx + dy * dy)
    if np.any(dist < r[i_idx] + r[j_idx] - 1e-9):
        return None
    return float(np.sum(r))


def _grid_initial(n=N):
    """Create a grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append((cx, cy, r))
    return np.array(cells[:n])


def _random_initial(n=N, seed=42):
    """Create a random initial configuration with non-overlapping circles."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    circles[:, 2] = 0.01 + 0.01 * rng.rand(n)
    circles[:, 0] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    circles[:, 1] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    return circles


def _structured_initial(n=N, seed=0):
    """Create a structured initial configuration based on hexagonal packing."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    
    cols = 5
    rows = 6
    dx = 1.0 / (cols + 0.5)
    dy = 1.0 / (rows + 0.5)
    r_base = min(dx, dy) * 0.48
    
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            x_offset = 0.5 * dx if row % 2 == 1 else 0
            cx = (col + 0.5 + x_offset) * dx
            cy = (row + 0.5) * dy
            cx += 0.01 * rng.randn()
            cy += 0.01 * rng.randn()
            cx = np.clip(cx, r_base, 1 - r_base)
            cy = np.clip(cy, r_base, 1 - r_base)
            circles[idx, 0] = cx
            circles[idx, 1] = cy
            circles[idx, 2] = r_base
            idx += 1
    
    return circles


def _corner_weighted_initial(n=N, seed=0):
    """Create initial config with circles biased toward corners for better packing."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    
    # Place some circles near corners, others in center
    corner_positions = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    center_positions = [(0.5, 0.3), (0.3, 0.5), (0.7, 0.5), (0.5, 0.7),
                        (0.35, 0.35), (0.65, 0.35), (0.35, 0.65), (0.65, 0.65)]
    
    positions = corner_positions + center_positions
    while len(positions) < n:
        positions.append((0.1 + 0.8 * rng.rand(), 0.1 + 0.8 * rng.rand()))
    
    for i in range(n):
        cx, cy = positions[i]
        cx += 0.02 * rng.randn()
        cy += 0.02 * rng.randn()
        cx = np.clip(cx, 0.05, 0.95)
        cy = np.clip(cy, 0.05, 0.95)
        circles[i, 0] = cx
        circles[i, 1] = cy
        circles[i, 2] = 0.06 + 0.02 * rng.rand()
    
    return circles


def _resolve_overlaps(circles, max_iter=500):
    """Iteratively resolve overlaps by pushing circles apart."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for it in range(max_iter):
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = r[i_idx] + r[j_idx] - dist
        
        if not np.any(overlap > 1e-9):
            break
        
        for k in range(len(i_idx)):
            if overlap[k] > 1e-9:
                i = i_idx[k]
                j = j_idx[k]
                if dist[k] < 1e-12:
                    dx_k = 1e-6
                    dy_k = 0
                else:
                    dx_k = dx[k] / dist[k]
                    dy_k = dy[k] / dist[k]
                
                push = overlap[k] * 0.5
                x[i] -= push * dx_k
                y[i] -= push * dy_k
                x[j] += push * dx_k
                y[j] += push * dy_k
                
                shrink = overlap[k] * 0.1
                r[i] = max(r[i] - shrink, 0.001)
                r[j] = max(r[j] - shrink, 0.001)
        
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))
    
    return np.column_stack([x, y, r])


def _force_optimize(circles, time_limit=20, seed=0):
    """Physics-based force optimization with adaptive step size."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    lr = 0.008
    decay = 0.99999
    momentum_x = np.zeros(n)
    momentum_y = np.zeros(n)
    momentum_r = np.zeros(n)
    beta = 0.90
    
    def _get_near_pairs(cx, cy):
        points = np.column_stack([cx, cy])
        try:
            tri = Delaunay(points)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        a, b = simplex[i], simplex[j]
                        if a > b:
                            a, b = b, a
                        edges.add((a, b))
            if not edges:
                return i_idx, j_idx
            edge_list = sorted(edges)
            return np.array([e[0] for e in edge_list]), np.array([e[1] for e in edge_list])
        except Exception:
            return i_idx, j_idx
    
    it = 0
    no_improve = 0
    max_no_improve = 500
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        if it % 30 == 0:
            d_i, d_j = _get_near_pairs(x, y)
        
        dx = x[d_i] - x[d_j]
        dy = y[d_i] - y[d_j]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        
        overlap = r[d_i] + r[d_j] - dist
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        mask = overlap > 0
        if np.any(mask):
            rep = 200.0 * overlap[mask] / dist[mask]
            gx[d_i[mask]] -= rep * dx[mask] / dist[mask]
            gy[d_i[mask]] -= rep * dy[mask] / dist[mask]
            gx[d_j[mask]] += rep * dx[mask] / dist[mask]
            gy[d_j[mask]] += rep * dy[mask] / dist[mask]
            gr[d_i[mask]] += 150.0
            gr[d_j[mask]] += 150.0
        
        gx += np.where(x < r, 350.0, 0.0) - np.where(1 - x < r, 350.0, 0.0)
        gy += np.where(y < r, 350.0, 0.0) - np.where(1 - y < r, 350.0, 0.0)
        
        growth_pressure = 3.0 + 2.0 * (it / max(1, int(time_limit * 50)))
        gr -= growth_pressure
        
        momentum_x = beta * momentum_x + (1 - beta) * gx
        momentum_y = beta * momentum_y + (1 - beta) * gy
        momentum_r = beta * momentum_r + (1 - beta) * gr
        
        x += lr * momentum_x
        y += lr * momentum_y
        r += lr * momentum_r
        
        x = np.clip(x, 0.001, 0.999)
        y = np.clip(y, 0.001, 0.999)
        r = np.maximum(r, 0.001)
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))
        
        lr *= decay
        it += 1
        
        if it % 10 == 0:
            current_circles = np.column_stack([x, y, r])
            score = _validate_and_score(current_circles)
            if score is not None and score > best_score:
                no_improve = 0
                best_score = score
                best_circles = current_circles.copy()
            else:
                no_improve += 1
            
            if no_improve >= max_no_improve:
                break
    
    final_circles = np.column_stack([x, y, r])
    final_score = _validate_and_score(final_circles)
    if final_score is not None and final_score > best_score:
        best_score = final_score
        best_circles = final_circles
    
    return best_circles, best_score


def _lbfgs_optimize(circles, time_limit=15, seed=0):
    """Use L-BFGS-B with smooth log-sum-exp penalty for robust local optimization."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    params = np.concatenate([x, y, r])
    
    pw = 100.0
    beta_lse = 80.0
    
    def _lse(v):
        if len(v) == 0:
            return 0.0
        v_max = np.max(v)
        if v_max < 1e-15:
            return 0.0
        return float(v_max + np.log(np.sum(np.exp(v - v_max)))) / beta_lse
    
    def objective(p):
        if time.time() - start_time > time_limit:
            return 1e10
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        
        bv = np.maximum(0, rr - xx) + np.maximum(0, rr - (1 - xx)) + \
              np.maximum(0, rr - yy) + np.maximum(0, rr - (1 - yy))
        
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        ov = np.maximum(0, rr[i_idx] + rr[j_idx] - dist)
        
        bv_pen = _lse(pw * bv) if np.max(bv) > 0 else 0.0
        ov_pen = _lse(pw * ov) if np.max(ov) > 0 else 0.0
        
        return -np.sum(rr) + bv_pen + ov_pen
    
    def gradient(p):
        if time.time() - start_time > time_limit:
            return np.zeros_like(p)
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        gx += np.where(xx < rr, pw, 0.0) - np.where(1 - xx < rr, pw, 0.0)
        gy += np.where(yy < rr, pw, 0.0) - np.where(1 - yy < rr, pw, 0.0)
        gr += np.where(xx < rr, pw, 0.0) + np.where(1 - xx < rr, pw, 0.0)
        gr += np.where(yy < rr, pw, 0.0) + np.where(1 - yy < rr, pw, 0.0)
        
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        overlap = rr[i_idx] + rr[j_idx] - dist
        
        mask = overlap > 0
        if np.any(mask):
            ov_masked = overlap[mask]
            w = np.exp(pw * ov_masked) / np.sum(np.exp(pw * ov_masked))
            rep = pw * w / dist[mask]
            gx[i_idx[mask]] -= rep * dx[mask] / dist[mask]
            gy[i_idx[mask]] -= rep * dy[mask] / dist[mask]
            gx[j_idx[mask]] += rep * dx[mask] / dist[mask]
            gy[j_idx[mask]] += rep * dy[mask] / dist[mask]
            gr[i_idx[mask]] += pw * w
            gr[j_idx[mask]] += pw * w
        
        gr -= 1.0
        
        return np.concatenate([gx, gy, gr])
    
    bounds = [(0.001, 0.999)] * n + [(0.001, 0.999)] * n + [(0.001, 0.49)] * n
    
    try:
        result = minimize(objective, params, method='L-BFGS-B', jac=gradient,
                          bounds=bounds, options={'maxiter': 800, 'maxfun': 5000,
                                                   'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
    except Exception:
        return circles.copy(), _validate_and_score(circles) or 0.0
    
    if time.time() - start_time > time_limit:
        return circles.copy(), _validate_and_score(circles) or 0.0
    
    new_params = result.x
    new_x = np.clip(new_params[:n], 0.001, 0.999)
    new_y = np.clip(new_params[n:2*n], 0.001, 0.999)
    new_r = np.maximum(new_params[2*n:3*n], 0.001)
    max_r = np.minimum(np.minimum(new_x, 1 - new_x), np.minimum(new_y, 1 - new_y))
    new_r = np.minimum(new_r, np.maximum(max_r, 0.001))
    
    new_circles = np.column_stack([new_x, new_y, new_r])
    new_score = _validate_and_score(new_circles)
    input_score = _validate_and_score(circles)
    if input_score is None:
        input_score = 0.0
    if new_score is None:
        new_score = 0.0
    
    if new_score >= input_score:
        return new_circles, new_score
    else:
        return circles.copy(), input_score


def _trust_region_optimize(circles, time_limit=25, seed=0):
    """Trust-region reflective algorithm for NLP formulation of circle packing.
    
    Minimizes sum of squared residuals between pairwise distances and radius sums,
    with hard constraints for non-overlap and boundary containment.
    """
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    n_pairs = len(i_idx)
    
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0
    
    params = np.concatenate([x, y, r])
    
    def objective(p):
        if time.time() - start_time > time_limit:
            return 1e10
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        
        # Primary objective: minimize sum of squared residuals (dist - r_i - r_j)^2
        # This pushes circles into contact
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        residual = dist - rr[i_idx] - rr[j_idx]
        obj = np.sum(residual ** 2)
        
        # Penalty for boundary violations
        bv = np.maximum(0, rr - xx) + np.maximum(0, rr - (1 - xx)) + \
              np.maximum(0, rr - yy) + np.maximum(0, rr - (1 - yy))
        obj += 500.0 * np.sum(bv ** 2)
        
        # Penalty for overlaps (hard constraint)
        ov = np.maximum(0, rr[i_idx] + rr[j_idx] - dist)
        obj += 1000.0 * np.sum(ov ** 2)
        
        # Secondary: maximize sum of radii (negative)
        obj -= 2.0 * np.sum(rr)
        
        return obj
    
    def gradient(p):
        if time.time() - start_time > time_limit:
            return np.zeros_like(p)
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        # Gradient of sum(residual^2) where residual = dist - r_i - r_j
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist_safe = np.maximum(dist, 1e-12)
        residual = dist_safe - rr[i_idx] - rr[j_idx]
        
        # d(residual^2)/d(x_i) = 2 * residual * (dx_i / dist)
        for k in range(n_pairs):
            i = i_idx[k]
            j = j_idx[k]
            factor = 2.0 * residual[k] / dist_safe[k]
            gx[i] += factor * dx[k]
            gy[i] += factor * dy[k]
            gx[j] -= factor * dx[k]
            gy[j] -= factor * dy[k]
            gr[i] += 2.0 * residual[k]
            gr[j] += 2.0 * residual[k]
        
        # Boundary penalty gradients
        bv_x = np.maximum(0, rr - xx)
        bv_1mx = np.maximum(0, rr - (1 - xx))
        bv_y = np.maximum(0, rr - yy)
        bv_1my = np.maximum(0, rr - (1 - yy))
        
        gx += 1000.0 * bv_x - 1000.0 * bv_1mx
        gy += 1000.0 * bv_y - 1000.0 * bv_1my
        gr += 1000.0 * (bv_x + bv_1mx + bv_y + bv_1my)
        
        # Overlap penalty gradients
        ov = np.maximum(0, rr[i_idx] + rr[j_idx] - dist_safe)
        mask = ov > 0
        if np.any(mask):
            for k in np.where(mask)[0]:
                i = i_idx[k]
                j = j_idx[k]
                factor = 2000.0 * ov[k] / dist_safe[k]
                gx[i] -= factor * dx[k]
                gy[i] -= factor * dy[k]
                gx[j] += factor * dx[k]
                gy[j] += factor * dy[k]
                gr[i] += 2000.0 * ov[k]
                gr[j] += 2000.0 * ov[k]
        
        # Secondary objective gradient
        gr -= 2.0
        
        return np.concatenate([gx, gy, gr])
    
    bounds = [(0.001, 0.999)] * n + [(0.001, 0.999)] * n + [(0.001, 0.49)] * n
    
    try:
        result = minimize(objective, params, method='trust-constr', jac=gradient,
                          bounds=bounds, options={'maxiter': 500, 'gtol': 1e-10,
                                                   'xtol': 1e-12, 'max_nfev': 3000,
                                                   'verbose': 0})
    except Exception:
        # Fallback to L-BFGS-B
        try:
            result = minimize(objective, params, method='L-BFGS-B', jac=gradient,
                              bounds=bounds, options={'maxiter': 1000, 'maxfun': 5000,
                                                       'ftol': 1e-14, 'gtol': 1e-12})
        except Exception:
            return circles.copy(), best_score
    
    if time.time() - start_time > time_limit:
        return circles.copy(), best_score
    
    new_params = result.x
    new_x = np.clip(new_params[:n], 0.001, 0.999)
    new_y = np.clip(new_params[n:2*n], 0.001, 0.999)
    new_r = np.maximum(new_params[2*n:3*n], 0.001)
    max_r = np.minimum(np.minimum(new_x, 1 - new_x), np.minimum(new_y, 1 - new_y))
    new_r = np.minimum(new_r, np.maximum(max_r, 0.001))
    
    new_circles = np.column_stack([new_x, new_y, new_r])
    new_score = _validate_and_score(new_circles)
    if new_score is None:
        new_score = 0.0
    
    if new_score >= best_score:
        return new_circles, new_score
    else:
        return best_circles, best_score


def _simulated_annealing(circles, time_limit=30, seed=0):
    """Simulated annealing with fast penalty-based scoring."""
    rng = np.random.RandomState(seed)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def _penalty_score(cx, cy, cr, pw=80.0):
        if np.any(cr <= 0):
            return -1e6
        bv = np.maximum(0, cr - cx) + np.maximum(0, cr - (1 - cx)) + \
              np.maximum(0, cr - cy) + np.maximum(0, cr - (1 - cy))
        dx = cx[i_idx] - cx[j_idx]
        dy = cy[i_idx] - cy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        ov = np.maximum(0, cr[i_idx] + cr[j_idx] - dist)
        return float(np.sum(cr) - pw * (np.sum(bv) + np.sum(ov)))
    
    def _strict_score(cx, cy, cr):
        if np.any(cr <= 0):
            return None
        if np.any(cx - cr < -1e-9) or np.any(cx + cr > 1 + 1e-9) or \
           np.any(cy - cr < -1e-9) or np.any(cy + cr > 1 + 1e-9):
            return None
        dx = cx[i_idx] - cx[j_idx]
        dy = cy[i_idx] - cy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        if np.any(dist < cr[i_idx] + cr[j_idx] - 1e-9):
            return None
        return float(np.sum(cr))
    
    best_strict_score = _strict_score(x, y, r)
    best_circles = np.column_stack([x, y, r]).copy()
    if best_strict_score is None:
        best_strict_score = 0.0
    
    current_ps = _penalty_score(x, y, r)
    
    t_init = 0.06
    t_min = 0.0005
    max_iter = 60000
    cooling = (t_min / t_init) ** (1.0 / max_iter)
    t = t_init
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        i = rng.randint(n)
        move_type = rng.rand()
        
        new_x = x.copy()
        new_y = y.copy()
        new_r = r.copy()
        
        if move_type < 0.40:
            delta = 0.015 * rng.randn(2)
            new_x[i] += delta[0]
            new_y[i] += delta[1]
        elif move_type < 0.70:
            new_r[i] *= (1 + 0.08 * rng.randn())
            new_r[i] = max(0.005, min(new_r[i], 0.5))
        elif move_type < 0.85:
            j = rng.randint(n)
            if j != i:
                new_r[i], new_r[j] = new_r[j], new_r[i]
        else:
            j = rng.randint(n)
            if j != i:
                delta = 0.005 * rng.randn()
                new_r[i] += delta
                new_r[j] -= delta
                new_r[i] = max(0.005, new_r[i])
                new_r[j] = max(0.005, new_r[j])
        
        new_ps = _penalty_score(new_x, new_y, new_r)
        delta = new_ps - current_ps
        if delta > 0 or rng.rand() < np.exp(delta / max(t, 1e-10)):
            x, y, r = new_x, new_y, new_r
            current_ps = new_ps
            
            if it % 40 == 0:
                s = _strict_score(x, y, r)
                if s is not None and s > best_strict_score:
                    best_strict_score = s
                    best_circles = np.column_stack([x, y, r]).copy()
        
        t *= cooling
    
    s = _strict_score(x, y, r)
    if s is not None and s > best_strict_score:
        best_strict_score = s
        best_circles = np.column_stack([x, y, r]).copy()
    
    return best_circles, best_strict_score


def _delaunay_polish(circles, time_limit=15, seed=0):
    """High-precision polish using Delaunay triangulation for contact pairs."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def _get_delaunay_pairs(cx, cy):
        points = np.column_stack([cx, cy])
        try:
            tri = Delaunay(points)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        a, b = simplex[i], simplex[j]
                        if a > b:
                            a, b = b, a
                        edges.add((a, b))
            if not edges:
                return i_idx, j_idx
            edge_list = sorted(edges)
            return np.array([e[0] for e in edge_list]), np.array([e[1] for e in edge_list])
        except Exception:
            return i_idx, j_idx
    
    lr = 0.003
    decay = 0.99999
    no_improve_count = 0
    max_no_improve = 600
    
    for it in range(25000):
        if time.time() - start_time > time_limit:
            break
        
        if it % 30 == 0:
            d_i, d_j = _get_delaunay_pairs(x, y)
        
        dx = x[d_i] - x[d_j]
        dy = y[d_i] - y[d_j]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        
        overlap = r[d_i] + r[d_j] - dist
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        mask = overlap > 0
        if np.any(mask):
            rep = 150.0 * overlap[mask] / dist[mask]
            gx[d_i[mask]] -= rep * dx[mask] / dist[mask]
            gy[d_i[mask]] -= rep * dy[mask] / dist[mask]
            gx[d_j[mask]] += rep * dx[mask] / dist[mask]
            gy[d_j[mask]] += rep * dy[mask] / dist[mask]
            gr[d_i[mask]] += 120.0
            gr[d_j[mask]] += 120.0
        
        gx += np.where(x < r, 300.0, 0.0) - np.where(1 - x < r, 300.0, 0.0)
        gy += np.where(y < r, 300.0, 0.0) - np.where(1 - y < r, 300.0, 0.0)
        
        gr -= 2.0
        
        x += lr * gx
        y += lr * gy
        r += lr * gr
        
        x = np.clip(x, 0.001, 0.999)
        y = np.clip(y, 0.001, 0.999)
        r = np.maximum(r, 0.001)
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))
        
        lr *= decay
        
        if it % 10 == 0:
            current_circles = np.column_stack([x, y, r])
            score = _validate_and_score(current_circles)
            if score is not None and score > best_score + 1e-14:
                no_improve_count = 0
                best_score = score
                best_circles = current_circles.copy()
            else:
                no_improve_count += 1
            
            if no_improve_count >= max_no_improve:
                break
    
    final_circles = np.column_stack([x, y, r])
    final_score = _validate_and_score(final_circles)
    if final_score is not None and final_score > best_score:
        best_score = final_score
        best_circles = final_circles
    
    return best_circles, best_score


def _refine_radii(circles, time_limit=10):
    """Try to grow radii while maintaining feasibility using iterative adjustment."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0
    
    for it in range(8000):
        if time.time() - start_time > time_limit:
            break
        
        max_allowed = np.copy(r)
        
        boundary_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        max_allowed = np.minimum(max_allowed, boundary_max)
        
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            max_allowed[i] = min(max_allowed[i], max(dist - r[j], 0.001))
            max_allowed[j] = min(max_allowed[j], max(dist - r[i], 0.001))
        
        headroom = max_allowed - r
        if np.max(headroom) < 1e-7:
            break
        
        n_grow = min(4, n)
        top_indices = np.argsort(headroom)[-n_grow:]
        grow_amount = 0.0015
        for idx in top_indices:
            r[idx] = min(r[idx] + grow_amount, max_allowed[idx])
        
        new_circles = np.column_stack([x, y, r])
        new_circles = _resolve_overlaps(new_circles, max_iter=150)
        
        score = _validate_and_score(new_circles)
        if score is not None and score > best_score:
            best_score = score
            best_circles = new_circles.copy()
            x = new_circles[:, 0].copy()
            y = new_circles[:, 1].copy()
            r = new_circles[:, 2].copy()
    
    return best_circles, best_score


def _high_precision_polish(circles, time_limit=20, seed=0):
    """High-precision polish: full-pairwise force relaxation + L-BFGS with smooth LSE penalty."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    # Phase 1: Full-pairwise force-based relaxation (no Delaunay restriction)
    lr = 0.004
    decay = 0.99998
    no_improve = 0
    max_no_improve = 400
    it = 0
    phase1_budget = time_limit * 0.45
    
    while True:
        if time.time() - start_time > phase1_budget:
            break
        
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        
        overlap = r[i_idx] + r[j_idx] - dist
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        mask = overlap > -0.003
        if np.any(mask):
            # Hertzian-style: force ~ overlap^(3/2) for more realistic contact
            smooth_ov = np.maximum(overlap[mask], 0.0)
            rep = 300.0 * np.sqrt(smooth_ov + 1e-8) / (dist[mask] + 0.01)
            gx[i_idx[mask]] -= rep * dx[mask] / dist[mask]
            gy[i_idx[mask]] -= rep * dy[mask] / dist[mask]
            gx[j_idx[mask]] += rep * dx[mask] / dist[mask]
            gy[j_idx[mask]] += rep * dy[mask] / dist[mask]
            gr[i_idx[mask]] += 180.0
            gr[j_idx[mask]] += 180.0
        
        # Boundary forces
        gx += np.where(x < r, 500.0, 0.0) - np.where(1 - x < r, 500.0, 0.0)
        gy += np.where(y < r, 500.0, 0.0) - np.where(1 - y < r, 500.0, 0.0)
        
        # Growth pressure
        gr -= 3.5
        
        x += lr * gx
        y += lr * gy
        r += lr * gr
        
        x = np.clip(x, 0.0005, 0.9995)
        y = np.clip(y, 0.0005, 0.9995)
        r = np.maximum(r, 0.001)
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))
        
        lr *= decay
        it += 1
        
        if it % 5 == 0:
            current_circles = np.column_stack([x, y, r])
            score = _validate_and_score(current_circles)
            if score is not None and score > best_score + 1e-14:
                no_improve = 0
                best_score = score
                best_circles = current_circles.copy()
            else:
                no_improve += 1
            
            if no_improve >= max_no_improve:
                break
    
    # Phase 2: L-BFGS with smooth full-pairwise LSE penalty
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 4:
        x0 = best_circles[:, 0].copy()
        y0 = best_circles[:, 1].copy()
        r0 = best_circles[:, 2].copy()
        params0 = np.concatenate([x0, y0, r0])
        
        pw = 120.0
        beta_lse = 100.0
        
        def _lse(v):
            if len(v) == 0:
                return 0.0
            v_max = np.max(v)
            if v_max < 1e-15:
                return 0.0
            return float(v_max + np.log(np.sum(np.exp(v - v_max)))) / beta_lse
        
        def objective(p):
            if time.time() - start_time > time_limit:
                return 1e10
            xx = p[:n]
            yy = p[n:2*n]
            rr = p[2*n:3*n]
            
            bv = np.maximum(0, rr - xx) + np.maximum(0, rr - (1 - xx)) + \
                  np.maximum(0, rr - yy) + np.maximum(0, rr - (1 - yy))
            
            dx = xx[i_idx] - xx[j_idx]
            dy = yy[i_idx] - yy[j_idx]
            dist = np.sqrt(dx * dx + dy * dy)
            dist = np.maximum(dist, 1e-12)
            ov = np.maximum(0, rr[i_idx] + rr[j_idx] - dist)
            
            bv_pen = _lse(pw * bv) if np.max(bv) > 0 else 0.0
            ov_pen = _lse(pw * ov) if np.max(ov) > 0 else 0.0
            
            return -np.sum(rr) + bv_pen + ov_pen
        
        def gradient(p):
            if time.time() - start_time > time_limit:
                return np.zeros_like(p)
            xx = p[:n]
            yy = p[n:2*n]
            rr = p[2*n:3*n]
            
            gx = np.zeros(n)
            gy = np.zeros(n)
            gr = np.zeros(n)
            
            gx += np.where(xx < rr, pw, 0.0) - np.where(1 - xx < rr, pw, 0.0)
            gy += np.where(yy < rr, pw, 0.0) - np.where(1 - yy < rr, pw, 0.0)
            gr += np.where(xx < rr, pw, 0.0) + np.where(1 - xx < rr, pw, 0.0)
            gr += np.where(yy < rr, pw, 0.0) + np.where(1 - yy < rr, pw, 0.0)
            
            dx = xx[i_idx] - xx[j_idx]
            dy = yy[i_idx] - yy[j_idx]
            dist = np.sqrt(dx * dx + dy * dy)
            dist = np.maximum(dist, 1e-12)
            overlap = rr[i_idx] + rr[j_idx] - dist
            
            mask = overlap > 0
            if np.any(mask):
                ov_masked = overlap[mask]
                w = np.exp(pw * ov_masked) / np.sum(np.exp(pw * ov_masked))
                rep = pw * w / dist[mask]
                gx[i_idx[mask]] -= rep * dx[mask] / dist[mask]
                gy[i_idx[mask]] -= rep * dy[mask] / dist[mask]
                gx[j_idx[mask]] += rep * dx[mask] / dist[mask]
                gy[j_idx[mask]] += rep * dy[mask] / dist[mask]
                gr[i_idx[mask]] += pw * w
                gr[j_idx[mask]] += pw * w
            
            gr -= 1.0
            
            return np.concatenate([gx, gy, gr])
        
        bounds = [(0.0005, 0.9995)] * n + [(0.0005, 0.9995)] * n + [(0.001, 0.49)] * n
        
        try:
            result = minimize(objective, params0, method='L-BFGS-B', jac=gradient,
                              bounds=bounds, options={'maxiter': 600, 'maxfun': 4000,
                                                       'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50})
            new_params = result.x
            new_x = np.clip(new_params[:n], 0.0005, 0.9995)
            new_y = np.clip(new_params[n:2*n], 0.0005, 0.9995)
            new_r = np.maximum(new_params[2*n:3*n], 0.001)
            max_r = np.minimum(np.minimum(new_x, 1 - new_x), np.minimum(new_y, 1 - new_y))
            new_r = np.minimum(new_r, np.maximum(max_r, 0.001))
            
            new_circles = np.column_stack([new_x, new_y, new_r])
            new_score = _validate_and_score(new_circles)
            if new_score is not None and new_score > best_score:
                best_score = new_score
                best_circles = new_circles
        except Exception:
            pass
    
    # Phase 3: Greedy radius growth with micro-adjustments
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 2:
        x = best_circles[:, 0].copy()
        y = best_circles[:, 1].copy()
        r = best_circles[:, 2].copy()
        
        for polish_iter in range(400):
            if time.time() - start_time > time_limit:
                break
            
            # Compute max allowed radius for each circle
            max_allowed = np.copy(r)
            boundary_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
            max_allowed = np.minimum(max_allowed, boundary_max)
            
            for k in range(len(i_idx)):
                i = i_idx[k]
                j = j_idx[k]
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                max_allowed[i] = min(max_allowed[i], max(dist - r[j], 0.0005))
                max_allowed[j] = min(max_allowed[j], max(dist - r[i], 0.0005))
            
            headroom = max_allowed - r
            if np.max(headroom) < 1e-10:
                break
            
            best_idx = np.argmax(headroom)
            if headroom[best_idx] < 1e-9:
                break
            
            improved = False
            for grow_frac in [0.5, 0.3, 0.1, 0.05]:
                grow_amount = headroom[best_idx] * grow_frac
                if grow_amount < 1e-10:
                    continue
                new_r = r.copy()
                new_r[best_idx] = min(r[best_idx] + grow_amount, max_allowed[best_idx])
                new_circles = np.column_stack([x, y, new_r])
                
                sc = _validate_and_score(new_circles)
                if sc is not None and sc > best_score + 1e-12:
                    r = new_r
                    best_score = sc
                    best_circles = np.column_stack([x, y, r]).copy()
                    improved = True
                    break
            
            if not improved:
                # Micro-adjust positions of contacting pairs to create room
                dx_all = x[i_idx] - x[j_idx]
                dy_all = y[i_idx] - y[j_idx]
                dist_all = np.sqrt(dx_all*dx_all + dy_all*dy_all)
                dist_all = np.maximum(dist_all, 1e-12)
                gap_all = dist_all - (r[i_idx] + r[j_idx])
                contact_mask = gap_all < 0.001
                
                if np.any(contact_mask):
                    for k in np.where(contact_mask)[0]:
                        i = i_idx[k]
                        j = j_idx[k]
                        dx_k = dx_all[k] / dist_all[k]
                        dy_k = dy_all[k] / dist_all[k]
                        push = 0.00015
                        x[i] -= push * dx_k
                        y[i] -= push * dy_k
                        x[j] += push * dx_k
                        y[j] += push * dy_k
                    
                    x = np.clip(x, 0.0005, 0.9995)
                    y = np.clip(y, 0.0005, 0.9995)
                    sc = _validate_and_score(np.column_stack([x, y, r]))
                    if sc is None or sc < best_score:
                        x = best_circles[:, 0].copy()
                        y = best_circles[:, 1].copy()
                        r = best_circles[:, 2].copy()
                else:
                    break
    
    return best_circles, best_score


def construct():
    """Construct optimal circle packing in unit square for N=26 circles."""
    start_time = time.time()
    time_budget = 145  # Keep 5s safety margin before 150s kill
    
    best_circles = None
    best_score = 0.0
    
    initializations = [
        ('structured1', _structured_initial(N, seed=1)),
        ('structured2', _structured_initial(N, seed=2)),
        ('structured3', _structured_initial(N, seed=7)),
        ('structured4', _structured_initial(N, seed=13)),
        ('corner1', _corner_weighted_initial(N, seed=1)),
        ('corner2', _corner_weighted_initial(N, seed=5)),
        ('grid', _grid_initial(N)),
        ('random1', _random_initial(N, seed=10)),
        ('random2', _random_initial(N, seed=20)),
        ('random3', _random_initial(N, seed=42)),
    ]
    
    # Phase 1: Explore with force optimization + SA on diverse starts
    for name, init_circles in initializations:
        elapsed = time.time() - start_time
        if elapsed > time_budget - 55:
            break
        
        resolved = _resolve_overlaps(init_circles, max_iter=300)
        
        # Force optimization
        remaining = time_budget - (time.time() - start_time) - 45
        if remaining > 12:
            loc_circles, loc_score = _force_optimize(
                resolved, time_limit=min(remaining * 0.25, 12), seed=hash(name) % 10000)
            if loc_score > best_score:
                best_score = loc_score
                best_circles = loc_circles
        
        # Trust-region optimization
        remaining = time_budget - (time.time() - start_time) - 38
        if remaining > 15:
            tr_circles, tr_score = _trust_region_optimize(
                resolved, time_limit=min(remaining * 0.30, 15), seed=hash(name) % 10000)
            if tr_score > best_score:
                best_score = tr_score
                best_circles = tr_circles
        
        # Simulated annealing
        remaining = time_budget - (time.time() - start_time) - 28
        if remaining > 12:
            sa_circles, sa_score = _simulated_annealing(
                best_circles if best_circles is not None else resolved,
                time_limit=min(remaining * 0.35, 15), seed=hash(name) % 10000)
            if sa_score > best_score:
                best_score = sa_score
                best_circles = sa_circles
    
    # Phase 2: Exploit with L-BFGS, trust-region, and high-precision polish
    if best_circles is not None:
        # L-BFGS
        remaining = time_budget - (time.time() - start_time) - 35
        if remaining > 10:
            lbfgs_circles, lbfgs_score = _lbfgs_optimize(
                best_circles, time_limit=min(remaining * 0.30, 10), seed=42)
            if lbfgs_score > best_score:
                best_score = lbfgs_score
                best_circles = lbfgs_circles
        
        # Trust-region refinement
        remaining = time_budget - (time.time() - start_time) - 28
        if remaining > 12:
            tr2_circles, tr2_score = _trust_region_optimize(
                best_circles, time_limit=min(remaining * 0.35, 14), seed=7777)
            if tr2_score > best_score:
                best_score = tr2_score
                best_circles = tr2_circles
        
        # High-precision polish (first pass)
        remaining = time_budget - (time.time() - start_time) - 22
        if remaining > 10:
            hp_circles, hp_score = _high_precision_polish(
                best_circles, time_limit=min(remaining * 0.45, 15), seed=9999)
            if hp_score > best_score:
                best_score = hp_score
                best_circles = hp_circles
        
        # Delaunay polish
        remaining = time_budget - (time.time() - start_time) - 15
        if remaining > 8:
            del_circles, del_score = _delaunay_polish(
                best_circles, time_limit=min(remaining * 0.35, 8), seed=9999)
            if del_score > best_score:
                best_score = del_score
                best_circles = del_circles
        
        # Radius refinement
        remaining = time_budget - (time.time() - start_time) - 10
        if remaining > 6:
            ref_circles, ref_score = _refine_radii(
                best_circles, time_limit=min(remaining * 0.30, 6))
            if ref_score > best_score:
                best_score = ref_score
                best_circles = ref_circles
        
        # Second L-BFGS pass
        remaining = time_budget - (time.time() - start_time) - 8
        if remaining > 5:
            lbfgs2_circles, lbfgs2_score = _lbfgs_optimize(
                best_circles, time_limit=min(remaining * 0.40, 6), seed=12345)
            if lbfgs2_score > best_score:
                best_score = lbfgs2_score
                best_circles = lbfgs2_circles
        
        # Final high-precision polish (most aggressive)
        remaining = time_budget - (time.time() - start_time) - 3
        if remaining > 5:
            hp2_circles, hp2_score = _high_precision_polish(
                best_circles, time_limit=min(remaining * 0.90, 18), seed=11111)
            if hp2_score > best_score:
                best_score = hp2_score
                best_circles = hp2_circles
        
        # One more L-BFGS pass after final polish
        remaining = time_budget - (time.time() - start_time) - 2
        if remaining > 4:
            lbfgs3_circles, lbfgs3_score = _lbfgs_optimize(
                best_circles, time_limit=min(remaining * 0.70, 8), seed=22222)
            if lbfgs3_score > best_score:
                best_score = lbfgs3_score
                best_circles = lbfgs3_circles
    
    # Final validation
    if best_circles is None:
        return _grid_initial(N)
    
    final_score = _validate_and_score(best_circles)
    if final_score is None:
        return _grid_initial(N)
    
    return best_circles

# EVOLVE-BLOCK-END
