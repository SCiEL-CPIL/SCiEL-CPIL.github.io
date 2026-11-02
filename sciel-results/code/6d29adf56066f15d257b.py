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
    
    # Hexagonal grid: 5 cols x 6 rows = 30, use 26
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


def _hex_dense_initial(n=N, seed=0):
    """Denser hexagonal initialization for better starting geometry."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    
    # Try a tighter hexagonal packing
    r_start = 0.08
    dx = 2 * r_start * 1.1
    dy = dx * np.sqrt(3) / 2
    
    x_offset = 0.5 - (4 * dx) / 2.0
    y_offset = 0.5 - (5 * dy) / 2.0
    
    for row in range(6):
        for col in range(5):
            cx = x_offset + col * dx + (dx / 2 if row % 2 else 0)
            cy = y_offset + row * dy
            cx = np.clip(cx, r_start, 1 - r_start)
            cy = np.clip(cy, r_start, 1 - r_start)
            circles[row * 5 + col, 0] = cx
            circles[row * 5 + col, 1] = cy
            circles[row * 5 + col, 2] = r_start
            
    circles = circles[:n]
    circles[:, 0] += 0.005 * rng.randn(n)
    circles[:, 1] += 0.005 * rng.randn(n)
    circles[:, 0] = np.clip(circles[:, 0], 0.01, 0.99)
    circles[:, 1] = np.clip(circles[:, 1], 0.01, 0.99)
    
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


def _get_delaunay_pairs(cx, cy, fallback_i, fallback_j):
    """Get Delaunay triangulation edges as index pairs."""
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
            return fallback_i, fallback_j
        edge_list = sorted(edges)
        return np.array([e[0] for e in edge_list]), np.array([e[1] for e in edge_list])
    except Exception:
        return fallback_i, fallback_j


def _force_optimize(circles, time_limit=20, seed=0):
    """Physics-based force optimization with adaptive step size and momentum.
    
    Uses repulsive forces between overlapping circles, boundary constraints,
    and radius growth pressure. Employs Delaunay edges for efficiency.
    """
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
    
    # Adaptive learning rate with slower decay
    lr = 0.008
    decay = 0.99999
    momentum_x = np.zeros(n)
    momentum_y = np.zeros(n)
    momentum_r = np.zeros(n)
    beta = 0.85
    
    it = 0
    no_improve = 0
    max_no_improve = 500
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        # Recompute Delaunay pairs periodically
        if it % 30 == 0:
            d_i, d_j = _get_delaunay_pairs(x, y, i_idx, j_idx)
        else:
            d_i, d_j = _get_delaunay_pairs(x, y, i_idx, j_idx)
        
        # Compute forces
        dx = x[d_i] - x[d_j]
        dy = y[d_i] - y[d_j]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        
        overlap = r[d_i] + r[d_j] - dist
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        # Repulsion from overlapping pairs
        mask = overlap > 0
        if np.any(mask):
            rep = 150.0 * overlap[mask] / dist[mask]
            gx[d_i[mask]] -= rep * dx[mask] / dist[mask]
            gy[d_i[mask]] -= rep * dy[mask] / dist[mask]
            gx[d_j[mask]] += rep * dx[mask] / dist[mask]
            gy[d_j[mask]] += rep * dy[mask] / dist[mask]
            gr[d_i[mask]] += 100.0
            gr[d_j[mask]] += 100.0
        
        # Boundary forces
        gx += np.where(x < r, 200.0, 0.0) - np.where(1 - x < r, 200.0, 0.0)
        gy += np.where(y < r, 200.0, 0.0) - np.where(1 - y < r, 200.0, 0.0)
        
        # Radius growth pressure
        gr -= 1.0
        
        # Update momentum
        momentum_x = beta * momentum_x + (1 - beta) * gx
        momentum_y = beta * momentum_y + (1 - beta) * gy
        momentum_r = beta * momentum_r + (1 - beta) * gr
        
        # Apply updates with momentum
        x += lr * momentum_x
        y += lr * momentum_y
        r += lr * momentum_r
        
        # Clamp
        x = np.clip(x, 0.001, 0.999)
        y = np.clip(y, 0.001, 0.999)
        r = np.maximum(r, 0.001)
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))
        
        lr *= decay
        it += 1
        
        # Check improvement periodically
        if it % 20 == 0:
            current_circles = np.column_stack([x, y, r])
            score = _validate_and_score(current_circles)
            if score is not None and score > best_score:
                if score - best_score < 1e-12:
                    no_improve += 1
                else:
                    no_improve = 0
                best_score = score
                best_circles = current_circles.copy()
            else:
                no_improve += 1
            
            if no_improve >= max_no_improve:
                break
    
    # Final check
    final_circles = np.column_stack([x, y, r])
    final_score = _validate_and_score(final_circles)
    if final_score is not None and final_score > best_score:
        best_score = final_score
        best_circles = final_circles
    
    return best_circles, best_score


def _delaunay_polish(circles, time_limit=15, seed=0):
    """High-precision polish using Delaunay triangulation to identify contact pairs."""
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
    
    lr = 0.005
    decay = 0.99995
    no_improve_count = 0
    max_no_improve = 500
    
    for it in range(20000):
        if time.time() - start_time > time_limit:
            break
        
        # Get current Delaunay pairs
        if it % 50 == 0:
            d_i, d_j = _get_delaunay_pairs(x, y, i_idx, j_idx)
        
        # Compute forces on Delaunay edges only
        dx = x[d_i] - x[d_j]
        dy = y[d_i] - y[d_j]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        
        overlap = r[d_i] + r[d_j] - dist
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        # Repulsion from overlapping Delaunay pairs
        mask = overlap > 0
        if np.any(mask):
            rep = 100.0 * overlap[mask] / dist[mask]
            gx[d_i[mask]] -= rep * dx[mask] / dist[mask]
            gy[d_i[mask]] -= rep * dy[mask] / dist[mask]
            gx[d_j[mask]] += rep * dx[mask] / dist[mask]
            gy[d_j[mask]] += rep * dy[mask] / dist[mask]
            gr[d_i[mask]] += 100.0
            gr[d_j[mask]] += 100.0
        
        # Boundary forces
        gx += np.where(x < r, 200.0, 0.0) - np.where(1 - x < r, 200.0, 0.0)
        gy += np.where(y < r, 200.0, 0.0) - np.where(1 - y < r, 200.0, 0.0)
        
        # Radius growth pressure
        gr -= 1.0
        
        # Apply updates
        x += lr * gx
        y += lr * gy
        r += lr * gr
        
        # Clamp
        x = np.clip(x, 0.001, 0.999)
        y = np.clip(y, 0.001, 0.999)
        r = np.maximum(r, 0.001)
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))
        
        lr *= decay
        
        # Check improvement every 20 iterations
        if it % 20 == 0:
            current_circles = np.column_stack([x, y, r])
            score = _validate_and_score(current_circles)
            if score is not None and score > best_score:
                if score - best_score < 1e-12:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                best_score = score
                best_circles = current_circles.copy()
            else:
                no_improve_count += 1
            
            if no_improve_count >= max_no_improve:
                break
    
    # Final check
    final_circles = np.column_stack([x, y, r])
    final_score = _validate_and_score(final_circles)
    if final_score is not None and final_score > best_score:
        best_score = final_score
        best_circles = final_circles
    
    return best_circles, best_score


def _lbfgs_optimize(circles, time_limit=15, seed=0):
    """Use L-BFGS-B with smooth penalty method for robust local optimization."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    params = np.concatenate([x, y, r])
    
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
        
        return -np.sum(rr) + 50.0 * (np.sum(bv) + np.sum(ov))
    
    def gradient(p):
        if time.time() - start_time > time_limit:
            return np.zeros_like(p)
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        gx += np.where(xx < rr, 50.0, 0.0) - np.where(1 - xx < rr, 50.0, 0.0)
        gy += np.where(yy < rr, 50.0, 0.0) - np.where(1 - yy < rr, 50.0, 0.0)
        gr += np.where(xx < rr, 50.0, 0.0) + np.where(1 - xx < rr, 50.0, 0.0)
        gr += np.where(yy < rr, 50.0, 0.0) + np.where(1 - yy < rr, 50.0, 0.0)
        
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        overlap = rr[i_idx] + rr[j_idx] - dist
        
        mask = overlap > 0
        if np.any(mask):
            rep = 50.0 * overlap[mask] / dist[mask]
            gx[i_idx[mask]] -= rep * dx[mask] / dist[mask]
            gy[i_idx[mask]] -= rep * dy[mask] / dist[mask]
            gx[j_idx[mask]] += rep * dx[mask] / dist[mask]
            gy[j_idx[mask]] += rep * dy[mask] / dist[mask]
            gr[i_idx[mask]] += 50.0
            gr[j_idx[mask]] += 50.0
        
        gr -= 1.0
        
        return np.concatenate([gx, gy, gr])
    
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.001, 0.5)] * n
    
    result = minimize(objective, params, method='L-BFGS-B', jac=gradient,
                      bounds=bounds, options={'maxiter': 800, 'maxfun': 3000,
                                               'ftol': 1e-10, 'gtol': 1e-8})
    
    if time.time() - start_time > time_limit:
        best_score = _validate_and_score(circles)
        if best_score is None:
            best_score = 0.0
        return circles.copy(), best_score
    
    new_params = result.x
    new_x = new_params[:n]
    new_y = new_params[n:2*n]
    new_r = new_params[2*n:3*n]
    
    new_r = np.maximum(new_r, 0.001)
    new_x = np.clip(new_x, 0, 1)
    new_y = np.clip(new_y, 0, 1)
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


def _simulated_annealing(circles, max_iter=50000, t_init=0.05, t_min=0.001,
                          time_limit=80, seed=0):
    """Simulated annealing with fast scoring to optimize circle packing."""
    rng = np.random.RandomState(seed)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def _penalty_score(cx, cy, cr, penalty_w=50.0):
        if np.any(cr <= 0):
            return -1e6
        bv = np.maximum(0, cr - cx) + np.maximum(0, cr - (1 - cx)) + \
              np.maximum(0, cr - cy) + np.maximum(0, cr - (1 - cy))
        dx = cx[i_idx] - cx[j_idx]
        dy = cy[i_idx] - cy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        ov = np.maximum(0, cr[i_idx] + cr[j_idx] - dist)
        return float(np.sum(cr) - penalty_w * (np.sum(bv) + np.sum(ov)))
    
    def _strict_score(cx, cy, cr):
        if np.any(cr <= 0):
            return None
        if np.any(cx - cr < -1e-9) or np.any(cx + cr > 1 + 1e-9) or np.any(cy - cr < -1e-9) or np.any(cy + cr > 1 + 1e-9):
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
    
    current_penalty_score = _penalty_score(x, y, r)
    
    t = t_init
    cooling = (t_min / t_init) ** (1.0 / max_iter)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        i = rng.randint(n)
        move_type = rng.rand()
        
        new_x = x.copy()
        new_y = y.copy()
        new_r = r.copy()
        
        if move_type < 0.45:
            delta = 0.012 * rng.randn(2)
            new_x[i] += delta[0]
            new_y[i] += delta[1]
        elif move_type < 0.75:
            new_r[i] *= (1 + 0.06 * rng.randn())
            new_r[i] = max(0.005, min(new_r[i], 0.5))
        else:
            j = rng.randint(n)
            if j == i:
                j = (i + 1) % n
            delta_r = 0.008 * rng.randn()
            new_r[i] += delta_r
            new_r[j] -= delta_r
            new_r[i] = max(0.005, new_r[i])
            new_r[j] = max(0.005, new_r[j])
        
        new_penalty = _penalty_score(new_x, new_y, new_r)
        
        delta = new_penalty - current_penalty_score
        if delta > 0 or rng.rand() < np.exp(delta / max(t, 1e-10)):
            x, y, r = new_x, new_y, new_r
            current_penalty_score = new_penalty
            
            if it % 50 == 0:
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


def construct():
    """Construct optimal circle packing in unit square for N=26 circles."""
    start_time = time.time()
    time_budget = 130  # seconds (evaluator kills at 150s, keep 20s margin)
    safety_margin = 20  # seconds
    
    best_circles = None
    best_score = 0.0
    
    initializations = [
        ('grid', _grid_initial(N)),
        ('random1', _random_initial(N, seed=1)),
        ('random2', _random_initial(N, seed=2)),
        ('structured1', _structured_initial(N, seed=1)),
        ('structured2', _structured_initial(N, seed=2)),
        ('hex_dense1', _hex_dense_initial(N, seed=1)),
        ('hex_dense2', _hex_dense_initial(N, seed=2)),
    ]
    
    for name, init_circles in initializations:
        elapsed = time.time() - start_time
        if elapsed > time_budget - safety_margin:
            break
        
        resolved = _resolve_overlaps(init_circles, max_iter=200)
        
        remaining = time_budget - (time.time() - start_time) - safety_margin
        if remaining > 8:
            loc_circles, loc_score = _force_optimize(resolved, time_limit=min(remaining * 0.25, 12), seed=hash(name) % 1000)
            if loc_score > best_score:
                best_score = loc_score
                best_circles = loc_circles
        
        remaining = time_budget - (time.time() - start_time) - safety_margin
        if remaining > 8:
            lbfgs_circles, lbfgs_score = _lbfgs_optimize(best_circles if best_circles is not None else resolved, 
                                                          time_limit=min(remaining * 0.25, 12), seed=hash(name) % 1000)
            if lbfgs_score > best_score:
                best_score = lbfgs_score
                best_circles = lbfgs_circles
        
        remaining = time_budget - (time.time() - start_time) - safety_margin
        if remaining > 10:
            sa_circles, sa_score = _simulated_annealing(best_circles if best_circles is not None else resolved,
                                                         max_iter=40000, t_init=0.04, t_min=0.0005,
                                                         time_limit=min(remaining * 0.35, 18), seed=hash(name) % 1000)
            if sa_score > best_score:
                best_score = sa_score
                best_circles = sa_circles
    
    if best_circles is not None:
        remaining = time_budget - (time.time() - start_time) - safety_margin
        if remaining > 8:
            del_circles, del_score = _delaunay_polish(best_circles, time_limit=min(remaining * 0.7, 20), seed=999)
            if del_score > best_score:
                best_score = del_score
                best_circles = del_circles
        
        remaining = time_budget - (time.time() - start_time) - safety_margin
        if remaining > 5:
            final_circles, final_score = _force_optimize(best_circles, time_limit=min(remaining * 0.5, 10), seed=1000)
            if final_score > best_score:
                best_score = final_score
                best_circles = final_circles
        
        remaining = time_budget - (time.time() - start_time) - safety_margin
        if remaining > 5:
            del2_circles, del2_score = _delaunay_polish(best_circles, time_limit=min(remaining * 0.8, 12), seed=1001)
            if del2_score > best_score:
                best_score = del2_score
                best_circles = del2_circles
    
    if best_circles is None:
        return _grid_initial(N)
    
    final_score = _validate_and_score(best_circles)
    if final_score is None:
        return _grid_initial(N)
    
    return best_circles

# EVOLVE-BLOCK-END
