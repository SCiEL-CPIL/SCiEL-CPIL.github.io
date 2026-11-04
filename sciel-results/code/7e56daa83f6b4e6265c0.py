"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    if n != N:
        return False
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _total_radius(circles):
    return float(np.sum(circles[:, 2]))

def _separate_circles(circles, w, h, max_iter=2000):
    """
    Directly resolve overlaps and boundary violations using pairwise repulsion.
    Returns (circles, is_valid).
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    for it in range(max_iter):
        moved = False

        # Resolve boundary violations by shrinking radii if needed
        for i in range(n):
            if r[i] <= 1e-6:
                r[i] = 1e-6
            # Clamp position to keep circle inside rectangle
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])

        # Resolve overlaps via pairwise separation
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        dist = 1e-6
                    overlap = min_dist - dist
                    # Distribute push based on relative sizes
                    total_r = r[i] + r[j]
                    wi = r[i] / total_r if total_r > 0 else 0.5
                    wj = r[j] / total_r if total_r > 0 else 0.5
                    push = overlap + 1e-8
                    x[i] += wi * push * dx / dist
                    y[i] += wi * push * dy / dist
                    x[j] -= wj * push * dx / dist
                    y[j] -= wj * push * dy / dist
                    moved = True

        # Re-clamp after separation, shrinking radius if circle doesn't fit
        for i in range(n):
            max_r_x = min(x[i], w - x[i])
            max_r_y = min(y[i], h - y[i])
            max_r = max(1e-6, min(max_r_x, max_r_y))
            if r[i] > max_r:
                r[i] = max_r
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])

        if not moved:
            break

    result = np.column_stack([x, y, r])
    valid = _validate(result, w, h)
    return result, valid

def _smooth_max(v, eps=1e-4):
    return 0.5 * (v + np.sqrt(v**2 + eps**2))

def _objective(params, w, h, penalty_weight):
    """
    Objective: minimize -sum(r) + penalties for violations.
    Vectorized for speed.
    """
    circles = params.reshape(N, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    obj = -np.sum(r)

    # Boundary penalties (smooth)
    v1 = r - x
    v2 = r - y
    v3 = x + r - w
    v4 = y + r - h
    v5 = -r

    obj += penalty_weight * (np.sum(_smooth_max(v1)**2) + np.sum(_smooth_max(v2)**2) +
                             np.sum(_smooth_max(v3)**2) + np.sum(_smooth_max(v4)**2) +
                             np.sum(_smooth_max(v5)**2))

    # Overlap penalties: vectorized computation
    dx_mat = x[:, np.newaxis] - x[np.newaxis, :]
    dy_mat = y[:, np.newaxis] - y[np.newaxis, :]
    dist_mat = np.sqrt(dx_mat**2 + dy_mat**2)
    np.fill_diagonal(dist_mat, 1e12)

    r_sum_mat = r[:, np.newaxis] + r[np.newaxis, :]
    v_overlap = r_sum_mat - dist_mat

    # Only use upper triangle to avoid double counting
    iu, ju = np.triu_indices(N, k=1)
    v_vec = v_overlap[iu, ju]
    obj += penalty_weight * np.sum(np.maximum(v_vec, 0)**2)

    return obj

def _optimize_lbfgsb(circles, w, h, deadline):
    """
    Optimize using L-BFGS-B with annealed penalties.
    """
    start = time.time()
    params = circles.flatten()
    r_max = min(w, h) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, w - 1e-6))
        bounds.append((1e-6, h - 1e-6))
        bounds.append((1e-6, r_max))

    best_circles = circles.copy()
    best_score = _total_radius(best_circles)

    # Anneal penalty weight
    weights = [50, 200, 1000, 5000, 20000]
    current_params = params.copy()

    for w_pen in weights:
        if time.time() > deadline - 2.0:
            break
        try:
            res = minimize(
                _objective, current_params, args=(w, h, w_pen),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            current_params = res.x
        except Exception:
            continue

        circles_new = current_params.reshape(N, 3)
        # Repair step
        circles_new, valid = _separate_circles(circles_new, w, h, max_iter=1000)
        if valid:
            score = _total_radius(circles_new)
            if score > best_score:
                best_score = score
                best_circles = circles_new.copy()
                current_params = best_circles.flatten()

    return best_circles, best_score

def _make_hexagonal_cluster(w, h):
    """
    Create a hexagonal cluster of 21 circles.
    Structure:
      3
      4
      5
      4
      3
    Total: 3+4+5+4+3 = 19? No.
    Let's use:
      2
      3
      4
      4
      3
      2
      3 ? No.
    
    Standard 21 hex cluster:
      1
      3
      5
      5
      3
      1
      3? No.
      
    Let's just use a grid of 5x5 and remove corners? No, that's 21.
    But hexagonal is better.
    
    Rows:
    1: 3 circles
    2: 4 circles
    3: 5 circles
    4: 4 circles
    5: 3 circles
    Total: 3+4+5+4+3 = 19. Need 2 more.
    
    Let's do:
    1: 2
    2: 4
    3: 5
    4: 4
    5: 3
    6: 3
    Total: 2+4+5+4+3+3 = 21.
    
    Or simpler:
    1: 3
    2: 4
    3: 5
    4: 5
    5: 4
    Total: 21.
    
    Let's use this symmetric 5-row structure.
    """
    rows = [3, 4, 5, 5, 4]
    circles = []
    total_width = 5.0 # 5 units wide
    total_height = 5.0 * np.sqrt(3) / 2 * 4 + 2 # approx height
    
    # Scale to fit w, h
    scale_x = w / total_width
    scale_y = h / total_height
    
    scale = min(scale_x, scale_y)
    
    # Hex spacing
    dx = scale
    dy = scale * np.sqrt(3) / 2
    
    # Center the cluster
    offset_x = (w - total_width * scale) / 2
    offset_y = (h - total_height * scale) / 2
    
    # Base radius slightly smaller than half spacing to allow optimization
    base_r = 0.45 * scale
    
    for i, n_circles in enumerate(rows):
        # Center each row
        row_width = (n_circles - 1) * dx
        start_x = offset_x + (w - total_width * scale) / 2 + (total_width * scale - row_width) / 2
        y = offset_y + i * dy
        
        for j in range(n_circles):
            x = start_x + j * dx
            r = base_r
            circles.append([x, y, r])
            
    return np.array(circles)

def _make_grid(n, w, h):
    """Fallback grid initialization."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            cx = (i + 0.5) * w / side
            cy = (j + 0.5) * h / side
            r = 0.5 * min(w, h) / side * 0.8
            cells.append([cx, cy, r])
    return np.array(cells[:n])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_value = 0.0
    best_result = None

    # Aspect ratios
    # Perimeter 4 => 2(w+h)=4 => w+h=2.
    # w = ar * h => h(ar+1)=2 => h = 2/(ar+1), w = 2ar/(ar+1)
    ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 1.1, 0.9, 1.3, 0.75, 1.5, 0.65]
    
    rng = np.random.default_rng(42)

    for ar in ratios:
        if time.time() > deadline - 10.0:
            break
        
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        # Try both orientations
        for (ww, hh) in [(w, h), (h, w)]:
            if time.time() > deadline - 10.0:
                break
            
            try:
                # 1. Hexagonal Cluster Init
                circles = _make_hexagonal_cluster(ww, hh)
                
                # Ensure valid start
                circles, valid = _separate_circles(circles, ww, hh, max_iter=500)
                if not valid:
                    # Fallback to grid if hex fails
                    circles = _make_grid(N, ww, hh)
                    circles, valid = _separate_circles(circles, ww, hh, max_iter=500)
                    if not valid:
                        continue
                
                # 2. L-BFGS-B Optimization
                if time.time() < deadline - 5.0:
                    circles, val = _optimize_lbfgsb(circles, ww, hh, deadline)
                    
                # 3. Final Polish/Validation
                circles, valid = _separate_circles(circles, ww, hh, max_iter=100)
                if valid:
                    current_value = _total_radius(circles)
                    if current_value > best_value:
                        best_value = current_value
                        best_result = circles.copy()
                        
            except Exception:
                continue

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 1.0 / side
                y = (j + 0.5) * 1.0 / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result
