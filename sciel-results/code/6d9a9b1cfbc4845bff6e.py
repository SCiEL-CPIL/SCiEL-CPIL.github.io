"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])
TRI_AREA = H / 2.0  # area of unit equilateral triangle

# Precompute all C(11,3) = 165 triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)
_n_tris = len(_idx_i)

# Precompute point masks: which triangles involve each point
_point_masks = []
for p in range(N):
    mask = ((_idx_i == p) | (_idx_j == p) | (_idx_k == p))
    _point_masks.append(mask)


def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return 0.5 * np.min(cross)


def _min_area_and_idx(points):
    """Return (min_area, index_of_worst_triangle)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    areas = 0.5 * cross
    idx = int(np.argmin(areas))
    return areas[idx], idx


def _is_inside(points, eps=1e-10):
    """Check if all points are inside the equilateral triangle."""
    x = points[:, 0]
    y = points[:, 1]
    return np.all(y >= -eps) and np.all(y <= H + eps) and \
           np.all(x >= y / (2 * H) - eps) and np.all(x <= 1 - y / (2 * H) + eps)


def _soft_min(areas, beta=50.0):
    """Soft minimum using log-sum-exp trick: -log(sum(exp(-beta*a)))/beta approximates min(a)."""
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    return -log_sum / beta


def _soft_min_grad(areas, beta=50.0):
    """Soft minimum and its gradient with respect to areas."""
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    shifted = neg_beta_areas - max_val
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    soft_min_val = -(max_val + np.log(sum_exp)) / beta
    # d(soft_min)/d(areas) = -(-beta * exp_shifted / sum_exp) / beta = exp_shifted / sum_exp
    grad = exp_shifted / sum_exp
    return soft_min_val, grad


def _boundary_penalty(points):
    """Compute smooth boundary penalty. Returns 0 if all points are well inside."""
    x = points[:, 0]
    y = points[:, 1]
    margin = 0.001
    # y >= 0
    viol1 = np.maximum(0.0, -y + margin)
    # y <= H
    viol2 = np.maximum(0.0, y - H + margin)
    # x >= y/(2H)
    viol3 = np.maximum(0.0, y / (2 * H) - x + margin)
    # x <= 1 - y/(2H)
    viol4 = np.maximum(0.0, x - (1 - y / (2 * H)) + margin)
    return np.sum(viol1**2) + np.sum(viol2**2) + np.sum(viol3**2) + np.sum(viol4**2)


def _boundary_grad(points):
    """Gradient of boundary penalty."""
    x = points[:, 0]
    y = points[:, 1]
    margin = 0.001
    grad = np.zeros_like(points)
    
    # y >= 0: penalty = max(0, -y+margin)^2, d/dy = -2*max(0,-y+margin)
    v1 = np.maximum(0.0, -y + margin)
    grad[:, 1] += -2 * v1
    
    # y <= H: penalty = max(0, y-H+margin)^2, d/dy = 2*max(0,y-H+margin)
    v2 = np.maximum(0.0, y - H + margin)
    grad[:, 1] += 2 * v2
    
    # x >= y/(2H): penalty = max(0, y/(2H)-x+margin)^2
    v3 = np.maximum(0.0, y / (2 * H) - x + margin)
    grad[:, 0] += -2 * v3
    grad[:, 1] += 2 * v3 / (2 * H)
    
    # x <= 1 - y/(2H): penalty = max(0, x-(1-y/(2H))+margin)^2
    v4 = np.maximum(0.0, x - (1 - y / (2 * H)) + margin)
    grad[:, 0] += 2 * v4
    grad[:, 1] += -2 * v4 / (2 * H)
    
    return grad


def _lfbfgs_objective(params, beta=50.0):
    """Objective for L-BFGS-B: negative soft-min area + boundary penalty."""
    points = params.reshape(N, 2)
    pen = _boundary_penalty(points)
    if pen > 1e-6:
        return -_min_area(points) + 1000.0 * pen
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    areas = 0.5 * cross
    return -_soft_min(areas, beta)


def _lfbfgs_grad(params, beta=50.0):
    """Gradient for L-BFGS-B."""
    points = params.reshape(N, 2)
    pen = _boundary_penalty(points)
    if pen > 1e-6:
        # Just use finite difference approximation for penalty region
        return 1000.0 * _boundary_grad(points)
    
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    dx1 = p2[:, 0] - p1[:, 0]
    dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]
    dy2 = p3[:, 1] - p1[:, 1]
    cross_raw = dx1 * dy2 - dy1 * dx2
    cross_abs = np.abs(cross_raw)
    areas = 0.5 * cross_abs
    
    sm_val, sm_grad = _soft_min_grad(areas, beta)
    
    # d(soft_min)/d(points) = sum_t sm_grad[t] * d(area_t)/d(points)
    # area_t = 0.5 * |cross_t|
    # d(area_t)/d(point_v) depends on whether point v is p1, p2, or p3 in triangle t
    
    grad = np.zeros((N, 2))
    
    # For each triangle t, cross = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
    # sign = sign(cross_raw)
    sign = np.sign(cross_raw)
    # d(area)/d(point) = 0.5 * sign * d(cross)/d(point)
    
    # d(cross)/d(x1) = -(y3-y1) = -dy2
    # d(cross)/d(y1) = -(x3-x1) = -dx2
    # d(cross)/d(x2) = +(y3-y1) = +dy2
    # d(cross)/d(y2) = +(x3-x1) = +dx2
    # d(cross)/d(x3) = -(y2-y1) = -dy1
    # d(cross)/d(y3) = +(x2-x1) = +dx1
    
    # Contribution to gradient from each triangle:
    # sm_grad[t] * 0.5 * sign[t] * d(cross_t)/d(point)
    
    # Accumulate for each point
    # For point i (as p1): grad[i] += sm_grad[t] * 0.5 * sign[t] * [-dy2, -dx2]
    # For point j (as p2): grad[j] += sm_grad[t] * 0.5 * sign[t] * [dy2, dx2]
    # For point k (as p3): grad[k] += sm_grad[t] * 0.5 * sign[t] * [-dy1, dx1]
    
    coeff = 0.5 * sm_grad * sign
    
    # Accumulate contributions
    # Using numpy advanced indexing
    np.add.at(grad, (_idx_i, 0), -coeff * dy2)
    np.add.at(grad, (_idx_i, 1), -coeff * dx2)
    np.add.at(grad, (_idx_j, 0), coeff * dy2)
    np.add.at(grad, (_idx_j, 1), coeff * dx2)
    np.add.at(grad, (_idx_k, 0), -coeff * dy1)
    np.add.at(grad, (_idx_k, 1), coeff * dx1)
    
    # Objective is -soft_min, so negate gradient
    return -grad.ravel()


def _lfbfgs_refine(points, deadline, beta=50.0, max_iter=200):
    """Refine using L-BFGS-B with soft-min objective."""
    start = time.time()
    params = points.ravel().copy()
    
    # Bounds: keep points inside triangle with small margin
    # x in [0, 1], y in [0, H] as simple bounds
    bounds = [(0.0, 1.0), (0.0, H)] * N
    
    try:
        result = minimize(
            _lfbfgs_objective, params,
            jac=_lfbfgs_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10,
                     'maxls': 50, 'maxfun': 1000}
        )
    except Exception:
        return points, _min_area(points)
    
    new_points = result.x.reshape(N, 2)
    
    # Project back inside triangle
    if not _is_inside(new_points, eps=1e-8):
        # Simple projection
        for i in range(N):
            x, y = new_points[i]
            if y < 0:
                y = 0.01
            if y > H:
                y = H - 0.01
            x_min = y / (2 * H) + 0.005
            x_max = 1 - y / (2 * H) - 0.005
            if x < x_min:
                x = x_min
            if x > x_max:
                x = x_max
            new_points[i] = [x, y]
    
    new_area = _min_area(new_points)
    old_area = _min_area(points)
    
    if new_area > old_area:
        return new_points, new_area
    else:
        return points, old_area


def _local_refine(points, deadline, max_iter=300):
    """Local refinement: move vertices of worst triangle to improve min area."""
    best_points = points.copy()
    best_area = _min_area(best_points)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Find worst triangle
        min_area, worst_idx = _min_area_and_idx(best_points)
        if min_area > best_area:
            best_area = min_area
        
        i, j, k = _idx_i[worst_idx], _idx_j[worst_idx], _idx_k[worst_idx]
        
        # Try moving each of the three vertices in small steps
        improved = False
        for vi in [i, j, k]:
            for dx in [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]:
                for dy in [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]:
                    new_points = best_points.copy()
                    new_points[vi, 0] += dx
                    new_points[vi, 1] += dy
                    if not _is_inside(new_points):
                        continue
                    new_area = _min_area(new_points)
                    if new_area > best_area + 1e-12:
                        best_points = new_points
                        best_area = new_area
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try larger steps
            for vi in [i, j, k]:
                for dx in [-0.05, -0.03, 0.03, 0.05]:
                    for dy in [-0.05, -0.03, 0.03, 0.05]:
                        new_points = best_points.copy()
                        new_points[vi, 0] += dx
                        new_points[vi, 1] += dy
                        if not _is_inside(new_points):
                            continue
                        new_area = _min_area(new_points)
                        if new_area > best_area + 1e-12:
                            best_points = new_points
                            best_area = new_area
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_points, best_area


def _lattice_initial():
    """Generate a good initial configuration using structured placement."""
    # 4-3-3-1 pattern from bottom to top
    row_ys = [0.08 * H, 0.30 * H, 0.55 * H, 0.78 * H]
    row_ns = [4, 3, 3, 1]
    
    points = np.zeros((N, 2))
    pi = 0
    for y_frac, n_pts in zip(row_ys, row_ns):
        y = y_frac
        x_left = 0.5 * y / H
        x_right = 1.0 - 0.5 * y / H
        width = x_right - x_left
        
        if n_pts == 1:
            points[pi, 0] = 0.5
            points[pi, 1] = y
            pi += 1
        else:
            margin_frac = 0.15
            usable = width * (1 - 2 * margin_frac)
            start = x_left + width * margin_frac
            xs = start + np.linspace(0, usable, n_pts)
            for x in xs:
                points[pi, 0] = x
                points[pi, 1] = y
                pi += 1
    
    return points


def _random_initial(rng):
    """Generate random points inside the triangle."""
    points = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1:
                points[i, 0] = u + 0.5 * v
                points[i, 1] = H * v
                break
    return points


def _symmetric_initial():
    """Generate a symmetric initial configuration."""
    # Try several symmetric patterns
    patterns = []
    
    # Pattern 1: 4-3-3-1
    p1 = _lattice_initial()
    patterns.append(p1)
    
    # Pattern 2: centered hexagonal-like
    p2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, H],
        [0.5, 0.0],
        [0.25, H/2],
        [0.75, H/2],
        [0.5, H/3],
        [1/3, 0.0],
        [2/3, 0.0],
        [0.35, H/4],
        [0.65, H/4],
    ])
    if _is_inside(p2):
        patterns.append(p2)
    
    # Pattern 3: more spread out
    p3 = np.array([
        [0.05, 0.05],
        [0.95, 0.05],
        [0.5, H - 0.05],
        [0.5, 0.05],
        [0.25, H/2],
        [0.75, H/2],
        [0.5, H/3],
        [0.2, 0.05],
        [0.8, 0.05],
        [0.35, H/4],
        [0.65, H/4],
    ])
    if _is_inside(p3):
        patterns.append(p3)
    
    return patterns


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Collect initial configurations
    inits = _symmetric_initial()
    rng = np.random.default_rng(42)
    for _ in range(8):
        inits.append(_random_initial(rng))
    
    best_points = None
    best_area = 0.0
    
    for init in inits:
        if time.time() > deadline - 20:
            break
        
        if not _is_inside(init):
            continue
        
        # Phase 1: L-BFGS-B refinement with soft-min
        refined, area = _lfbfgs_refine(init, deadline - 15, beta=50.0, max_iter=100)
        
        # Phase 2: Local refinement with discrete moves
        refined, area = _local_refine(refined, deadline - 10, max_iter=200)
        
        # Phase 3: Another L-BFGS-B pass with higher beta
        refined, area = _lfbfgs_refine(refined, deadline - 5, beta=100.0, max_iter=50)
        
        # Phase 4: Final local polish
        refined, area = _local_refine(refined, deadline - 2, max_iter=100)
        
        if area > best_area:
            best_area = area
            best_points = refined.copy()
    
    # Final high-precision polish on the best
    if best_points is not None and time.time() < deadline - 2:
        best_points, best_area = _local_refine(best_points, deadline - 1, max_iter=200)
    
    if best_points is None:
        # Fallback
        best_points = _lattice_initial()
    
    return best_points
