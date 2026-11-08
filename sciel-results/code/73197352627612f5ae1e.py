import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3.0)
H = SQRT3 / 2.0  # height of the equilateral triangle with side 1

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute triangle indices for vectorized area computation
TRI_IDX = np.array(list(combinations(range(N), 3)))
NUM_TRI = TRI_IDX.shape[0]

# Precompute which triangles each point participates in
POINT_TRI = [[] for _ in range(N)]
for idx, (i, j, k) in enumerate(TRI_IDX):
    POINT_TRI[i].append(idx)
    POINT_TRI[j].append(idx)
    POINT_TRI[k].append(idx)
POINT_TRI = [np.array(lst) for lst in POINT_TRI]

def _all_areas(points):
    """Compute all triangle areas vectorized."""
    pi = points[TRI_IDX[:, 0]]
    pj = points[TRI_IDX[:, 1]]
    pk = points[TRI_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def min_triangle_area(points):
    """Compute the minimum area over all C(n,3) triangles using vectorization."""
    return float(np.min(_all_areas(points)))

def is_inside_triangle(p, eps=1e-9):
    """Check if point p is inside the equilateral triangle."""
    x, y = p
    if y < -eps or x < -eps or x > 1.0 + eps:
        return False
    if y > SQRT3 * x + eps:
        return False
    if y > SQRT3 * (1.0 - x) + eps:
        return False
    return True

def project_to_triangle(p):
    """Project a point onto the triangle if it's outside."""
    x, y = float(p[0]), float(p[1])
    if y < 0:
        y = 0.0
    if y > H:
        y = H
    x_min = y / SQRT3
    x_max = 1.0 - y / SQRT3
    if x < x_min:
        x = x_min
    if x > x_max:
        x = x_max
    return np.array([x, y])

def project_points_to_triangle(pts):
    """Project all points to triangle."""
    return np.array([project_to_triangle(p) for p in pts])

def _soft_min_surrogate(points, alpha=100.0):
    """
    Smooth approximation of -min(areas) using log-sum-exp.
    We maximize this (minimize its negative).
    """
    areas = _all_areas(points)
    # Soft min: -log(sum(exp(-alpha * areas))) / alpha
    # This approximates -min(areas) as alpha -> inf
    shifted = -alpha * areas
    shifted -= np.max(shifted)  # for numerical stability
    log_sum = np.log(np.sum(np.exp(shifted)))
    return -(log_sum + np.max(shifted)) / alpha  # This is -soft_min(areas)

def _soft_min_grad(points, alpha=100.0):
    """
    Gradient of the soft-min surrogate with respect to points.
    Returns gradient of -soft_min(areas) (so we can minimize it).
    """
    areas = _all_areas(points)
    pi = points[TRI_IDX[:, 0]]
    pj = points[TRI_IDX[:, 1]]
    pk = points[TRI_IDX[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    # Weights for soft min
    shifted = -alpha * areas
    shifted -= np.max(shifted)
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    # Gradient of area_i w.r.t. each point
    # area_i = 0.5 * |cross_i|
    # d(area_i)/d(point) depends on which point and the sign of cross
    
    grad = np.zeros_like(points)
    sign = np.sign(cross)
    # If cross > 0: area = 0.5 * cross, d(area)/d(points)
    # If cross < 0: area = -0.5 * cross, d(area)/d(points) = -0.5 * d(cross)/d(points)
    
    for t in range(NUM_TRI):
        w = weights[t]
        if w < 1e-10:
            continue
        i, j, k = TRI_IDX[t]
        s = sign[t]
        # d(cross)/d(p_i) = -(v1_y, -v1_x) ... let me derive properly
        # cross = (pj-pi)_x * (pk-pi)_y - (pj-pi)_y * (pk-pi)_x
        # Let A=pj-pi, B=pk-pi
        # cross = A_x*B_y - A_y*B_x
        # d(cross)/d(pi) = -(B_y, -B_x) ... actually:
        # d(A_x)/d(pi_x) = -1, d(A_y)/d(pi_y) = -1
        # d(B_x)/d(pi_x) = -1, d(B_y)/d(pi_y) = -1
        # d(cross)/d(pi_x) = (-1)*B_y + A_y*1 = -B_y + A_y
        # d(cross)/d(pi_y) = A_x*1 - (-1)*B_x = A_x + B_x
        # So d(cross)/d(pi) = (A_y - B_y, A_x + B_x)
        
        # d(cross)/d(pj): A = pj - pi, so dA/d(pj) = (1,0) or (0,1)
        # d(cross)/d(pj_x) = 1*B_y - 0 = B_y
        # d(cross)/d(pj_y) = 0 - 1*B_x = -B_x
        # d(cross)/d(pj) = (B_y, -B_x)
        
        # d(cross)/d(pk): B = pk - pi
        # d(cross)/d(pk_x) = 0 - A_y*1 = -A_y
        # d(cross)/d(pk_y) = A_x*1 - 0 = A_x
        # d(cross)/d(pk) = (-A_y, A_x)
        
        if s > 0:
            # area = 0.5 * cross, so d(area)/d = 0.5 * d(cross)/d
            darea_di = 0.5 * np.array([v1[1] - v2[1], v1[0] + v2[0]])  # A_y-B_y, A_x+B_x
            darea_dj = 0.5 * np.array([v2[1], -v2[0]])  # B_y, -B_x
            darea_dk = 0.5 * np.array([-v1[1], v1[0]])  # -A_y, A_x
        else:
            # area = -0.5 * cross, so d(area)/d = -0.5 * d(cross)/d
            darea_di = -0.5 * np.array([v1[1] - v2[1], v1[0] + v2[0]])
            darea_dj = -0.5 * np.array([v2[1], -v2[0]])
            darea_dk = -0.5 * np.array([-v1[1], v1[0]])
        
        # We are computing gradient of -soft_min, which is -sum(w_i * area_i)
        # So d(-soft_min)/d(point) = -sum(w_i * d(area_i)/d(point))
        grad[i] += -w * darea_di
        grad[j] += -w * darea_dj
        grad[k] += -w * darea_dk
    
    return grad

def generate_structured_initial():
    """Generate a structured initial guess with good symmetry."""
    points = []
    # Vertices
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, H])
    
    # Edge midpoints
    points.append([0.5, 0.0])
    points.append([0.25, H/2])
    points.append([0.75, H/2])
    
    # Center
    points.append([0.5, H/3])
    
    # Additional points - better spread
    points.append([0.2, 0.05])
    points.append([0.8, 0.05])
    points.append([0.3, H*0.35])
    points.append([0.7, H*0.35])
    
    return np.array(points[:N])

def generate_random_points(rng):
    """Generate random points inside the triangle."""
    points = []
    for _ in range(N):
        u = rng.random()
        v = rng.random()
        if u + v > 1.0:
            u = 1.0 - u
            v = 1.0 - v
        p = (1.0 - u - v) * V0 + u * V1 + v * V2
        points.append(p)
    return np.array(points)

def _flatten(points):
    """Flatten 2D points array to 1D for scipy optimizer."""
    return points.flatten()

def _unflatten(flat):
    """Unflatten 1D array to 2D points."""
    pts = flat.reshape(N, 2)
    return project_points_to_triangle(pts)

def lbfgs_optimize(points, alpha=50.0, maxiter=500, deadline=None):
    """
    Use L-BFGS-B to minimize the soft-min surrogate.
    """
    start = time.time()
    
    def objective(flat):
        pts = _unflatten(flat)
        return _soft_min_surrogate(pts, alpha)
    
    def gradient(flat):
        pts = _unflatten(flat)
        grad_2d = _soft_min_grad(pts, alpha)
        return grad_2d.flatten()
    
    x0 = _flatten(points)
    
    try:
        result = minimize(
            objective, x0, method='L-BFGS-B', jac=gradient,
            options={'maxiter': maxiter, 'gtol': 1e-6, 'maxls': 50}
        )
    except Exception:
        return points, min_triangle_area(points)
    
    best_pts = _unflatten(result.x)
    best_val = min_triangle_area(best_pts)
    
    return best_pts, best_val

def local_optimize(points, rng, max_iter=200, step_size=0.03, deadline=None):
    """Local optimization using coordinate descent with adaptive steps."""
    points = points.copy()
    best_val = min_triangle_area(points)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        improved = False
        for i in range(N):
            best_local = points[i].copy()
            best_local_val = best_val
            
            # Try multiple random perturbations with decreasing step
            for attempt in range(20):
                if deadline is not None and time.time() > deadline:
                    break
                    
                curr_step = step_size * (0.5 ** (attempt // 5))
                dx = rng.uniform(-curr_step, curr_step, 2)
                new_p = best_local + dx
                new_p = project_to_triangle(new_p)
                
                if not is_inside_triangle(new_p, eps=1e-6):
                    continue
                    
                new_points = points.copy()
                new_points[i] = new_p
                
                val = min_triangle_area(new_points)
                if val > best_local_val + 1e-10:
                    best_local_val = val
                    best_local = new_p
                    improved = True
            
            points[i] = best_local
        
        current_val = min_triangle_area(points)
        if current_val > best_val:
            best_val = current_val
        
        if not improved:
            step_size *= 0.8
            if step_size < 1e-7:
                break
    
    return points, best_val

def construct():
    """Construct 11 points in the equilateral triangle to maximize the minimum triangle area."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.RandomState(42)
    
    best_points = None
    best_value = -1
    
    # Generate diverse initializations
    inits = []
    inits.append(generate_structured_initial())
    
    for seed in range(15):
        inits.append(generate_random_points(np.random.RandomState(seed * 123 + 7)))
    
    # Also try perturbed structured
    for seed in range(5):
        base = generate_structured_initial().copy()
        noise = np.random.RandomState(seed * 99 + 13).uniform(-0.05, 0.05, base.shape)
        perturbed = project_points_to_triangle(base + noise)
        inits.append(perturbed)
    
    total_inits = len(inits)
    time_per_init = (deadline - start_time - 15) / total_inits  # reserve 15s for final polish
    
    for idx, init_points in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = deadline - start_time - 15  # reserve for final
        if remaining < 5:
            break
        
        alloc = min(time_per_init, remaining - 5)
        init_deadline = start_time + elapsed + alloc
        
        # Step 1: L-BFGS with soft-min
        if time.time() < init_deadline - 2:
            lbfgs_pts, lbfgs_val = lbfgs_optimize(init_points, alpha=30.0, maxiter=300, deadline=init_deadline - 2)
            if lbfgs_val > best_value:
                best_value = lbfgs_val
                best_points = lbfgs_pts.copy()
            current_best = lbfgs_pts
        else:
            current_best = init_points.copy()
        
        # Step 2: Local hill climbing from L-BFGS result
        if time.time() < init_deadline:
            hc_pts, hc_val = local_optimize(current_best, rng, max_iter=100, step_size=0.02, deadline=init_deadline)
            if hc_val > best_value:
                best_value = hc_val
                best_points = hc_pts.copy()
    
    # Final polish with finer L-BFGS
    if best_points is not None and time.time() < deadline - 5:
        final_pts, final_val = lbfgs_optimize(best_points, alpha=80.0, maxiter=200, deadline=deadline - 2)
        if final_val > best_value:
            best_value = final_val
            best_points = final_pts.copy()
    
    # Verify all points are inside the triangle
    if best_points is not None:
        for i, p in enumerate(best_points):
            if not is_inside_triangle(p, eps=1e-6):
                best_points[i] = project_to_triangle(p)
    
    if best_points is None:
        best_points = generate_structured_initial()
    
    return best_points
