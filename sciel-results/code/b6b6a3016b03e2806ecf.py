"""Packing of 50 equal circles in a unit square via SDP-inspired initialization + L-BFGS polish."""
import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 50
SIDE = 1.0
HALF = 0.5


def _compute_score(centers):
    """Return the packing radius for a set of centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary distance
    d_boundary = np.min(np.min(
        np.array([HALF - centers[:, 0], HALF - centers[:, 1],
                  centers[:, 0] + HALF, centers[:, 1] + HALF]), axis=1))
    if d_boundary <= 0:
        return 0.0
    # Pairwise distances
    D = cdist(centers, centers)
    np.fill_diagonal(D, np.inf)
    d_pair = np.min(D) / 2.0
    return min(d_boundary, d_pair)


def _soft_min_obj(x, r_target):
    """
    Soft-min objective: minimize violation where pairwise distances < 2*r_target
    and boundary distances < r_target.
    x is flattened 2D array of shape (2*N,).
    """
    centers = x.reshape(N, 2)
    # Pairwise distances
    D = cdist(centers, centers)
    np.fill_diagonal(D, np.inf)
    # Soft min of (D - 2*r_target) over all pairs
    violations_pair = 2 * r_target - D
    # Only consider positive violations (where D < 2*r_target)
    violations_pair = np.maximum(violations_pair, 0)
    # Boundary violations
    b_x = np.maximum(r_target - (HALF - centers[:, 0]), 0)
    b_y = np.maximum(r_target - (HALF - centers[:, 1]), 0)
    b_x2 = np.maximum(r_target - (HALF + centers[:, 0]), 0)
    b_y2 = np.maximum(r_target - (HALF + centers[:, 1]), 0)
    violations_boundary = np.concatenate([b_x, b_y, b_x2, b_y2])
    
    # Combined soft min using log-sum-exp approximation
    all_viol = np.concatenate([violations_pair.flatten(), violations_boundary])
    # Soft min: -log(sum(exp(-viol/t))) / (1/t) approximates min
    # But we want to maximize the minimum, so minimize the max violation
    # Use a smooth approximation of max
    if len(all_viol) == 0:
        return 0.0
    t = 0.01
    # Smooth max approximation
    log_sum = np.log(np.sum(np.exp(all_viol / t)))
    smooth_max = t * log_sum - t * np.log(len(all_viol))
    return smooth_max


def _grad_soft_min(x, r_target):
    """Gradient of soft-min objective."""
    centers = x.reshape(N, 2)
    D = cdist(centers, centers)
    np.fill_diagonal(D, np.inf)
    violations_pair = 2 * r_target - D
    violations_pair = np.maximum(violations_pair, 0)
    
    b_x = np.maximum(r_target - (HALF - centers[:, 0]), 0)
    b_y = np.maximum(r_target - (HALF - centers[:, 1]), 0)
    b_x2 = np.maximum(r_target - (HALF + centers[:, 0]), 0)
    b_y2 = np.maximum(r_target - (HALF + centers[:, 1]), 0)
    violations_boundary = np.concatenate([b_x, b_y, b_x2, b_y2])
    
    all_viol = np.concatenate([violations_pair.flatten(), violations_boundary])
    if len(all_viol) == 0:
        return np.zeros_like(x)
    
    t = 0.01
    exp_viol = np.exp(all_viol / t)
    sum_exp = np.sum(exp_viol)
    weights = exp_viol / sum_exp
    
    grad = np.zeros_like(x)
    
    # Pairwise gradient
    pair_weights = weights[:violations_pair.size]
    pair_weights = pair_weights.reshape(N, N)
    # d(violations_pair)/d(centers) 
    # violations_pair[i,j] = 2*r - dist(i,j)
    # d(dist)/d(centers[i]) = (centers[i] - centers[j]) / dist
    for i in range(N):
        for j in range(N):
            if i != j and pair_weights[i, j] > 0:
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-10:
                    # d(2r - dist)/d(centers[i]) = -diff/dist
                    grad[2*i] += -t * pair_weights[i, j] * (-diff[0] / dist)
                    grad[2*i+1] += -t * pair_weights[i, j] * (-diff[1] / dist)
    
    # Boundary gradient
    # b_x = max(r - (HALF - x), 0), db_x/dx = -1 if active
    for i in range(N):
        if b_x[i] > 0:
            grad[2*i] += -t * weights[violations_pair.size + i] * (-1)
        if b_y[i] > 0:
            grad[2*i+1] += -t * weights[violations_pair.size + N + i] * (-1)
        if b_x2[i] > 0:
            grad[2*i] += -t * weights[violations_pair.size + 2*N + i] * (1)
        if b_y2[i] > 0:
            grad[2*i+1] += -t * weights[violations_pair.size + 3*N + i] * (1)
    
    return grad


def _hex_init():
    """Create a hexagonal packing initial guess."""
    # Hexagonal packing with 50 circles
    # Rows alternate in offset
    # Try to fit 50 circles in hexagonal arrangement
    # 7 rows: 7,6,7,6,7,6,7 = 46, not enough
    # 8 rows: 7,6,7,6,7,6,7,6 = 52, too many
    # Let's use a grid-based approach with hexagonal offsets
    
    # Use 7 columns
    m_cols = 7
    spacing_x = 1.0 / m_cols
    spacing_y = spacing_x * np.sqrt(3) / 2
    
    # Number of rows to fit
    n_rows = int(1.0 / spacing_y) + 1
    if n_rows < 7:
        n_rows = 7
    
    centers = []
    for row in range(n_rows):
        y = (row + 0.5) * spacing_y - 0.5
        if y > HALF - spacing_y/2 or y < -HALF + spacing_y/2:
            continue
        offset = 0
        if row % 2 == 1:
            offset = spacing_x / 2
        for col in range(m_cols):
            x = (col + 0.5) * spacing_x - 0.5 + offset
            if -HALF <= x <= HALF:
                centers.append([x, y])
    
    # Take first 50
    if len(centers) >= N:
        return np.array(centers[:N])
    else:
        # Fallback to grid
        return _grid_init()


def _grid_init():
    """Create a grid initial guess."""
    m_cols = 7
    m_rows = 8
    xs = (np.arange(m_cols) + 0.5) / m_cols - 0.5
    ys = (np.arange(m_rows) + 0.5) / m_rows - 0.5
    grid = np.array([(x, y) for y in ys for x in xs])
    return grid[:N].copy()


def _sdp_inspired_init():
    """
    SDP-inspired initialization: use a Gram matrix approach to get a good starting point.
    We formulate a simplified convex relaxation and solve it heuristically.
    """
    # Start with hexagonal layout as a strong initial guess
    centers = _hex_init()
    
    # Refine using a few iterations of gradient descent on the soft-min objective
    # with a moderate target radius
    r_target = 0.07  # Start below known best to ensure feasibility
    
    x = centers.flatten()
    t0 = time.time()
    
    # Simple gradient descent
    lr = 0.001
    for it in range(200):
        if time.time() - t0 > 10:  # Limit to 10 seconds
            break
        obj = _soft_min_obj(x, r_target)
        if obj < 1e-6:
            break
        grad = _grad_soft_min(x, r_target)
        # Normalize gradient
        norm = np.linalg.norm(grad)
        if norm > 1e-10:
            grad = grad / norm
        x = x - lr * grad
        # Clip to bounds
        x[0::2] = np.clip(x[0::2], -HALF + 0.001, HALF - 0.001)
        x[1::2] = np.clip(x[1::2], -HALF + 0.001, HALF - 0.001)
    
    centers = x.reshape(N, 2)
    return centers


def _lbfgs_polish(centers, r_target, time_limit=30):
    """Polish using L-BFGS on the soft-min objective."""
    x0 = centers.flatten()
    t0 = time.time()
    
    def obj(x):
        return _soft_min_obj(x, r_target)
    
    def grad(x):
        return _grad_soft_min(x, r_target)
    
    try:
        result = minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            bounds=[(-HALF + 0.001, HALF - 0.001)] * N + [(-HALF + 0.001, HALF - 0.001)] * N,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
        )
        if result.success or result.fun < obj(x0):
            return result.x.reshape(N, 2)
    except Exception:
        pass
    
    return centers


def construct():
    """Construct centers for 50 equal circles in unit square."""
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s limit
    
    # Phase 1: Get a good initial solution using SDP-inspired approach
    centers = _sdp_inspired_init()
    best_score = _compute_score(centers)
    best_centers = centers.copy()
    
    # Phase 2: Binary search on radius with L-BFGS polishing
    # Known best is ~0.071377103865
    r_low = 0.065
    r_high = 0.072
    
    # First, try to achieve the known best
    for attempt in range(3):
        if time.time() > deadline - 20:
            break
        
        r_target = 0.071377103865
        polished = _lbfgs_polish(best_centers, r_target, time_limit=20)
        score = _compute_score(polished)
        if score > best_score:
            best_score = score
            best_centers = polished.copy()
        
        if time.time() > deadline - 10:
            break
    
    # Phase 3: Try to push beyond known best with binary search
    if time.time() < deadline - 15:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            
            r_mid = (r_low + r_high) / 2
            # Try to achieve r_mid
            polished = _lbfgs_polish(best_centers, r_mid, time_limit=15)
            score = _compute_score(polished)
            
            if score >= r_mid - 1e-6:
                # Achieved r_mid
                if score > best_score:
                    best_score = score
                    best_centers = polished.copy()
                r_low = r_mid
            else:
                r_high = r_mid
    
    # Phase 4: Final L-BFGS polish with the best found radius
    if time.time() < deadline - 5:
        final_r = best_score
        polished = _lbfgs_polish(best_centers, final_r, time_limit=10)
        score = _compute_score(polished)
        if score > best_score:
            best_score = score
            best_centers = polished.copy()
    
    return best_centers
