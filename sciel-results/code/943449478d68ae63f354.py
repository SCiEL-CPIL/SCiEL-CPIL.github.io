# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools
from scipy.optimize import minimize


def _compute_score_fast(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points using fully vectorized triangle areas."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    # Fully vectorized: compute all 286 triangle areas at once
    # Use the cross product formula: area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    # For all triples (i, j, k) with i < j < k
    min_area = float('inf')
    
    # Build all triples
    idx_i, idx_j, idx_k = np.indices((n, n, n))
    mask = (idx_i < idx_j) & (idx_j < idx_k)
    ii = idx_i[mask]
    jj = idx_j[mask]
    kk = idx_k[mask]
    
    pi = points[ii]
    pj = points[jj]
    pk = points[kk]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    min_area = float(np.min(areas))
    
    if min_area <= 0:
        return 0.0
    
    return min_area / A_hull


def _find_worst_triangle(points: np.ndarray):
    """Find the triangle with minimum area and its index triple."""
    n = points.shape[0]
    min_area = float('inf')
    worst_triple = (0, 1, 2)
    
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                pi = points[i]
                pj = points[j]
                pk = points[k]
                v1 = pj - pi
                v2 = pk - pi
                area = abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2.0
                if area < min_area:
                    min_area = area
                    worst_triple = (i, j, k)
    
    return worst_triple, min_area


def _neg_score(points: np.ndarray) -> float:
    """Negative of the score for minimization."""
    s = _compute_score_fast(points)
    return -s


def _local_search(points: np.ndarray, deadline: float, max_iters: int = 20000) -> np.ndarray:
    """Local search targeting the worst triangle with multi-directional moves."""
    best_points = points.copy()
    best_score = _compute_score_fast(best_points)
    
    n = len(best_points)
    step = 0.03
    min_step = 1e-9
    decay = 0.85
    no_improve_count = 0
    
    # 12 directions: 8 compass + 4 diagonal-ish
    directions = []
    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        directions.append(np.array([np.cos(angle), np.sin(angle)]))
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Find the worst triangle
        worst_triple, min_area = _find_worst_triangle(best_points)
        
        # Try to move each vertex of the worst triangle in multiple directions
        for idx in worst_triple:
            for d in directions:
                for sign in [1.0, -1.0]:
                    candidate = best_points.copy()
                    candidate[idx] += sign * step * d
                    # Keep within [0, 1]
                    if np.any(candidate[idx] < -0.01) or np.any(candidate[idx] > 1.01):
                        continue
                    np.clip(candidate, 0, 1, out=candidate)
                    
                    score = _compute_score_fast(candidate)
                    if score > best_score + 1e-12:
                        best_points = candidate
                        best_score = score
                        improved = True
                        no_improve_count = 0
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 5:
                step *= decay
                no_improve_count = 0
                if step < min_step:
                    break
        else:
            # Slightly increase step if we found improvement
            step = min(step * 1.05, 0.1)
    
    return best_points


def _powell_refine(points: np.ndarray, deadline: float) -> np.ndarray:
    """Use Powell's method to refine the point configuration."""
    def objective(x):
        if time.time() > deadline:
            return 0.0
        pts = x.reshape(-1, 2)
        if np.any(pts < -0.01) or np.any(pts > 1.01):
            return 0.0
        pts = np.clip(pts, 0, 1)
        return -_compute_score_fast(pts)
    
    x0 = points.flatten()
    
    try:
        result = minimize(objective, x0, method='Powell', 
                          options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-10,
                                   'disp': False})
        best_pts = result.x.reshape(-1, 2)
        best_pts = np.clip(best_pts, 0, 1)
        
        if _compute_score_fast(best_pts) > _compute_score_fast(points):
            return best_pts
    except Exception:
        pass
    
    return points


def _simulated_annealing(points: np.ndarray, deadline: float,
                          T_init: float = 0.08, T_min: float = 1e-7,
                          cooling_rate: float = 0.996, max_iters: int = 30000) -> np.ndarray:
    """Simulated annealing with adaptive step size."""
    best_points = points.copy()
    current_points = points.copy()
    current_score = _compute_score_fast(current_points)
    best_score = current_score
    
    n = len(current_points)
    step_size = 0.04
    min_step = 1e-9
    accept_count = 0
    total_count = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        T = T_init * (cooling_rate ** it)
        if T < T_min:
            T = T_min
        
        # Choose random point and direction
        idx = np.random.randint(n)
        angle = np.random.random() * 2 * np.pi
        magnitude = np.random.random() * step_size
        dx = magnitude * np.cos(angle)
        dy = magnitude * np.sin(angle)
        
        candidate = current_points.copy()
        candidate[idx, 0] += dx
        candidate[idx, 1] += dy
        
        if candidate[idx, 0] < 0 or candidate[idx, 0] > 1 or \
           candidate[idx, 1] < 0 or candidate[idx, 1] > 1:
            continue
        
        candidate_score = _compute_score_fast(candidate)
        
        # Acceptance
        if candidate_score >= current_score:
            current_points = candidate
            current_score = candidate_score
            accept_count += 1
            if current_score > best_score:
                best_score = current_score
                best_points = candidate.copy()
        else:
            delta = current_score - candidate_score
            if np.random.random() < np.exp(-delta / max(T, 1e-10)):
                current_points = candidate
                current_score = candidate_score
        
        total_count += 1
        
        # Adaptive step size
        if total_count % 200 == 0:
            ratio = accept_count / max(total_count, 1)
            if ratio < 0.1:
                step_size *= 0.8
            elif ratio > 0.7:
                step_size *= 1.2
            step_size = max(min_step, min(step_size, 0.15))
            accept_count = 0
            total_count = 0
    
    return best_points


def _hexagonal_initialization(n: int) -> np.ndarray:
    """Create a hexagonal lattice initialization."""
    points = []
    spacing = 0.22
    row_height = spacing * np.sqrt(3) / 2
    
    row = 0
    while len(points) < n:
        y = 0.1 + row * row_height
        if row % 2 == 0:
            cols = 5
            x_start = 0.08
        else:
            cols = 4
            x_start = 0.08 + spacing / 2
        
        for col in range(cols):
            if len(points) >= n:
                break
            x = x_start + col * spacing
            if 0 <= x <= 1 and 0 <= y <= 1:
                points.append([x, y])
        row += 1
    
    return np.array(points[:n])


def _grid_initialization(n: int) -> np.ndarray:
    """Create a grid initialization."""
    points = []
    spacing = 0.2
    for i in range(5):
        for j in range(5):
            if len(points) >= n:
                break
            x = 0.1 + i * spacing
            y = 0.1 + j * spacing
            points.append([x, y])
    return np.array(points[:n])


def _concentric_initialization(n: int) -> np.ndarray:
    """Create a concentric circle initialization."""
    points = [[0.5, 0.5]]
    for r in [0.12, 0.24, 0.36]:
        num_pts = max(3, int(2 * np.pi * r / 0.18))
        for angle in np.linspace(0, 2 * np.pi, num_pts, endpoint=False):
            if len(points) >= n:
                break
            x = 0.5 + r * np.cos(angle)
            y = 0.5 + r * np.sin(angle)
            if 0 <= x <= 1 and 0 <= y <= 1:
                points.append([x, y])
    return np.array(points[:n])


def _random_initialization(n: int, seed: int) -> np.ndarray:
    """Create a random initialization."""
    rng = np.random.default_rng(seed=seed)
    return rng.random((n, 2)) * 0.8 + 0.1


def _perturb_best(points: np.ndarray, amount: float, seed: int) -> np.ndarray:
    """Perturb the best configuration slightly."""
    rng = np.random.default_rng(seed=seed)
    perturbed = points.copy()
    noise = rng.randn(*perturbed.shape) * amount
    perturbed += noise
    np.clip(perturbed, 0, 1, out=perturbed)
    return perturbed


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # Safety margin: 20s before 600s limit
    
    best_points = None
    best_score = 0.0
    
    # Generate multiple initial configurations
    initial_configs = []
    initial_configs.append(_hexagonal_initialization(n))
    initial_configs.append(_grid_initialization(n))
    initial_configs.append(_concentric_initialization(n))
    
    for seed in range(25):
        initial_configs.append(_random_initialization(n, seed))
    
    # Phase 1: Quick evaluation and selection of top candidates
    scored_configs = []
    for config in initial_configs:
        if time.time() > deadline - 400:
            break
        score = _compute_score_fast(config)
        scored_configs.append((score, config))
    
    scored_configs.sort(key=lambda x: -x[0])
    top_configs = [config for _, config in scored_configs[:8]]
    
    # Phase 2: Simulated annealing on top candidates
    sa_deadline = deadline - 120
    for config in top_configs:
        if time.time() > sa_deadline:
            break
        
        sa_result = _simulated_annealing(config, sa_deadline,
                                          T_init=0.06, T_min=1e-7,
                                          cooling_rate=0.996, max_iters=25000)
        sa_score = _compute_score_fast(sa_result)
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_result.copy()
    
    # Phase 3: Local search refinement
    if best_points is not None:
        ls_deadline = deadline - 40
        ls_result = _local_search(best_points, ls_deadline, max_iters=30000)
        ls_score = _compute_score_fast(ls_result)
        
        if ls_score > best_score:
            best_score = ls_score
            best_points = ls_result.copy()
    
    # Phase 4: Powell refinement
    if best_points is not None and time.time() < deadline - 30:
        powell_deadline = deadline - 15
        powell_result = _powell_refine(best_points, powell_deadline)
        powell_score = _compute_score_fast(powell_result)
        
        if powell_score > best_score:
            best_score = powell_score
            best_points = powell_result.copy()
    
    # Phase 5: Additional random restarts if time permits
    if best_points is not None and time.time() < deadline - 60:
        for restart in range(10):
            if time.time() > deadline - 30:
                break
            
            perturbed = _perturb_best(best_points, 0.06, seed=100 + restart)
            
            # Quick local search
            quick_deadline = deadline - 30
            quick_ls = _local_search(perturbed, quick_deadline, max_iters=5000)
            quick_score = _compute_score_fast(quick_ls)
            
            if quick_score > best_score:
                best_score = quick_score
                best_points = quick_ls.copy()
    
    # Final safety check
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _grid_initialization(n)
    
    return best_points


# EVOLVE-BLOCK-END
