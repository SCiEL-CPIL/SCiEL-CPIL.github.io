# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from itertools import combinations


def _triangle_areas(points: np.ndarray) -> np.ndarray:
    """Compute areas of all C(n,3) triangles. Returns 2D array of areas."""
    n = points.shape[0]
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n-2), np.arange(1, n-1), np.arange(2, n),
        indexing='ij'
    )
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    cross = np.abs(v1x * v2y - v1y * v2x)
    return 0.5 * cross, (i_idx, j_idx, k_idx)


def _get_triangle_indices(n: int):
    """Precompute triangle index arrays for n points."""
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n-2), np.arange(1, n-1), np.arange(2, n),
        indexing='ij'
    )
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    return i_idx[mask], j_idx[mask], k_idx[mask]


def _compute_score(points: np.ndarray) -> float:
    """Compute A_min / A_hull."""
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
    
    areas, _ = _triangle_areas(points)
    min_area = float(np.min(areas))
    if min_area <= 0:
        return 0.0
    return min_area / A_hull


def _compute_score_fast(points: np.ndarray, tri_indices: tuple) -> float:
    """Compute A_min / A_hull using precomputed triangle indices."""
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
    
    i_idx, j_idx, k_idx = tri_indices
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    cross = np.abs(v1x * v2y - v1y * v2x)
    areas = 0.5 * cross
    min_area = float(np.min(areas))
    if min_area <= 0:
        return 0.0
    return min_area / A_hull


def _find_worst_triangle(points: np.ndarray, tri_indices: tuple) -> tuple:
    """Find the index of the smallest triangle and its area."""
    i_idx, j_idx, k_idx = tri_indices
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    cross = np.abs(v1x * v2y - v1y * v2x)
    areas = 0.5 * cross
    min_idx = np.argmin(areas)
    return min_idx, areas[min_idx], i_idx[min_idx], j_idx[min_idx], k_idx[min_idx]


def _build_hex_configuration(r1: float, r2: float, center: tuple) -> np.ndarray:
    """
    Build a hexagonal configuration:
    - 1 center
    - 6 on a hexagon of radius r1
    - 6 on a hexagon of radius r2
    
    That's 13 points.
    """
    cx, cy = center
    points = [[cx, cy]]
    
    # 6 points on inner hexagon
    for i in range(6):
        angle = np.pi / 3 * i
        x = cx + r1 * np.cos(angle)
        y = cy + r1 * np.sin(angle)
        points.append([x, y])
    
    # 6 points on outer hexagon
    for i in range(6):
        angle = np.pi / 3 * i
        x = cx + r2 * np.cos(angle)
        y = cy + r2 * np.sin(angle)
        points.append([x, y])
    
    return np.array(points)


def _build_regular_polygon(n: int, center: tuple, radius: float = 0.45) -> np.ndarray:
    """Build a regular n-gon centered at center with given radius."""
    cx, cy = center
    points = []
    for i in range(n):
        angle = 2 * np.pi * i / n - np.pi / 2  # Start from top
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        points.append([x, y])
    return np.array(points)


def _build_concentric_config(n: int, center: tuple) -> np.ndarray:
    """Build concentric polygon configuration."""
    cx, cy = center
    points = [[cx, cy]]
    
    # Ring 1: 5 points (pentagon)
    r1 = 0.18
    for i in range(5):
        angle = 2 * np.pi * i / 5
        points.append([cx + r1 * np.cos(angle), cy + r1 * np.sin(angle)])
    
    # Ring 2: 7 points (heptagon)
    r2 = 0.38
    for i in range(7):
        angle = 2 * np.pi * i / 7
        points.append([cx + r2 * np.cos(angle), cy + r2 * np.sin(angle)])
    
    return np.array(points[:n])


def _build_grid_config(n: int) -> np.ndarray:
    """Build a grid configuration, 4x4 minus 3 = 13."""
    points = []
    for i in range(4):
        for j in range(4):
            if len(points) >= n:
                break
            points.append([0.1 + i * 0.26, 0.1 + j * 0.26])
    return np.array(points[:n])


def _build_random_config(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, 2)) * 0.8 + 0.1


def _local_refine(points: np.ndarray, deadline: float, max_iters: int = 50000) -> np.ndarray:
    """
    Fast local search: at each step, find the worst triangle, then try to move
    one of its vertices to improve. Uses precomputed triangle indices.
    """
    best = points.copy()
    best_score = _compute_score(best)
    n = len(best)
    
    tri_indices = _get_triangle_indices(n)
    
    step_size = 0.02
    min_step = 1e-9
    decay = 0.92
    stall = 0
    max_stall = 30
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Find worst triangle
        min_idx, min_area, wi, wj, wk = _find_worst_triangle(best, tri_indices)
        
        # Hull area
        try:
            A_hull = ConvexHull(best).volume
        except:
            break
        
        improved = False
        
        # Try moving each vertex of the worst triangle
        for idx in (wi, wj, wk):
            # Try several random directions
            for _ in range(8):
                dx = np.random.randn() * step_size
                dy = np.random.randn() * step_size
                
                cand = best.copy()
                cand[idx, 0] = np.clip(best[idx, 0] + dx, 0, 1)
                cand[idx, 1] = np.clip(best[idx, 1] + dy, 0, 1)
                
                s = _compute_score_fast(cand, tri_indices)
                if s > best_score:
                    best = cand
                    best_score = s
                    improved = True
                    stall = 0
                    break
            if improved:
                break
        
        if not improved:
            stall += 1
            if stall >= max_stall:
                step_size *= decay
                if step_size < min_step:
                    break
                stall = 0
                max_stall = max(15, max_stall // 2)
    
    return best


def _simulated_annealing(points: np.ndarray, deadline: float,
                          T_init: float = 0.03, T_min: float = 1e-9,
                          cooling_rate: float = 0.999, max_iters: int = 40000) -> np.ndarray:
    """Simulated annealing with precomputed triangle indices."""
    best = points.copy()
    cur = points.copy()
    cur_score = _compute_score(cur)
    best_score = cur_score
    n = len(cur)
    
    tri_indices = _get_triangle_indices(n)
    
    step_size = 0.03
    min_step = 1e-9
    accept = 0
    reject = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        T = max(T_init * (cooling_rate ** it), T_min)
        
        idx = np.random.randint(n)
        dx = np.random.randn() * step_size
        dy = np.random.randn() * step_size
        
        cand = cur.copy()
        cand[idx, 0] = np.clip(cur[idx, 0] + dx, 0, 1)
        cand[idx, 1] = np.clip(cur[idx, 1] + dy, 0, 1)
        
        cand_score = _compute_score_fast(cand, tri_indices)
        
        if cand_score >= cur_score:
            cur = cand
            cur_score = cand_score
            accept += 1
            if cur_score > best_score:
                best_score = cur_score
                best = cand.copy()
        else:
            delta = cur_score - cand_score
            if np.random.random() < np.exp(-delta / max(T, 1e-12)):
                cur = cand
                cur_score = cand_score
                accept += 1
            else:
                reject += 1
        
        if (accept + reject) % 300 == 0 and (accept + reject) > 0:
            rate = accept / (accept + reject)
            if rate < 0.15:
                step_size *= 0.8
            elif rate > 0.85:
                step_size *= 1.15
            step_size = max(min_step, min(step_size, 0.08))
            accept = 0
            reject = 0
    
    return best


def heilbronn_convex13() -> np.ndarray:
    """
    Construct 13 points to maximize A_min / A_hull.
    Uses known hexagonal structure and regular polygon as seeds, then refines.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    configs = []
    
    # Regular 13-gon (known to be good for convex hull configurations)
    for radius in [0.40, 0.45, 0.50]:
        cfg = _build_regular_polygon(13, (0.5, 0.5), radius)
        configs.append(cfg)
    
    # Hexagonal: center + 6 inner + 6 outer
    for r1 in [0.15, 0.18, 0.20, 0.22]:
        for r2 in [0.35, 0.40, 0.42, 0.45]:
            cfg = _build_hex_configuration(r1, r2, (0.5, 0.5))
            configs.append(cfg)
    
    # Concentric pentagon + heptagon
    configs.append(_build_concentric_config(13, (0.5, 0.5)))
    
    # Grid
    configs.append(_build_grid_config(13))
    
    # Random seeds
    for seed in range(10):
        configs.append(_build_random_config(13, seed))
    
    # Evaluate all configs
    scored = []
    for cfg in configs:
        if time.time() > deadline - 300:
            break
        s = _compute_score(cfg)
        scored.append((s, cfg))
    
    scored.sort(key=lambda x: -x[0])
    
    # Take top 6 for refinement (include regular polygon if it scored well)
    top_configs = [cfg for _, cfg in scored[:6]]
    
    # Phase 1: Simulated annealing on top configs
    sa_deadline = deadline - 120
    for cfg in top_configs:
        if time.time() > sa_deadline:
            break
        sa_result = _simulated_annealing(cfg, sa_deadline,
                                          T_init=0.025, T_min=1e-9,
                                          cooling_rate=0.9992, max_iters=30000)
        s = _compute_score(sa_result)
        if s > best_score:
            best_score = s
            best_points = sa_result.copy()
    
    # Phase 2: Local search refinement
    if best_points is not None:
        ls_deadline = deadline - 40
        ls_result = _local_refine(best_points, ls_deadline, max_iters=50000)
        s = _compute_score(ls_result)
        if s > best_score:
            best_score = s
            best_points = ls_result.copy()
    
    # Phase 3: Additional random restarts if time permits
    if best_points is not None and time.time() < deadline - 60:
        for _ in range(8):
            if time.time() > deadline - 40:
                break
            perturbed = best_points.copy()
            noise = np.random.randn(*perturbed.shape) * 0.05
            perturbed += noise
            np.clip(perturbed, 0, 1, out=perturbed)
            
            quick_ls = _local_refine(perturbed, deadline - 40, max_iters=15000)
            s = _compute_score(quick_ls)
            if s > best_score:
                best_score = s
                best_points = quick_ls.copy()
    
    # Safety
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _build_regular_polygon(13, (0.5, 0.5), 0.45)
    
    return best_points


# EVOLVE-BLOCK-END
