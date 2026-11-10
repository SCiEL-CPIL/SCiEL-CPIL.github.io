# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize

_N = 13
_TRIS = np.array([(i, j, k) for i in range(_N) for j in range(i+1, _N) for k in range(j+1, _N)], dtype=np.int32)
_I_IDX = _TRIS[:, 0]
_J_IDX = _TRIS[:, 1]
_K_IDX = _TRIS[:, 2]


def _triangle_areas(pts):
    x1, y1 = pts[_I_IDX, 0], pts[_I_IDX, 1]
    x2, y2 = pts[_J_IDX, 0], pts[_J_IDX, 1]
    x3, y3 = pts[_K_IDX, 0], pts[_K_IDX, 1]
    areas = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
    return areas


def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    if len(pts) < 3:
        return 0.0
    areas = _triangle_areas(pts)
    a_min = np.min(areas)
    a_hull = _hull_area(pts)
    if a_hull <= 0 or not np.isfinite(a_min) or not np.isfinite(a_hull):
        return 0.0
    return a_min / a_hull


def _regular_polygon(n=13):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _hexagonal_lattice_init(n=13):
    """Create a hexagonal lattice arrangement."""
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # First ring
    for k in range(6):
        angle = k * np.pi / 3
        pts.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
    # Second ring
    for k in range(6):
        angle = k * np.pi / 3 + np.pi / 6
        pts.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
    pts = np.array(pts[:n])
    return pts


def _delaunay_local_search(pts, n_iters=200, time_limit=600, start_time=None):
    """
    Local search using Delaunay triangulation.
    For each interior point, try moving it to improve the minimum triangle area.
    """
    if start_time is None:
        start_time = time.time()
    
    pts = pts.copy()
    best_score = _score(pts)
    
    for it in range(n_iters):
        if time.time() - start_time > time_limit - 30:
            break
        
        # Get Delaunay triangulation
        try:
            tri = Delaunay(pts)
        except Exception:
            break
        
        # Find interior points (not on convex hull)
        try:
            hull = ConvexHull(pts)
            hull_mask = np.zeros(_N, dtype=bool)
            hull_mask[hull.vertices] = True
        except Exception:
            break
        
        internal_indices = np.where(~hull_mask)[0]
        if len(internal_indices) == 0:
            break
        
        areas = _triangle_areas(pts)
        min_area = np.min(areas)
        
        improved = False
        for idx in internal_indices:
            # Get triangles involving this point
            tri_indices = np.where(
                (tri.simplices == idx).any(axis=1)
            )
            
            if len(tri_indices) == 0:
                continue
            
            # Try moving the point in several directions
            current_pos = pts[idx].copy()
            best_new_pos = current_pos.copy()
            best_local_score = best_score
            
            # Directions: toward each neighbor in Delaunay
            neighbors = set()
            for ti in tri_indices:
                for v in tri.simplices[ti]:
                    if v != idx:
                        neighbors.add(v)
            
            for nb in neighbors:
                direction = pts[nb] - current_pos
                norm = np.linalg.norm(direction)
                if norm < 1e-10:
                    continue
                direction /= norm
                
                # Try several step sizes
                for step in [0.02, 0.05, 0.1, 0.15, 0.2]:
                    new_pos = current_pos + step * direction
                    pts[idx] = new_pos
                    new_score = _score(pts)
                    if new_score > best_local_score:
                        best_local_score = new_score
                        best_new_pos = new_pos.copy()
            
            # Also try moving away from centroid of surrounding triangles
            centroids = []
            for ti in tri_indices:
                c = pts[tri.simplices[ti]].mean(axis=0)
                centroids.append(c)
            if centroids:
                centroid = np.mean(centroids, axis=0)
                direction = current_pos - centroid
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction /= norm
                    for step in [0.02, 0.05, 0.1]:
                        new_pos = current_pos + step * direction
                        pts[idx] = new_pos
                        new_score = _score(pts)
                        if new_score > best_local_score:
                            best_local_score = new_score
                            best_new_pos = new_pos.copy()
            
            if np.linalg.norm(best_new_pos - current_pos) > 1e-12:
                pts[idx] = best_new_pos
                best_score = best_local_score
                improved = True
            else:
                pts[idx] = current_pos
        
        if not improved:
            break
    
    return pts


def _simulated_annealing(pts, n_iters=5000, t_start=0.1, t_end=0.001, time_limit=600, start_time=None, rng=None):
    """
    Simulated annealing to escape local optima.
    """
    if start_time is None:
        start_time = time.time()
    if rng is None:
        rng = np.random.default_rng(42)
    
    pts = pts.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    current_score = best_score
    
    for it in range(n_iters):
        if time.time() - start_time > time_limit - 30:
            break
        
        # Temperature schedule (geometric cooling)
        t = t_start * (t_end / t_start) ** (it / max(n_iters, 1))
        
        # Propose a move: pick a random point and perturb it
        idx = rng.integers(0, _N)
        step_size = 0.05 + 0.1 * t / t_start
        delta = rng.normal(0, step_size, 2)
        
        old_pos = pts[idx].copy()
        pts[idx] += delta
        
        new_score = _score(pts)
        
        if new_score >= current_score or rng.random() < np.exp((new_score - current_score) / max(t, 1e-10)):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
        else:
            pts[idx] = old_pos
    
    return best_pts


def _gradient_polish(pts, time_limit=600, start_time=None):
    """
    Use gradient-based optimization as a final polish.
    """
    if start_time is None:
        start_time = time.time()
    
    x0 = pts.flatten()
    n = _N
    
    def objective(x):
        p = np.column_stack([x[0:n], x[n:2*n]])
        return -_score(p)
    
    # Try L-BFGS-B with numerical gradient
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-6}
        )
        if result.fun < objective(x0) - 1e-12:
            new_pts = np.column_stack([result.x[0:n], result.x[n:2*n]])
            if _score(new_pts) > _score(pts):
                return new_pts
    except Exception:
        pass
    
    return pts


def _concentric_rings_init(inner_n, outer_n, inner_r=0.5):
    """Create concentric ring initialization."""
    inner_angles = np.linspace(0, 2*np.pi, inner_n, endpoint=False)
    outer_angles = np.linspace(0, 2*np.pi, outer_n, endpoint=False) + np.pi/outer_n
    
    inner_pts = np.column_stack([inner_r*np.cos(inner_angles), inner_r*np.sin(inner_angles)])
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    
    pts = np.vstack([inner_pts, outer_pts])
    return pts


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area.
    """
    start_time = time.time()
    time_limit = 600  # Safety margin from 630s
    
    best_pts = _regular_polygon(_N)
    best_score = _score(best_pts)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Regular polygon
    initializations.append(_regular_polygon(_N))
    
    # 2. Hexagonal lattice
    initializations.append(_hexagonal_lattice_init(_N))
    
    # 3. Concentric rings
    for inner_n, outer_n, r in [(5, 8, 0.5), (6, 7, 0.4), (4, 9, 0.6), (3, 10, 0.45)]:
        initializations.append(_concentric_rings_init(inner_n, outer_n, r))
    
    # 4. Grid-like
    grid_pts = []
    for i in range(4):
        for j in range(4):
            grid_pts.append([0.3 + 0.4*i, 0.3 + 0.4*j])
    initializations.append(np.array(grid_pts[:_N]))
    
    # 5. Random perturbations of regular polygon
    rng = np.random.default_rng(12345)
    for seed in range(8):
        rng2 = np.random.default_rng(seed)
        perturbed = _regular_polygon(_N)
        perturbed += rng2.normal(0, 0.15, (_N, 2))
        initializations.append(perturbed)
    
    # 6. Random points
    for seed in range(5):
        rng2 = np.random.default_rng(100 + seed)
        rand_pts = rng2.random((_N, 2))
        initializations.append(rand_pts)
    
    # 7. Square-based initialization
    sq_pts = []
    for i in range(4):
        for j in range(4):
            sq_pts.append([0.2 + 0.6*i/3, 0.2 + 0.6*j/3])
    initializations.append(np.array(sq_pts[:_N]))
    
    # Process each initialization
    for init_idx, init_pts in enumerate(initializations):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 60:
            break
        
        pts = init_pts.copy()
        
        # Phase 1: Delaunay local search
        pts = _delaunay_local_search(pts, n_iters=300, time_limit=time_limit, start_time=start_time)
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
        
        elapsed = time.time() - start_time
        if elapsed > time_limit - 40:
            break
        
        # Phase 2: Simulated annealing
        rng_sa = np.random.default_rng(500 + init_idx)
        pts_sa = _simulated_annealing(
            pts, n_iters=3000, t_start=0.08, t_end=0.0005,
            time_limit=time_limit, start_time=start_time, rng=rng_sa
        )
        score_sa = _score(pts_sa)
        if score_sa > best_score:
            best_score = score_sa
            best_pts = pts_sa.copy()
        
        elapsed = time.time() - start_time
        if elapsed > time_limit - 20:
            break
        
        # Phase 3: Gradient polish
        pts_polish = _gradient_polish(pts_sa, time_limit=time_limit, start_time=start_time)
        score_polish = _score(pts_polish)
        if score_polish > best_score:
            best_score = score_polish
            best_pts = pts_polish.copy()
    
    # Final refinement on best solution
    elapsed = time.time() - start_time
    if elapsed < time_limit - 15 and best_score > 0:
        # More aggressive Delaunay search
        pts_final = _delaunay_local_search(
            best_pts, n_iters=500, time_limit=time_limit, start_time=start_time
        )
        score_final = _score(pts_final)
        if score_final > best_score:
            best_score = score_final
            best_pts = pts_final.copy()
        
        # One more SA pass
        elapsed = time.time() - start_time
        if elapsed < time_limit - 5:
            rng_final = np.random.default_rng(9999)
            pts_final2 = _simulated_annealing(
                best_pts, n_iters=2000, t_start=0.03, t_end=0.0001,
                time_limit=time_limit, start_time=start_time, rng=rng_final
            )
            score_final2 = _score(pts_final2)
            if score_final2 > best_score:
                best_score = score_final2
                best_pts = pts_final2.copy()
    
    # Validate
    if best_score <= 0 or not np.all(np.isfinite(best_pts)):
        return _regular_polygon(_N)
    
    return best_pts


# EVOLVE-BLOCK-END
