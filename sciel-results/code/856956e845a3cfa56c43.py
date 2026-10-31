import numpy as np
import time
import warnings
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

warnings.filterwarnings('ignore')

N = 13

def _simulated_annealing(points, deadline, best_score_ref):
    """Run simulated annealing with adaptive step size and periodic local polish."""
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_s = current_score
    
    t_start = time.time()
    time_budget = deadline - 5.0
    
    # Estimate iterations: each _score call takes ~0.5-2ms, so ~500-2000 per second
    total_time = max(1.0, time_budget - t_start)
    est_iters = int(total_time * 800)
    est_iters = max(200, min(est_iters, 50000))
    
    T0 = 0.1
    T_end = 0.0005
    alpha = (T_end / T0) ** (1.0 / max(1, est_iters - 1))
    
    rng = np.random.default_rng(42)
    polish_interval = max(1, est_iters // 8)
    last_polish = 0
    
    for it in range(est_iters):
        if time.time() > time_budget:
            break
        
        T = T0 * (alpha ** it)
        
        # Adaptive perturbation: mix of global and local moves
        if rng.random() < 0.7:
            # Local move: perturb subset of points
            n_perturb = rng.integers(1, 5)
            idx = rng.choice(N, size=n_perturb, replace=False)
            noise = np.zeros((N, 2))
            noise[idx] = rng.normal(0, T * 0.5, size=(n_perturb, 2))
        else:
            # Global move
            noise = rng.normal(0, T, size=(N, 2))
        
        candidate = current + noise
        candidate = np.clip(candidate, -5, 5)
        
        cand_score = _score(candidate)
        
        if cand_score > current_score:
            current = candidate
            current_score = cand_score
            if current_score > best_s:
                best_s = current_score
                best = current.copy()
        else:
            delta = current_score - cand_score
            if delta < 1e-12:
                current = candidate
                current_score = cand_score
            else:
                prob = np.exp(-delta / max(T, 1e-10))
                if rng.random() < prob:
                    current = candidate
                    current_score = cand_score
        
        # Periodic local polish to escape shallow local minima
        if it - last_polish >= polish_interval and time.time() < time_budget - 2:
            try:
                res = minimize(
                    _objective_full,
                    current.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                new_score = _score(res.x.reshape(N, 2))
                if new_score > current_score:
                    current = res.x.reshape(N, 2)
                    current_score = new_score
                    if current_score > best_s:
                        best_s = current_score
                        best = current.copy()
            except Exception:
                pass
            last_polish = it
    
    return best, best_s

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _normalize(points):
    """Center and normalize points."""
    centroid = np.mean(points, axis=0)
    pts = points - centroid
    # Normalize so hull area = 1 for consistent comparison
    hull_area = _hull_area(pts)
    if hull_area > 1e-12:
        pts = pts / np.sqrt(hull_area)
    return pts

def _objective_full(x_flat):
    """Full objective: minimize negative score."""
    points = x_flat.reshape(N, 2)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _smooth_objective(x_flat, p=30):
    """Smooth surrogate using log-sum-exp based soft-min for better conditioning."""
    points = x_flat.reshape(N, 2)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    if np.allclose(points, 0):
        return 1e10
    
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    areas = np.maximum(areas, 1e-15)
    
    # Log-sum-exp based soft-min: approximates min(areas) smoothly
    # soft_min = -p * log(mean(exp(-areas/p)))
    neg_areas_over_p = -areas / p
    max_neg = np.max(neg_areas_over_p)
    log_sum = np.log(np.mean(np.exp(neg_areas_over_p - max_neg))) + max_neg
    soft_min = -p * log_sum
    
    score = soft_min / hull_area
    if score <= 0:
        return 1e10
    return -score

def _coordinate_descent_polish(points, deadline, max_rounds=10):
    """Coordinate descent: optimize each point one at a time using Nelder-Mead."""
    pts = points.copy()
    best_s = _score(pts)
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            
            def obj_1d(x):
                new_pts = pts.copy()
                new_pts[i] = x
                s = _score(new_pts)
                if s <= 0:
                    return 1e10
                return -s
            
            x0 = pts[i].copy()
            try:
                res = minimize(
                    obj_1d,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                new_pts = pts.copy()
                new_pts[i] = res.x
                new_s = _score(new_pts)
                if new_s > best_s + 1e-12:
                    pts = new_pts
                    best_s = new_s
                    improved = True
            except Exception:
                pass
        
        if not improved:
            break
    
    return pts, best_s

def _make_ring_config(n_outer, n_inner, r_inner, angle_offset):
    """Create a two-ring configuration."""
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_inner, endpoint=False)
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    return np.vstack([outer_pts, inner_pts])

def _optimize_affine_grid(G, deadline):
    """
    Optimize affine transformation for a fixed grid G.
    G is (N, 2).
    Points P = G @ A.T
    A is 2x2.
    """
    def obj(a_flat):
        A = a_flat.reshape(2, 2)
        P = G @ A.T
        # Center P to avoid drift? Affine transform includes translation if we added a column, 
        # but we are only optimizing shape. Centering P makes the score calculation stable.
        P = P - np.mean(P, axis=0)
        s = _score(P)
        if s <= 0:
            return 1e10
        return -s

    # Initial guess: Identity
    a0 = np.eye(2).flatten()
    
    try:
        res = minimize(
            obj,
            a0,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        A_opt = res.x.reshape(2, 2)
        P_opt = G @ A_opt.T
        P_opt = P_opt - np.mean(P_opt, axis=0)
        return P_opt
    except Exception:
        return None

def _get_base_grids():
    """Generate candidate base grids for affine optimization."""
    grids = []
    
    # Grid 1: 4x4 centered, remove 3 corners
    # Coords: -1.5, -0.5, 0.5, 1.5
    coords = [-1.5, -0.5, 0.5, 1.5]
    all_pts = np.array([(x, y) for x in coords for y in coords])
    # Remove (-1.5, -1.5), (1.5, -1.5), (-1.5, 1.5)
    to_remove = [(-1.5, -1.5), (1.5, -1.5), (-1.5, 1.5)]
    mask = np.array([not (p[0] == r[0] and p[1] == r[1]) for p in all_pts for r in [to_remove[0], to_remove[1], to_remove[2]] if False] or [True]*16) # Dummy logic, do it properly
    
    mask = np.ones(16, dtype=bool)
    for r in to_remove:
        idx = np.where((all_pts[:, 0] == r[0]) & (all_pts[:, 1] == r[1]))[0]
        if len(idx) > 0:
            mask[idx[0]] = False
            
    G1 = all_pts[mask]
    if len(G1) == N:
        grids.append(G1)

    # Grid 2: 3x5 centered, remove 2 corners
    # x: -1.5, 0, 1.5
    # y: -2.0, -1.0, 0.0, 1.0, 2.0
    coords_x = [-1.5, 0.0, 1.5]
    coords_y = [-2.0, -1.0, 0.0, 1.0, 2.0]
    all_pts_35 = np.array([(x, y) for x in coords_x for y in coords_y])
    to_remove_35 = [(-1.5, -2.0), (1.5, -2.0)]
    mask_35 = np.ones(15, dtype=bool)
    for r in to_remove_35:
        idx = np.where((all_pts_35[:, 0] == r[0]) & (all_pts_35[:, 1] == r[1]))[0]
        if len(idx) > 0:
            mask_35[idx[0]] = False
            
    G2 = all_pts_35[mask_35]
    if len(G2) == N:
        grids.append(G2)

    # Grid 3: Hexagonal 1+6+6
    # Center
    center = np.array([[0.0, 0.0]])
    # Ring 1 (6 points)
    angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    r1 = 1.0
    ring1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    # Ring 2 (6 points) - pick 6 from the 12 positions of a hexagonal ring
    # Hexagonal ring 2 points are at distance 2.
    # Angles are same as ring 1 and offset by pi/6?
    # In hexagonal lattice, ring 2 has 12 points.
    # Let's just take 6 points at angles 0, pi/3, ... distance 2.
    # This creates a star of 6 points.
    angles2 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    r2 = 2.0
    ring2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    
    G3 = np.vstack([center, ring1, ring2])
    if len(G3) == N:
        grids.append(G3)
        
    return grids

def _optimize_from_start(x0, deadline, best_score_ref):
    """Run multi-phase optimization from a starting point."""
    best_x = x0.copy()
    best_s = _score(x0.reshape(N, 2))
    
    # Phase 1: Smooth optimization (shorter to leave time for SA)
    if time.time() < deadline - 15:
        try:
            res = minimize(
                lambda x: _smooth_objective(x, p=30),
                x0,
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if _score(res.x.reshape(N, 2)) > best_s:
                best_s = _score(res.x.reshape(N, 2))
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Hard objective polish (shorter)
    if time.time() < deadline - 10:
        try:
            res_nm = minimize(
                _objective_full,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            s_nm = _score(res_nm.x.reshape(N, 2))
            if s_nm > best_s:
                best_s = s_nm
                best_x = res_nm.x.copy()
        except Exception:
            pass
    
    return best_x, best_s

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
            
    # Generate diverse initial configurations
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]).flatten())
    
    # 2. Two-ring configurations with various parameters
    for n_out in [6, 7, 8]:
        n_in = N - n_out
        for r_in in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for offset in [0, np.pi / n_out, np.pi / (2 * n_out)]:
                pts = _make_ring_config(n_out, n_in, r_in, offset)
                inits.append(pts.flatten())
                
    # 3. Three-ring configurations
    for n1, n2, n3 in [(5, 4, 4), (6, 5, 2), (7, 3, 3), (4, 4, 5)]:
        for r2 in [0.5, 0.6, 0.7]:
            for r3 in [0.2, 0.3, 0.4]:
                a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
                a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n1
                a3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + np.pi / n2
                pts = np.vstack([
                    np.column_stack([np.cos(a1), np.sin(a1)]),
                    np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
                    np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)])
                ])
                inits.append(pts.flatten())
                
    # 4. Random starts
    rng = np.random.default_rng(12345)
    for _ in range(15):
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts.flatten())
        
    # 5. Affine Grid Optimizations
    base_grids = _get_base_grids()
    for G in base_grids:
        if time.time() > deadline - 25:
            break
        P_opt = _optimize_affine_grid(G, deadline)
        if P_opt is not None:
            inits.append(P_opt.flatten())
            _update_best(P_opt)

    # Shuffle inits to diversify
    rng2 = np.random.default_rng(999)
    indices = rng2.permutation(len(inits))
    inits = [inits[i] for i in indices]
    
    # Optimize from each start
    # Leave more time for SA: stop local opt when 25s remain
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 25:
            break
        
        t_start = time.time()
        x_opt, s_opt = _optimize_from_start(x0, deadline, best_score)
        _update_best(x_opt.reshape(N, 2))
        
        elapsed = time.time() - t_start
        if i % 5 == 0:
            print(f"Init {i}: score={best_score:.6f}, elapsed={elapsed:.1f}s", flush=True)
            
    # Final polish on best with local optimization
    if best_points is not None and time.time() < deadline - 10:
        try:
            # Coordinate descent first for finer tuning
            pts_cd, s_cd = _coordinate_descent_polish(best_points, deadline - 5, max_rounds=5)
            _update_best(pts_cd)
            
            # Then Nelder-Mead with tight tolerance
            res_final = minimize(
                _objective_full,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            _update_best(res_final.x.reshape(N, 2))
        except Exception:
            pass
            
    # Simulated Annealing to escape local optima
    if best_points is not None and time.time() < deadline - 10:
        try:
            sa_best, sa_score = _simulated_annealing(best_points, deadline - 5, best_score)
            _update_best(sa_best)
        except Exception:
            pass
            
    # One more local polish after SA: coordinate descent + high-precision NM
    if best_points is not None and time.time() < deadline - 1:
        try:
            # Coordinate descent for final fine-tuning
            pts_final_cd, s_final_cd = _coordinate_descent_polish(best_points, deadline - 0.5, max_rounds=10)
            _update_best(pts_final_cd)
            
            # High-precision Nelder-Mead
            res_post = minimize(
                _objective_full,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            _update_best(res_post.x.reshape(N, 2))
            
            # Final coordinate descent after NM
            pts_cd2, s_cd2 = _coordinate_descent_polish(best_points, deadline - 0.1, max_rounds=5)
            _update_best(pts_cd2)
        except Exception:
            pass
            
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
