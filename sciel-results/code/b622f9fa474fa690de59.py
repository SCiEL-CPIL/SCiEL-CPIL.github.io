# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize

# Precompute triangle indices for 13 points
_N = 13
_I_IDX, _J_IDX, _K_IDX = [], [], []
for _i in range(_N):
    for _j in range(_i + 1, _N):
        for _k in range(_j + 1, _N):
            _I_IDX.append(_i)
            _J_IDX.append(_j)
            _K_IDX.append(_k)
_I_IDX = np.array(_I_IDX)
_J_IDX = np.array(_J_IDX)
_K_IDX = np.array(_K_IDX)


def _all_triangle_areas(pts):
    """Compute all 286 triangle areas vectorized."""
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


def _exact_score(pts):
    """Compute A_min / A_hull exactly."""
    if len(pts) < 3:
        return 0.0
    areas = _all_triangle_areas(pts)
    a_min = np.min(areas)
    a_hull = _hull_area(pts)
    if a_hull <= 0 or not np.isfinite(a_min) or not np.isfinite(a_hull):
        return 0.0
    return a_min / a_hull


def _normalize_hull_area(pts, target_area=1.0):
    """Scale points so that the convex hull area is target_area."""
    a = _hull_area(pts)
    if a <= 0 or not np.isfinite(a):
        return pts
    scale = np.sqrt(target_area / a)
    return pts * scale


def _delaunay_guided_relax(pts, n_iter=150, time_limit=40.0):
    """
    Structural relaxation using Delaunay triangulation with adaptive step size.
    Combines area equalization forces with strong short-range repulsion.
    """
    start_time = time.time()
    pts = pts.copy()
    n = len(pts)
    
    # Adaptive step size parameters
    initial_step = 0.1
    decay_rate = 0.98
    
    for it in range(n_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(pts)
        except Exception:
            # Fallback: simple repulsion if Delaunay fails
            forces = np.zeros_like(pts)
            for i in range(n):
                for j in range(i + 1, n):
                    d = pts[i] - pts[j]
                    dist = np.linalg.norm(d)
                    if dist < 1e-6:
                        d = np.array([1e-3, 1e-3])
                        dist = np.linalg.norm(d)
                    # Strong repulsion for close points
                    if dist < 0.5:
                        forces[i] += d / (dist * dist * dist) * 0.1
                        forces[j] -= d / (dist * dist * dist) * 0.1
                    else:
                        forces[i] += d / (dist * dist) * 0.01
                        forces[j] -= d / (dist * dist) * 0.01
            pts += initial_step * forces
            pts = _normalize_hull_area(pts, 1.0)
            initial_step *= decay_rate
            continue
        
        # Build adjacency from Delaunay triangles
        neighbors = [set() for _ in range(n)]
        for simplex in tri.simplices:
            for a in simplex:
                for b in simplex:
                    if a != b:
                        neighbors[a].add(b)
        
        forces = np.zeros_like(pts)
        
        # 1. Area Equalization Force
        # Compute all triangle areas
        areas = _all_triangle_areas(pts)
        avg_area = np.mean(areas)
        
        # For each triangle, if area < avg_area, push vertices away from centroid
        for idx in range(len(_I_IDX)):
            a, b, c = _I_IDX[idx], _J_IDX[idx], _K_IDX[idx]
            area = areas[idx]
            if area < avg_area * 0.8:
                centroid = (pts[a] + pts[b] + pts[c]) / 3.0
                strength = (avg_area - area) / (avg_area + 1e-12) * 0.05
                for v in [a, b, c]:
                    d = pts[v] - centroid
                    dist = np.linalg.norm(d)
                    if dist > 1e-8:
                        forces[v] += d / dist * strength
        
        # 2. Laplacian Smoothing
        for i in range(n):
            if neighbors[i]:
                centroid = np.mean([pts[j] for j in neighbors[i]], axis=0)
                forces[i] += 0.1 * (centroid - pts[i])
        
        # 3. Global Repulsion (Stronger for close points)
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-6:
                    d = np.array([1e-3, 1e-3])
                    dist = np.linalg.norm(d)
                
                # Inverse square law, but boosted for very close points
                if dist < 0.5:
                    repulsion = 0.1 / (dist * dist * dist)
                else:
                    repulsion = 0.01 / (dist * dist)
                
                force = d / dist * repulsion
                forces[i] += force
                forces[j] -= force
        
        # Update with adaptive step
        pts = pts + initial_step * forces
        pts = _normalize_hull_area(pts, 1.0)
        
        # Decay step size
        initial_step *= decay_rate
        
        # Safety: ensure finite
        if not np.all(np.isfinite(pts)):
            break
    
    return pts


def _polish(pts, time_limit=15.0):
    """Polish the configuration using Nelder-Mead on the exact score."""
    start_time = time.time()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10
        p = x.reshape(13, 2)
        if not np.all(np.isfinite(p)):
            return 1e10
        s = _exact_score(p)
        if s <= 0:
            return 1e10
        return -s
    
    try:
        result = minimize(
            obj,
            pts.flatten(),
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-12,
                'fatol': 1e-14,
                'adaptive': True
            }
        )
        if np.all(np.isfinite(result.x)):
            return result.x.reshape(13, 2)
    except Exception:
        pass
    return pts


def _generate_initial_configs():
    """Generate a diverse set of initial configurations."""
    configs = []
    rng = np.random.default_rng(42)
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Concentric rings: 1 center, 6 inner (hexagon), 6 outer (hexagon)
    for r_inner in [0.3, 0.4, 0.5]:
        for r_outer in [0.8, 1.0, 1.2]:
            for a_off in [0, np.pi/6]:
                center = np.array([0.0, 0.0])
                inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + a_off
                inner_pts = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
                outer_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + a_off + np.pi/6
                outer_pts = np.column_stack([r_outer * np.cos(outer_angles), r_outer * np.sin(outer_angles)])
                pts = np.vstack([center.reshape(1,2), inner_pts, outer_pts])
                configs.append(_normalize_hull_area(pts, 1.0))
                
    # 3. Hexagonal Lattice Fragment
    # Generate a hexagonal grid and pick 13 points
    hex_pts = []
    a = 1.0
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = a * (i + j/2)
            y = a * (j * np.sqrt(3)/2)
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    # Center at origin
    hex_pts -= hex_pts.mean(axis=0)
    # Pick 13 closest to origin? Or just take a subset
    dists = np.linalg.norm(hex_pts, axis=1)
    indices = np.argsort(dists)[:13]
    hex_config = hex_pts[indices]
    configs.append(_normalize_hull_area(hex_config, 1.0))
    
    # 4. Random perturbations of the regular polygon
    for _ in range(20):
        base = configs[0]
        perturbed = base + rng.normal(0, 0.1, base.shape)
        configs.append(_normalize_hull_area(perturbed, 1.0))
        
    # 5. Random points in unit disk
    for _ in range(10):
        pts = rng.uniform(-1, 1, (13, 2))
        try:
            ConvexHull(pts)
            configs.append(_normalize_hull_area(pts, 1.0))
        except Exception:
            pass
            
    return configs


def heilbronn_convex13() -> np.ndarray:
    """
    Find 13 points maximizing A_min / A_hull.
    Uses Delaunay-guided relaxation followed by Nelder-Mead polishing.
    """
    start_time = time.time()
    time_limit = 570  # Safety margin from 600s limit
    
    def time_check(remaining=20):
        return time.time() - start_time < time_limit - remaining
        
    best_pts = None
    best_score = 0.0
    
    def update_best(pts):
        nonlocal best_pts, best_score
        if pts is None or not np.all(np.isfinite(pts)) or len(pts) != 13:
            return
        score = _exact_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
            
    # Phase 1: Generate and evaluate initial configs
    initial_configs = _generate_initial_configs()
    
    for pts in initial_configs:
        if not time_check(100):
            break
        update_best(pts)
        
    # Phase 2: Relaxation and Polishing for top candidates
    scored_configs = []
    for pts in initial_configs:
        score = _exact_score(pts)
        scored_configs.append((score, pts))
    scored_configs.sort(key=lambda x: -x[0])
    
    # Take top 15 configurations to increase diversity in relaxation
    top_configs = [cfg for _, cfg in scored_configs[:15]]
    
    for init_pts in top_configs:
        if not time_check(50):
            break
            
        pts = init_pts.copy()
        
        # Delaunay-guided relaxation
        relaxed = _delaunay_guided_relax(pts, n_iter=150, time_limit=30.0)
        update_best(relaxed)
        
        # Polish
        if time_check(20):
            polished = _polish(relaxed, time_limit=10.0)
            update_best(polished)
    
    # Phase 3: If we have time, do a final polish on the best
    if best_pts is not None and time_check(30):
        final = _polish(best_pts, time_limit=20.0)
        update_best(final)
        
        # Optional: Second polish with perturbation if time allows
        if time_check(15):
            perturbed_best = best_pts + np.random.normal(0, 0.01, best_pts.shape)
            perturbed_best = _normalize_hull_area(perturbed_best, 1.0)
            final2 = _polish(perturbed_best, time_limit=10.0)
            update_best(final2)
    
    # Final safety check
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        angles13 = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_pts = np.column_stack([np.cos(angles13), np.sin(angles13)])
    
    return best_pts


# EVOLVE-BLOCK-END
