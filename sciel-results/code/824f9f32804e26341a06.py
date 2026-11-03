import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools
from itertools import combinations

N = 13

# Precompute all triples and their index arrays for vectorized area computation
_TRIPLES = list(combinations(range(N), 3))
_I_IDX = np.array([t[0] for t in _TRIPLES])
_J_IDX = np.array([t[1] for t in _TRIPLES])
_K_IDX = np.array([t[2] for t in _TRIPLES])
_N_TRIPLES = len(_TRIPLES)

def _triangle_areas(pts):
    """Vectorized computation of all triangle areas."""
    x = pts[:, 0]
    y = pts[:, 1]
    v1x = x[_J_IDX] - x[_I_IDX]
    v1y = y[_J_IDX] - y[_I_IDX]
    v2x = x[_K_IDX] - x[_I_IDX]
    v2y = y[_K_IDX] - y[_I_IDX]
    return 0.5 * np.abs(v1x * v2y - v1y * v2x)

def _min_triangle_area(pts):
    return float(np.min(_triangle_areas(pts)))

def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _min_triangle_area(pts) / hull_a

def _soft_min(areas, alpha):
    """Soft minimum via log-sum-exp."""
    if len(areas) == 0:
        return 0.0
    M = np.min(areas)
    shifted = areas - M
    # Guard against overflow
    exp_terms = np.exp(-alpha * shifted)
    if np.any(~np.isfinite(exp_terms)):
        return M
    log_sum = -alpha * M + np.log(np.sum(exp_terms))
    return -log_sum / alpha

def _neg_soft_score(params, alpha):
    pts = params.reshape(-1, 2)
    areas = _triangle_areas(pts)
    soft_min = _soft_min(areas, alpha)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return -(soft_min / hull_a)

def _neg_score(params):
    pts = params.reshape(-1, 2)
    return -_score(pts)

def _generate_initializations():
    """Generate diverse initial configurations."""
    rng = np.random.RandomState(123)
    inits = []
    
    # Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Two-ring configurations
    for split in [(7, 6), (8, 5), (9, 4), (6, 7), (10, 3), (12, 1)]:
        n1, n2 = split
        for r2 in [0.3, 0.4, 0.5, 0.6, 0.7]:
            ang1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
            ang2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / max(n2, 1)
            ring1 = np.column_stack([np.cos(ang1), np.sin(ang1)])
            ring2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
            inits.append(np.vstack([ring1, ring2]))
    
    # Three-ring configurations
    for n1, n2, n3 in [(6, 4, 3), (5, 4, 4), (7, 3, 3), (8, 3, 2), (5, 5, 3)]:
        for r2 in [0.4, 0.55]:
            for r3 in [0.2, 0.35]:
                ang1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
                ang2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
                ang3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + np.pi / n3
                ring1 = np.column_stack([np.cos(ang1), np.sin(ang1)])
                ring2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
                ring3 = np.column_stack([r3 * np.cos(ang3), r3 * np.sin(ang3)])
                inits.append(np.vstack([ring1, ring2, ring3]))
    
    # Random in disk
    for _ in range(12):
        r = rng.rand(N) ** 0.5
        theta = rng.rand(N) * 2 * np.pi
        inits.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
    
    # Random concentric
    for _ in range(15):
        n_circles = rng.choice([2, 3])
        if n_circles == 2:
            n1 = rng.choice([5, 6, 7, 8])
            n2 = N - n1
            r2 = rng.uniform(0.3, 0.7)
            ang1 = np.linspace(0, 2 * np.pi, n1, endpoint=False) + rng.uniform(0, 2 * np.pi)
            ang2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + rng.uniform(0, 2 * np.pi)
            ring1 = np.column_stack([np.cos(ang1), np.sin(ang1)])
            ring2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
            inits.append(np.vstack([ring1, ring2]))
        else:
            n1 = rng.choice([4, 5, 6])
            n2 = rng.choice([3, 4])
            n3 = N - n1 - n2
            if n3 < 1:
                continue
            r2 = rng.uniform(0.4, 0.7)
            r3 = rng.uniform(0.15, 0.35)
            ang1 = np.linspace(0, 2 * np.pi, n1, endpoint=False) + rng.uniform(0, 2 * np.pi)
            ang2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + rng.uniform(0, 2 * np.pi)
            ang3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + rng.uniform(0, 2 * np.pi)
            ring1 = np.column_stack([np.cos(ang1), np.sin(ang1)])
            ring2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
            ring3 = np.column_stack([r3 * np.cos(ang3), r3 * np.sin(ang3)])
            inits.append(np.vstack([ring1, ring2, ring3]))
    
    return inits

def _refine_soft_min(pts, deadline):
    """Anneal alpha for soft-min optimization."""
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    alphas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0]
    
    for alpha in alphas:
        if time.time() > deadline - 8.0:
            break
        try:
            def closure(params, a=alpha):
                return _neg_soft_score(params, a)
            
            res = minimize(
                closure,
                current_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 40, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            soft_pts = res.x.reshape(-1, 2)
            soft_score = _score(soft_pts)
            if soft_score > current_score:
                current_pts = soft_pts
                current_score = soft_score
        except Exception:
            pass
    
    return current_pts

def _refine_nelder_mead(pts, deadline, maxiter=400):
    """Nelder-Mead polish on exact score."""
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    if time.time() > deadline - 5.0:
        return current_pts
    
    try:
        res = minimize(
            _neg_score,
            current_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        nm_pts = res.x.reshape(-1, 2)
        nm_score = _score(nm_pts)
        if nm_score > current_score:
            current_pts = nm_pts
            current_score = nm_score
    except Exception:
        pass
    
    return current_pts

def _targeted_worst_triangle_opt(pts, deadline):
    """
    Structurally different refinement: identify worst triangles and use
    a local search that specifically moves points to increase the minimum
    area triangle, using a penalty-based approach.
    """
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    if time.time() > deadline - 8.0:
        return current_pts
    
    # Find worst triangles
    areas = _triangle_areas(current_pts)
    min_area = np.min(areas)
    threshold = min_area * 1.03
    
    # Get indices of worst triangles
    worst_mask = areas <= threshold
    worst_indices = np.where(worst_mask)[0]
    
    # Limit to top 8 worst
    if len(worst_indices) > 8:
        worst_sorted = worst_indices[np.argsort(areas[worst_indices])[:8]]
    else:
        worst_sorted = worst_indices
    
    rng = np.random.RandomState(42)
    step = 0.03
    
    for _ in range(60):
        if time.time() > deadline - 4.0:
            break
        
        improved = False
        for wi in worst_sorted:
            i, j, k = _TRIPLES[wi]
            
            # Try moving each of the 3 points in several directions
            for pt_idx in [i, j, k]:
                # Direction: away from centroid of the triangle
                centroid = (current_pts[i] + current_pts[j] + current_pts[k]) / 3.0
                direction = current_pts[pt_idx] - centroid
                norm = np.linalg.norm(direction)
                if norm < 1e-9:
                    direction = np.array([1.0, 0.0])
                else:
                    direction = direction / norm
                
                # Try several step sizes and directions
                for s in [step, step * 0.5, -step * 0.3]:
                    trial_pts = current_pts.copy()
                    trial_pts[pt_idx] = current_pts[pt_idx] + s * direction
                    
                    # Also try perpendicular
                    perp = np.array([-direction[1], direction[0]])
                    trial_pts2 = current_pts.copy()
                    trial_pts2[pt_idx] = current_pts[pt_idx] + s * perp
                    
                    for tp in [trial_pts, trial_pts2]:
                        s_score = _score(tp)
                        if s_score > current_score:
                            current_pts = tp
                            current_score = s_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                # Recompute worst triangles
                areas = _triangle_areas(current_pts)
                min_area = np.min(areas)
                threshold = min_area * 1.03
                worst_mask = areas <= threshold
                worst_indices = np.where(worst_mask)[0]
                if len(worst_indices) > 8:
                    worst_sorted = worst_indices[np.argsort(areas[worst_indices])[:8]]
                else:
                    worst_sorted = worst_indices
                break
        
        if not improved:
            step *= 0.7
            if step < 1e-5:
                break
    
    return current_pts

def _symmetric_optimize(n_hull, n_inner, deadline):
    """
    Optimize a symmetric two-ring configuration by optimizing the 4 parameters:
    r_hull, r_inner, rot_hull, rot_inner.
    This reduces the 26D problem to 4D, making it much easier to optimize.
    """
    if time.time() > deadline - 10.0:
        return None, 0.0
    
    def make_points(params):
        r_hull, r_inner, rot_hull, rot_inner = params
        if r_hull < 0.05 or r_inner < 0.0 or r_hull <= r_inner:
            return None
        angles_hull = np.linspace(rot_hull, rot_hull + 2 * np.pi, n_hull, endpoint=False)
        if n_inner > 0:
            angles_inner = np.linspace(rot_inner, rot_inner + 2 * np.pi, n_inner, endpoint=False)
            xs = np.concatenate([r_hull * np.cos(angles_hull), r_inner * np.cos(angles_inner)])
            ys = np.concatenate([r_hull * np.sin(angles_hull), r_inner * np.sin(angles_inner)])
        else:
            xs = r_hull * np.cos(angles_hull)
            ys = r_hull * np.sin(angles_hull)
        return np.column_stack([xs, ys])
    
    def obj_sym(params):
        pts = make_points(params)
        if pts is None:
            return 1e6
        return -_score(pts)
    
    def obj_soft_sym(params, beta=30.0):
        pts = make_points(params)
        if pts is None:
            return 1e6
        areas = _triangle_areas(pts)
        soft_min = _soft_min(areas, beta)
        hull_a = _hull_area(pts)
        if hull_a <= 1e-12:
            return 1e6
        return -(soft_min / hull_a)
    
    best_pts = None
    best_score = 0.0
    
    # Try multiple initial guesses
    r_inner_guesses = [0.2, 0.3, 0.4, 0.5, 0.6] if n_inner > 0 else [0.0]
    rot_guesses = [0.0, np.pi / max(n_hull, 1), np.pi / max(n_inner, 1)]
    
    for r_in in r_inner_guesses:
        for rot_h in rot_guesses[:2]:
            for rot_i in rot_guesses[:2]:
                if time.time() > deadline - 5.0:
                    break
                x0 = np.array([1.0, r_in, rot_h, rot_i])
                
                # Soft-min phase
                try:
                    res = minimize(
                        lambda p: obj_soft_sym(p, beta=30.0),
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    pts = make_points(res.x)
                    if pts is not None:
                        s = _score(pts)
                        if s > best_score:
                            best_score = s
                            best_pts = pts.copy()
                except Exception:
                    pass
                
                # Hard polish
                if best_pts is not None:
                    try:
                        res2 = minimize(
                            obj_sym,
                            res.x if 'res' in dir() else x0,
                            method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12}
                        )
                        pts2 = make_points(res2.x)
                        if pts2 is not None:
                            s2 = _score(pts2)
                            if s2 > best_score:
                                best_score = s2
                                best_pts = pts2.copy()
                    except Exception:
                        pass
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = 0.0
    best_pts = np.zeros((N, 2))
    
    # Generate initializations
    inits = _generate_initializations()
    
    # Evaluate all initializations
    scored_inits = []
    for init_pts in inits:
        if time.time() > deadline - 80.0:
            break
        s = _score(init_pts)
        scored_inits.append((s, init_pts))
        if s > best_score:
            best_score = s
            best_pts = init_pts.copy()
    
    # Sort by score descending
    scored_inits.sort(key=lambda x: -x[0])
    
    # Symmetric optimization for best splits
    for n_hull, n_inner in [(7, 6), (8, 5), (9, 4), (10, 3), (12, 1), (13, 0)]:
        if time.time() > deadline - 40.0:
            break
        sym_pts, sym_score = _symmetric_optimize(n_hull, n_inner, deadline)
        if sym_pts is not None and sym_score > best_score:
            best_score = sym_score
            best_pts = sym_pts.copy()
    
    # Refine top candidates from initializations
    top_k = min(6, len(scored_inits))
    for rank in range(top_k):
        if time.time() > deadline - 20.0:
            break
        
        s, pts = scored_inits[rank]
        
        # Phase 1: Soft-min annealing
        refined = _refine_soft_min(pts, deadline)
        refined_score = _score(refined)
        
        # Phase 2: Targeted worst-triangle optimization
        refined = _targeted_worst_triangle_opt(refined, deadline)
        refined_score = _score(refined)
        
        # Phase 3: Nelder-Mead polish
        refined = _refine_nelder_mead(refined, deadline)
        refined_score = _score(refined)
        
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined.copy()
    
    # If we have time, do one more round of targeted optimization on best
    if time.time() < deadline - 10.0:
        refined = _targeted_worst_triangle_opt(best_pts, deadline)
        refined_score = _score(refined)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined.copy()
    
    # Final Nelder-Mead polish
    if time.time() < deadline - 3.0:
        try:
            res_final = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            final_pts = res_final.x.reshape(-1, 2)
            final_score = _score(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_pts = final_pts.copy()
        except Exception:
            pass
    
    return best_pts
