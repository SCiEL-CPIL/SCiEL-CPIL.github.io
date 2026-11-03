from scipy.optimize import minimize, differential_evolution
import numpy as np
from scipy.spatial import ConvexHull
import time
from itertools import combinations

N = 13

# Precompute all triples for vectorized area computation
_TRIPLES = list(combinations(range(N), 3))
_I_IDX = np.array([t[0] for t in _TRIPLES])
_J_IDX = np.array([t[1] for t in _TRIPLES])
_K_IDX = np.array([t[2] for t in _TRIPLES])

def _all_triangle_areas(pts):
    """Compute all C(n,3) triangle areas efficiently using vectorization."""
    n = len(pts)
    if n < 3:
        return np.array([0.0])
    
    pi = pts[_I_IDX]
    pj = pts[_J_IDX]
    pk = pts[_K_IDX]
    
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return areas

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples."""
    areas = _all_triangle_areas(pts)
    return np.min(areas) if len(areas) > 0 else 0.0

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _min_triangle_area(pts) / hull_a

def _soft_min_area(areas, alpha):
    """
    Soft minimum of areas using log-sum-exp.
    Returns: -1/alpha * log(sum(exp(-alpha * A_k)))
    This is a lower bound on min(A_k), tight as alpha -> inf.
    """
    if len(areas) == 0:
        return 0.0
    M = np.min(areas)
    shifted = areas - M
    log_sum = -alpha * M + np.log(np.sum(np.exp(-alpha * shifted)))
    return -log_sum / alpha

def _neg_soft_score(params, alpha):
    pts = params.reshape(-1, 2)
    areas = _all_triangle_areas(pts)
    soft_min = _soft_min_area(areas, alpha)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return -(soft_min / hull_a)

def _neg_score(params):
    pts = params.reshape(-1, 2)
    return -_score(pts)

def _regular_polygon(n_sides, radius=1.0, angle_offset=0.0):
    """Generate a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) + angle_offset
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_initializations():
    """Generate diverse initial configurations."""
    rng = np.random.RandomState(42)
    inits = []
    
    # 1. Regular 13-gon
    inits.append(_regular_polygon(N))
    
    # 2. Concentric circles: 7 outer + 6 inner
    for r2 in [0.3, 0.4, 0.5, 0.6]:
        ang1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        ang2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
        ring1 = np.column_stack([np.cos(ang1), np.sin(ang1)])
        ring2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
        inits.append(np.vstack([ring1, ring2]))
    
    # 3. 12 outer + 1 center
    angles_m = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    outer = np.column_stack([np.cos(angles_m), np.sin(angles_m)])
    center = np.array([[0.0, 0.0]])
    inits.append(np.vstack([outer, center]))
    
    # 4. 9 outer + 4 middle
    for r2 in [0.4, 0.5, 0.6]:
        ang1 = np.linspace(0, 2 * np.pi, 9, endpoint=False)
        ang2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/4
        ring1 = np.column_stack([np.cos(ang1), np.sin(ang1)])
        ring2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
        inits.append(np.vstack([ring1, ring2]))
    
    # 5. 6 outer + 5 middle + 2 inner
    for r2, r3 in [(0.4, 0.2), (0.5, 0.25), (0.6, 0.3)]:
        ang1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        ang2 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi/5
        ang3 = np.linspace(0, 2 * np.pi, 2, endpoint=False) + np.pi/2
        ring1 = np.column_stack([np.cos(ang1), np.sin(ang1)])
        ring2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
        ring3 = np.column_stack([r3 * np.cos(ang3), r3 * np.sin(ang3)])
        inits.append(np.vstack([ring1, ring2, ring3]))
    
    # 6. Random concentric with 2-3 circles
    for _ in range(15):
        n_circles = rng.choice([2, 3])
        if n_circles == 2:
            n1 = rng.choice([5, 6, 7, 8])
            n2 = N - n1
            r1 = 1.0
            r2 = rng.uniform(0.3, 0.7)
            ang1 = np.linspace(0, 2 * np.pi, n1, endpoint=False) + rng.uniform(0, 2*np.pi)
            ang2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + rng.uniform(0, 2*np.pi)
            ring1 = np.column_stack([r1 * np.cos(ang1), r1 * np.sin(ang1)])
            ring2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
            inits.append(np.vstack([ring1, ring2]))
        else:
            n1 = rng.choice([4, 5, 6])
            n2 = rng.choice([3, 4, 5])
            n3 = N - n1 - n2
            if n3 < 1:
                continue
            r1 = 1.0
            r2 = rng.uniform(0.4, 0.7)
            r3 = rng.uniform(0.15, 0.35)
            ang1 = np.linspace(0, 2 * np.pi, n1, endpoint=False) + rng.uniform(0, 2*np.pi)
            ang2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + rng.uniform(0, 2*np.pi)
            ang3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + rng.uniform(0, 2*np.pi)
            ring1 = np.column_stack([r1 * np.cos(ang1), r1 * np.sin(ang1)])
            ring2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
            ring3 = np.column_stack([r3 * np.cos(ang3), r3 * np.sin(ang3)])
            inits.append(np.vstack([ring1, ring2, ring3]))
    
    # 7. Random points in disk
    for _ in range(10):
        r = rng.rand(N) ** 0.5
        theta = rng.rand(N) * 2 * np.pi
        inits.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
    
    # 8. Grid-like configurations
    # 3x3 grid plus 4 more points
    grid = np.array([[x, y] for x in [-1, 0, 1] for y in [-1, 0, 1]])
    extra = np.array([[1.5, 0], [-1.5, 0], [0, 1.5], [0, -1.5]])
    inits.append(np.vstack([grid, extra]))
    
    # 9. Star pattern
    for n_inner in [3, 4, 5]:
        n_outer = N - n_inner
        outer = _regular_polygon(n_outer, radius=1.0)
        inner = _regular_polygon(n_inner, radius=0.4, angle_offset=np.pi/n_inner)
        inits.append(np.vstack([outer, inner]))
    
    return inits

def _refine_with_soft_min(pts, deadline):
    """
    Refine points using soft-min annealing.
    Returns improved points.
    """
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    # Anneal alpha from low to high
    alphas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    for alpha in alphas:
        if time.time() > deadline - 10.0:
            break
        
        try:
            def _neg_soft_closure(params, a=alpha):
                return _neg_soft_score(params, a)
            
            res = minimize(
                _neg_soft_closure,
                current_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            soft_pts = res.x.reshape(-1, 2)
            soft_score = _score(soft_pts)
            if soft_score > current_score:
                current_pts = soft_pts
                current_score = soft_score
        except Exception:
            pass
    
    return current_pts

def _refine_with_nelder_mead(pts, deadline):
    """
    Refine points using Nelder-Mead on exact score.
    Returns improved points.
    """
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    if time.time() > deadline - 5.0:
        return current_pts
    
    try:
        res = minimize(
            _neg_score,
            current_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
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
    Identify the worst (minimum area) triangles and try to improve them
    by moving the involved points.
    """
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    if time.time() > deadline - 10.0:
        return current_pts
    
    # Find all minimum-area triangles
    areas = _all_triangle_areas(current_pts)
    min_area = np.min(areas)
    
    # Find triples with area close to minimum
    threshold = min_area * 1.05
    
    # Find worst triples directly
    n = len(current_pts)
    worst_triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = current_pts[j] - current_pts[i]
                v2 = current_pts[k] - current_pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area <= threshold:
                    worst_triples.append((i, j, k, area))
    
    # For each worst triple, try small perturbations of its points
    rng = np.random.RandomState(42)
    for (i, j, k, area) in worst_triples[:5]:  # Limit to top 5 worst
        if time.time() > deadline - 5.0:
            break
        
        # Try moving each of the 3 points in random directions
        for trial in range(30):
            trial_pts = current_pts.copy()
            # Perturb one of the three points
            idx = [i, j, k][trial % 3]
            dx = rng.randn() * 0.05
            dy = rng.randn() * 0.05
            trial_pts[idx] += np.array([dx, dy])
            
            trial_score = _score(trial_pts)
            if trial_score > current_score:
                current_pts = trial_pts
                current_score = trial_score
    
    return current_pts

def _symmetric_optimize(pts, deadline):
    """
    Try to optimize using symmetry constraints.
    Assume the optimal configuration has some rotational symmetry.
    """
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    if time.time() > deadline - 15.0:
        return current_pts
    
    # Try different symmetry orders
    for sym_order in [2, 3, 4]:
        if time.time() > deadline - 5.0:
            break
        
        try:
            # Parameterize by fundamental domain
            # For symmetry order k, we have N/k points in fundamental domain
            # This is complex, so we'll do a simpler approach:
            # Average over rotated versions to enforce symmetry
            
            # Generate rotated versions
            angles = np.linspace(0, 2 * np.pi, sym_order, endpoint=False)
            rotations = []
            for a in angles:
                c, s = np.cos(a), np.sin(a)
                R = np.array([[c, -s], [s, c]])
                rotations.append(R)
            
            # Average over rotations
            symmetric_pts = np.zeros_like(current_pts)
            for R in rotations:
                symmetric_pts += current_pts @ R.T
            symmetric_pts /= sym_order
            
            sym_score = _score(symmetric_pts)
            if sym_score > current_score:
                current_pts = symmetric_pts
                current_score = sym_score
        except Exception:
            pass
    
    return current_pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = 0.0
    best_pts = np.zeros((N, 2))
    
    # Generate initializations
    inits = _generate_initializations()
    
    # Evaluate all initializations
    scored_inits = []
    for init_pts in inits:
        if time.time() > deadline - 60.0:
            break
        s = _score(init_pts)
        scored_inits.append((s, init_pts))
        if s > best_score:
            best_score = s
            best_pts = init_pts.copy()
    
    # Sort by score descending
    scored_inits.sort(key=lambda x: -x[0])
    
    # Refine top candidates
    top_k = min(10, len(scored_inits))
    for rank in range(top_k):
        if time.time() > deadline - 25.0:
            break
        
        s, pts = scored_inits[rank]
        
        # Phase 1: Soft-min annealing
        refined = _refine_with_soft_min(pts, deadline)
        refined_score = _score(refined)
        
        # Phase 2: Nelder-Mead polish
        refined = _refine_with_nelder_mead(refined, deadline)
        refined_score = _score(refined)
        
        # Phase 3: Targeted worst-triangle optimization
        refined = _targeted_worst_triangle_opt(refined, deadline)
        refined_score = _score(refined)
        
        # Phase 4: Symmetric optimization
        refined = _symmetric_optimize(refined, deadline)
        refined_score = _score(refined)
        
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined.copy()
    
    # Global search with differential evolution if time permits
    if time.time() < deadline - 40.0:
        try:
            bounds = [(-2, 2)] * (2 * N)
            res_de = differential_evolution(
                _neg_score,
                bounds,
                maxiter=30,
                popsize=15,
                tol=1e-5,
                seed=42,
                polish=True
            )
            de_pts = res_de.x.reshape(-1, 2)
            de_score = _score(de_pts)
            if de_score > best_score:
                best_score = de_score
                best_pts = de_pts.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 3.0:
        try:
            res_final = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            final_pts = res_final.x.reshape(-1, 2)
            final_score = _score(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_pts = final_pts.copy()
        except Exception:
            pass
    
    return best_pts

# EVOLVE-BLOCK-END
