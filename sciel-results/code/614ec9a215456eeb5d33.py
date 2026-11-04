import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import time
from itertools import combinations

N = 13
TRI_INDICES = np.array(list(combinations(range(N), 3)))
NUM_TRI = len(TRI_INDICES)

def _all_triangle_areas(pts):
    """Vectorized computation of all triangle areas."""
    i = TRI_INDICES[:, 0]
    j = TRI_INDICES[:, 1]
    k = TRI_INDICES[:, 2]
    pi = pts[i]
    pj = pts[j]
    pk = pts[k]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pk[:, 0] - pi[:, 0]) * (pj[:, 1] - pi[:, 1])
    return 0.5 * np.abs(cross)

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Calculate the actual score: min triangle area / hull area."""
    areas = _all_triangle_areas(pts)
    min_area = np.min(areas)
    hull_area = _hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area

def _soft_min_objective(x, beta):
    """
    Objective for optimization: minimize -soft_min(areas) / hull_area.
    x: (2*N,) flattened points
    """
    pts = x.reshape(N, 2)
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e6
    
    areas = _all_triangle_areas(pts)
    
    # Soft min via log-sum-exp
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    lse = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -lse / beta
    
    return -soft_min / hull_area

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _concentric_rings(n):
    """Generate points on concentric rings for structured initialization."""
    # 1 center + 6 on inner ring + 6 on outer ring
    pts = np.zeros((n, 2))
    # Inner ring: 6 points at radius 0.5
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts[1 + i] = [0.5 * np.cos(angle), 0.5 * np.sin(angle)]
    # Outer ring: 6 points at radius 1.0, offset by 30 degrees
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi / 6
        pts[7 + i] = [1.0 * np.cos(angle), 1.0 * np.sin(angle)]
    return pts

def _halton_sequence(n):
    """Generate Halton sequence in [0,1]^2."""
    def halton(i, base):
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    points = []
    for i in range(1, n + 1):
        points.append([halton(i, 2), halton(i, 3)])
    return np.array(points) * 2 - 1  # Map to [-1, 1]

def _random_points(n, seed=None):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = np.sqrt(rng.uniform(0, 1, n))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _local_search_angular(pts, max_iters=200, tol=1e-10):
    """
    Local search in angular space for points on/near convex hull.
    This is a simplified approach: perturb each point's angle and radius,
    accept if score improves.
    """
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    # Convert to polar
    centers = np.mean(current_pts, axis=0)
    dx = current_pts[:, 0] - centers[0]
    dy = current_pts[:, 1] - centers[1]
    radii = np.sqrt(dx**2 + dy**2)
    angles = np.arctan2(dy, dx)
    
    step_size = 0.01
    
    for it in range(max_iters):
        improved = False
        for i in range(N):
            for dim in range(2):
                for direction in [-1, 1]:
                    trial_angles = angles.copy()
                    trial_radii = radii.copy()
                    if dim == 0:
                        trial_angles[i] += direction * step_size
                    else:
                        trial_radii[i] *= (1 + direction * step_size * 0.1)
                    
                    trial_pts = np.column_stack([
                        centers[0] + trial_radii * np.cos(trial_angles),
                        centers[1] + trial_radii * np.sin(trial_angles)
                    ])
                    
                    s = _score(trial_pts)
                    if s > current_score + tol:
                        current_pts = trial_pts
                        current_score = s
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            step_size *= 0.9
            if step_size < 1e-6:
                break
    
    return current_pts, current_score

def _optimize_from_params(x0, beta_schedule=None):
    """
    Optimize from initial point x0 using L-BFGS-B with annealed beta.
    """
    if beta_schedule is None:
        beta_schedule = [5, 20, 50, 100, 200, 500]
    
    current_x = x0.copy()
    best_x = x0.copy()
    best_score = _score(x0.reshape(N, 2))
    
    for beta in beta_schedule:
        res = minimize(
            _soft_min_objective,
            current_x,
            args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if res.fun < 1e5:
            trial_pts = res.x.reshape(N, 2)
            s = _score(trial_pts)
            if s > best_score:
                best_score = s
                best_x = res.x.copy()
                current_x = res.x.copy()
    
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_pts = None
    best_score = -np.inf
    
    # Generate diverse initial candidates
    candidates = []
    candidates.append(_regular_polygon(N))
    candidates.append(_concentric_rings(N))
    candidates.append(_halton_sequence(N))
    candidates.append(_random_points(N, seed=42))
    candidates.append(_random_points(N, seed=123))
    candidates.append(_random_points(N, seed=456))
    candidates.append(_random_points(N, seed=789))
    candidates.append(_random_points(N, seed=1011))
    
    # Evaluate and keep the best few
    scored_candidates = []
    for cand in candidates:
        if time.time() > deadline:
            break
        try:
            s = _score(cand)
            scored_candidates.append((s, cand))
            if s > best_score:
                best_score = s
                best_pts = cand.copy()
        except Exception:
            pass
    
    # Sort by score descending
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Phase 1: Local angular search from best candidates
    if time.time() < deadline - 40:
        for s, cand in scored_candidates[:4]:
            if time.time() > deadline - 40:
                break
            try:
                refined, refined_score = _local_search_angular(cand, max_iters=100)
                if refined_score > best_score:
                    best_score = refined_score
                    best_pts = refined.copy()
            except Exception:
                pass
    
    # Phase 2: Differential Evolution (Global Search)
    if time.time() < deadline - 30:
        try:
            bounds = [(-2.0, 2.0)] * (2 * N)
            de_result = differential_evolution(
                _soft_min_objective,
                bounds,
                args=(10.0,),
                maxiter=30,
                popsize=12,
                tol=1e-8,
                seed=42,
                polish=False
            )
            
            if de_result.success:
                pts_de = de_result.x.reshape(N, 2)
                s = _score(pts_de)
                if s > best_score:
                    best_score = s
                    best_pts = pts_de.copy()
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B refinement from multiple starting points
    if best_pts is not None:
        start_points = [best_pts.flatten()]
        # Add top scored candidates as additional starts
        for s, cand in scored_candidates[:2]:
            if time.time() > deadline - 10:
                break
            start_points.append(cand.flatten())
        
        for x0 in start_points:
            if time.time() > deadline - 10:
                break
            try:
                optimized, opt_score = _optimize_from_params(x0)
                if opt_score > best_score:
                    best_score = opt_score
                    best_pts = optimized.reshape(N, 2).copy()
            except Exception:
                pass
        
        # Final high-beta polish
        if time.time() < deadline - 5:
            try:
                x0 = best_pts.flatten()
                res = minimize(
                    _soft_min_objective,
                    x0,
                    args=(1000,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.fun < 1e5:
                    new_pts = res.x.reshape(N, 2)
                    s = _score(new_pts)
                    if s > best_score:
                        best_score = s
                        best_pts = new_pts.copy()
            except Exception:
                pass
    
    # Fallback
    if best_pts is None:
        best_pts = _regular_polygon(N)
    
    return best_pts
