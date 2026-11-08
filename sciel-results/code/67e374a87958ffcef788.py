"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools
from math import comb

N = 14

# EVOLVE-BLOCK-START
def _triangle_areas_vec(points):
    """Compute all triangle areas vectorized. Returns array of shape (C(N,3),)."""
    n = points.shape[0]
    # Build index arrays
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    return areas, i_idx, j_idx, k_idx


def _convex_hull_area(points):
    """Compute convex hull area using Andrew's monotone chain."""
    pts = np.array(points, dtype=float)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    n = len(pts)
    if n < 3:
        return 0.0
    lower = []
    for p in pts:
        while len(lower) >= 2:
            x1, y1 = lower[-1]
            x2, y2 = p
            x0, y0 = lower[-2]
            cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            if cross <= 0:
                lower.pop()
            else:
                break
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2:
            x1, y1 = upper[-1]
            x2, y2 = p
            x0, y0 = upper[-2]
            cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            if cross <= 0:
                upper.pop()
            else:
                break
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    x = hull[:, 0]
    y = hull[:, 1]
    area = 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
    return area


def _score(points):
    """Score = min triangle area / convex hull area."""
    areas, _, _ = _triangle_areas_vec(points)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    return min_area / hull_area


def _soft_min_neg_score(x, beta=50.0, hull_ref=None):
    """
    Smooth surrogate: minimize -soft_min(triangle_areas) / hull_area.
    If hull_ref is provided, use it instead of computing hull (faster, smoother).
    """
    points = x.reshape(N, 2)
    areas, _, _ = _triangle_areas_vec(points)
    
    # Soft-min
    max_a = np.max(areas)
    shifted = areas - max_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = max_a - (1.0 / beta) * log_sum
    
    if hull_ref is not None:
        hull_area = hull_ref
    else:
        hull_area = _convex_hull_area(points)
    
    if hull_area < 1e-10:
        return 1e10
    
    return -soft_min / hull_area


def _grad_soft_min_neg(x, beta=50.0, hull_ref=None):
    """
    Analytical gradient of the soft-min objective.
    soft_min(a) = max_a - (1/beta)*log(sum(exp(-beta*a_i)))
    d(soft_min)/da_i = -(1/beta) * exp(-beta*a_i) / sum(exp(-beta*a_j))
    Then chain rule: a_k = 0.5*|cross(pj-pi, pk-pi)|
    """
    points = x.reshape(N, 2)
    areas, i_idx, j_idx, k_idx = _triangle_areas_vec(points)
    
    # Soft-min weights
    max_a = np.max(areas)
    shifted = areas - max_a
    exp_neg = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_neg)
    weights = exp_neg / sum_exp  # softmax(-beta * shifted)
    
    # d(soft_min)/da_k = -weights[k]
    # a_k = 0.5 * |cross(v1, v2)| where v1 = pj-pi, v2 = pk-pi
    # For each triangle, compute gradient of area w.r.t. point coordinates
    
    n = N
    grad_points = np.zeros((n, 2))
    
    for t in range(len(areas)):
        i, j, k = i_idx[t], j_idx[t], k_idx[t]
        pi = points[i]
        pj = points[j]
        pk = points[k]
        
        v1 = pj - pi
        v2 = pk - pi
        cross_val = v1[0] * v2[1] - v1[1] * v2[0]
        sign = 1.0 if cross_val >= 0 else -1.0
        
        # a = 0.5 * sign * (v1[0]*v2[1] - v1[1]*v2[0])
        # d(a)/d(pi) = -0.5*sign*(v2[1], -v2[0]) - 0.5*sign*(v1[1], -v1[0])
        # Actually: a = 0.5*sign*( (pjx-pix)*(pky-piy) - (pjy-piy)*(pkx-pix) )
        # Let's expand:
        # a = 0.5*sign*( pjx*pky - pjx*piy - pix*pky + pix*piy - pjy*pkx + pjy*pix + piy*pkx - piy*pix )
        # a = 0.5*sign*( pjx*pky - pjx*piy - pix*pky + pjy*pix + piy*pkx - pjy*pkx )
        
        # d(a)/d(pix) = 0.5*sign*( -pky + pjy + piy*0 ) = 0.5*sign*(pjy - pky)
        # d(a)/d(piy) = 0.5*sign*( -pjx + pkx )
        # d(a)/d(pjx) = 0.5*sign*( pky - piy )
        # d(a)/d(pjy) = 0.5*sign*( -pkx + pix )
        # d(a)/d(pkx) = 0.5*sign*( -pjy + piy )
        # d(a)/d(pky) = 0.5*sign*( pjx - pix )
        
        w = weights[t]
        factor = -w * 0.5 * sign  # d(soft_min)/da_k * da_k/d(coord)
        
        # For point i:
        grad_points[i, 0] += factor * (pj[1] - pk[1])
        grad_points[i, 1] += factor * (pk[0] - pj[0])
        # For point j:
        grad_points[j, 0] += factor * (pk[1] - pi[1])
        grad_points[j, 1] += factor * (pi[0] - pk[0])
        # For point k:
        grad_points[k, 0] += factor * (pi[1] - pj[1])
        grad_points[k, 1] += factor * (pj[0] - pi[0])
    
    # Now we need d(objective)/d(x) where objective = -soft_min / hull_area
    # If hull_ref is constant: d(obj)/d(x) = -grad(soft_min)/hull_ref
    # grad(soft_min)/d(x) = grad_points (since soft_min = sum of a_k weighted)
    # Wait, soft_min is a function of areas, and areas are functions of points.
    # d(soft_min)/d(point_p) = sum over triangles containing p of (d(soft_min)/da_t) * (da_t/d(point_p))
    # That's what grad_points computes: d(soft_min)/d(point_p)
    
    if hull_ref is not None:
        hull_area = hull_ref
    else:
        hull_area = _convex_hull_area(points)
    
    if hull_area < 1e-10:
        return np.zeros_like(x)
    
    # objective = -soft_min / hull_area
    # d(objective)/d(x) = -grad_points / hull_area
    # But grad_points is d(soft_min)/d(point_p), and x is flattened points
    obj_grad = -grad_points / hull_area
    
    return obj_grad.flatten()


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _two_rings(n_inner=7, n_outer=7, r_inner=0.5, r_outer=1.0, offset=0.0):
    """Two concentric rings of points."""
    angles_in = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + offset
    angles_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + offset + np.pi / n_outer
    pts_in = np.column_stack([r_inner * np.cos(angles_in), r_inner * np.sin(angles_in)])
    pts_out = np.column_stack([r_outer * np.cos(angles_out), r_outer * np.sin(angles_out)])
    return np.vstack([pts_in, pts_out])


def _hexagonal_lattice(n=14):
    """Points arranged in hexagonal lattice pattern."""
    # Hexagonal lattice with spacing
    dx = 1.0
    dy = dx * np.sqrt(3) / 2
    points = []
    for row in range(4):
        for col in range(4):
            x = col * dx + (row % 2) * dx / 2
            y = row * dy
            points.append([x, y])
    points = np.array(points[:n])
    # Center and scale
    points -= points.mean(axis=0)
    max_r = np.max(np.sqrt(points[:, 0]**2 + points[:, 1]**2))
    points /= max_r
    return points


def _jittered_polygon(n, radius=1.0, jitter=0.05, seed=42):
    """Regular polygon with small random jitter."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(-jitter, jitter, n)
    radii = radius + rng.uniform(-jitter, jitter, n)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _random_points(n, seed=0):
    """Random points in unit square."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 1, (n, 2))


def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    
    best_score = 0.0
    best_points = None
    
    # Initial configurations to try
    initial_configs = []
    
    # 1. Regular polygon
    initial_configs.append(_regular_polygon(N, radius=1.0))
    
    # 2. Two rings with different parameters
    initial_configs.append(_two_rings(7, 7, 0.4, 1.0, 0.0))
    initial_configs.append(_two_rings(6, 8, 0.5, 1.0, 0.0))
    initial_configs.append(_two_rings(5, 9, 0.45, 1.0, 0.0))
    initial_configs.append(_two_rings(7, 7, 0.5, 1.0, np.pi/7))
    initial_configs.append(_two_rings(6, 8, 0.4, 1.0, np.pi/8))
    
    # 3. Hexagonal lattice
    initial_configs.append(_hexagonal_lattice(N))
    
    # 4. Jittered polygons
    for seed in range(5):
        initial_configs.append(_jittered_polygon(N, radius=1.0, jitter=0.03, seed=seed))
    
    # 5. Random points
    for seed in range(5):
        initial_configs.append(_random_points(N, seed=seed))
    
    # First pass: evaluate all initial configs to find good starting points
    scored_inits = []
    for init_pts in initial_configs:
        s = _score(init_pts)
        scored_inits.append((s, init_pts))
        if s > best_score:
            best_score = s
            best_points = init_pts.copy()
    
    # Sort by score descending, take top candidates
    scored_inits.sort(key=lambda x: -x[0])
    top_inits = [pts for _, pts in scored_inits[:8]]
    
    # Optimize each top initial config
    for init_idx, init_pts in enumerate(top_inits):
        if time.time() > deadline - 5:
            break
        
        x0 = init_pts.flatten()
        
        # Use hull_ref from initial config for smoother optimization
        hull_ref = _convex_hull_area(init_pts)
        
        # Try L-BFGS-B with analytical gradient and fixed hull reference
        try:
            result = minimize(
                _soft_min_neg_score,
                x0,
                method='L-BFGS-B',
                jac=_grad_soft_min_neg,
                args=(30.0, hull_ref),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            opt_pts = result.x.reshape(N, 2)
            s = _score(opt_pts)
            if s > best_score:
                best_score = s
                best_points = opt_pts.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 5:
            break
        
        # Second pass: recompute hull and optimize with updated hull
        if best_points is not None:
            hull_ref2 = _convex_hull_area(best_points)
            x_best = best_points.flatten()
            try:
                result = minimize(
                    _soft_min_neg_score,
                    x_best,
                    method='L-BFGS-B',
                    jac=_grad_soft_min_neg,
                    args=(50.0, hull_ref2),
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                opt_pts = result.x.reshape(N, 2)
                s = _score(opt_pts)
                if s > best_score:
                    best_score = s
                    best_points = opt_pts.copy()
            except Exception:
                pass
    
    # Final polish: high-precision optimization from best point
    if best_points is not None and time.time() < deadline - 10:
        hull_ref_final = _convex_hull_area(best_points)
        x_best = best_points.flatten()
        
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 5:
                break
            try:
                result = minimize(
                    _soft_min_neg_score,
                    x_best,
                    method='L-BFGS-B',
                    jac=_grad_soft_min_neg,
                    args=(beta, hull_ref_final),
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                opt_pts = result.x.reshape(N, 2)
                s = _score(opt_pts)
                if s > best_score:
                    best_score = s
                    best_points = opt_pts.copy()
                    hull_ref_final = _convex_hull_area(opt_pts)
                    x_best = opt_pts.flatten()
            except Exception:
                pass
    
    if best_points is None:
        best_points = _regular_polygon(N, radius=1.0)
    
    return best_points
# EVOLVE-BLOCK-END
