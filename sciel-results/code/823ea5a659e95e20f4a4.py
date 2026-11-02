"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3.0)
H = S3 / 2.0  # height of unit equilateral triangle

# Precompute all C(11,3) = 165 triangle index combinations
from itertools import combinations
TRI_INDICES = list(combinations(range(N), 3))
M = len(TRI_INDICES)  # 165

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])


def bary_to_cart(bary):
    """Convert barycentric coords (u,v,w) with u+v+w=1 to Cartesian.
    We use u, v as free params; w = 1-u-v.
    Point = u*V0 + v*V1 + w*V2 = v*V1 + (1-u-v)*V2
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * H
    return np.column_stack([x, y])


def cart_to_bary(points):
    """Convert Cartesian to barycentric (u,v) where w=1-u-v."""
    x = points[:, 0]
    y = points[:, 1]
    # x = v + 0.5*(1-u-v) = 0.5 + 0.5*v - 0.5*u
    # y = H*(1-u-v)
    v = (y / H)
    # from y: 1-u-v = y/H => u = 1 - v - y/H ... wait
    # y = H*(1-u-v) => 1-u-v = y/H => u + v = 1 - y/H
    # x = v + 0.5*(1-u-v) = v + 0.5*(y/H)
    # v = x - 0.5*y/H
    v_bary = x - 0.5 * y / H
    u_bary = 1.0 - y / H - v_bary
    return np.column_stack([u_bary, v_bary])


def compute_areas_cartesian(points):
    """Compute areas of all triangles formed by point triples.
    Area of triangle with points P_i, P_j, P_k = 0.5 * |det(P_j-P_i, P_k-P_i)|
    We return the doubled areas (i.e., the determinant magnitudes) for efficiency.
    """
    n = points.shape[0]
    areas = np.zeros(M)
    for idx, (i, j, k) in enumerate(TRI_INDICES):
        dx1 = points[j, 0] - points[i, 0]
        dy1 = points[j, 1] - points[i, 1]
        dx2 = points[k, 0] - points[i, 0]
        dy2 = points[k, 1] - points[i, 1]
        det = dx1 * dy2 - dy1 * dx2
        areas[idx] = abs(det) * 0.5
    return areas


def min_area_score(points):
    """The objective: minimum triangle area."""
    areas = compute_areas_cartesian(points)
    return np.min(areas)


def smooth_min(areas, beta):
    """Log-sum-exp based smooth minimum.
    min(a) ≈ -log(sum(exp(-beta*a)))/beta
    """
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    sum_exp = np.sum(np.exp(neg_beta_areas - max_val))
    return -(max_val + np.log(sum_exp)) / beta


def negative_smooth_min(points, beta):
    """Negative of smooth min area (for minimization)."""
    areas = compute_areas_cartesian(points)
    return -smooth_min(areas, beta)


def gradient_smooth_min(points, beta):
    """Gradient of negative smooth min area w.r.t. point coordinates."""
    n = points.shape[0]
    areas = compute_areas_cartesian(points)
    
    # Soft min weights: w_i = exp(-beta * a_i) / sum(exp(-beta * a_j))
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    exp_vals = np.exp(neg_beta_areas - max_val)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # sum to 1
    
    # Gradient of -smooth_min = sum(w_i * grad(a_i))
    # grad(a_i) w.r.t. point coordinates
    # a_i = 0.5 * |det|, so grad(a_i) = 0.5 * sign(det) * grad(det)
    # For simplicity, assume det > 0 (we can handle sign)
    # Actually, let's compute gradient of area directly
    
    grad = np.zeros_like(points)
    for idx, (i, j, k) in enumerate(TRI_INDICES):
        w = weights[idx]
        if w < 1e-15:
            continue
        pi = points[i]
        pj = points[j]
        pk = points[k]
        # det = (pj-pi) x (pk-pi)
        # a = 0.5 * |det|
        # grad a / grad pi = -0.5 * sign(det) * (pk-pj) rotated 90°... 
        # Let's use: det = (pj_x-pi_x)*(pk_y-pi_y) - (pj_y-pi_y)*(pk_x-pi_x)
        # d(det)/d(pi_x) = -(pk_y-pi_y) + (pj_y-pi_y) = pj_y - pk_y
        # d(det)/d(pi_y) = (pk_x-pi_x) - (pj_x-pi_x) = pj_x - pk_x
        # d(det)/d(pj_x) = (pk_y-pi_y)
        # d(det)/d(pj_y) = -(pk_x-pi_x)
        # d(det)/d(pk_x) = -(pj_y-pi_y)
        # d(det)/d(pk_y) = (pj_x-pi_x)
        
        det = (pj[0]-pi[0])*(pk[1]-pi[1]) - (pj[1]-pi[1])*(pk[0]-pi[0])
        sign_det = np.sign(det) if abs(det) > 1e-15 else 1.0
        
        # grad of a = 0.5 * sign(det) * grad(det)
        grad[i, 0] += w * 0.5 * sign_det * (pj[1] - pk[1])
        grad[i, 1] += w * 0.5 * sign_det * (pk[0] - pj[0])
        grad[j, 0] += w * 0.5 * sign_det * (pk[1] - pi[1])
        grad[j, 1] += w * 0.5 * sign_det * (-(pk[0] - pi[0]))
        grad[k, 0] += w * 0.5 * sign_det * (-(pj[1] - pi[1]))
        grad[k, 1] += w * 0.5 * sign_det * (pj[0] - pi[0])
    
    # We want to minimize -smooth_min, so gradient is -grad(smooth_min) = -sum(w_i * grad(a_i))
    # But we defined negative_smooth_min = -smooth_min, so its gradient = -sum(w_i*grad(a_i))
    # Wait: smooth_min ≈ -log(sum(exp(-beta*a)))/beta
    # d(smooth_min)/d(x) = -1/beta * d/dx log(sum(exp(-beta*a)))
    # = -1/beta * (1/sum) * sum(exp(-beta*a_i) * (-beta) * d(a_i)/dx)
    # = sum(w_i * d(a_i)/dx) where w_i = exp(-beta*a_i)/sum
    # So gradient of smooth_min = sum(w_i * grad(a_i))
    # Gradient of negative_smooth_min = -sum(w_i * grad(a_i))
    
    return -grad


def in_triangle(points, margin=0.0):
    """Check if all points are inside the equilateral triangle (with margin)."""
    x = points[:, 0]
    y = points[:, 1]
    # Constraints: y >= 0, y <= H*(1-x) for x in [0,1]... 
    # Actually: y >= 0, x >= 0, x <= 1, y <= H*(1 - x) is not quite right.
    # The triangle has vertices (0,0), (1,0), (0.5, H).
    # Edges: bottom y=0, left edge from (0,0) to (0.5,H): y = 2*H*x, right edge from (1,0) to (0.5,H): y = 2*H*(1-x)
    # So constraints: y >= 0, y <= 2*H*x, y <= 2*H*(1-x)
    # i.e., y >= 0, x >= y/(2H), x <= 1 - y/(2H)
    
    if np.any(y < -margin):
        return False
    if np.any(x < -margin):
        return False
    if np.any(x > 1.0 + margin):
        return False
    if np.any(y > 2*H*x + margin):
        return False
    if np.any(y > 2*H*(1-x) + margin):
        return False
    return True


def nudge_search(points, n_iter=200, step_scale=0.05, rng=None):
    """Local nudge search: perturb points in the minimum triangles."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    best_points = points.copy()
    best_score = min_area_score(best_points)
    
    for it in range(n_iter):
        areas = compute_areas_cartesian(best_points)
        min_idx = np.argmin(areas)
        i, j, k = TRI_INDICES[min_idx]
        
        # Try nudging each of the three points
        improved = False
        for pidx in [i, j, k]:
            for _ in range(10):
                delta = rng.normal(0, step_scale, size=2)
                candidate = best_points.copy()
                candidate[pidx] += delta
                
                # Project back into triangle if needed
                if not in_triangle(candidate, margin=-0.01):
                    # Simple projection: clamp
                    cx, cy = candidate[pidx]
                    cy = np.clip(cy, 0, 2*H*max(0, min(1, cx)))
                    cx = np.clip(cx, cy/(2*H) if cy > 0 else 0, 1 - cy/(2*H) if cy > 0 else 1)
                    candidate[pidx] = [cx, cy]
                
                score = min_area_score(candidate)
                if score > best_score + 1e-10:
                    best_points = candidate
                    best_score = score
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try random nudges of any point
            pidx = rng.integers(0, N)
            delta = rng.normal(0, step_scale * 0.5, size=2)
            candidate = best_points.copy()
            candidate[pidx] += delta
            if in_triangle(candidate, margin=-0.005):
                score = min_area_score(candidate)
                if score > best_score + 1e-10:
                    best_points = candidate
                    best_score = score
    
    return best_points, best_score


def differential_evolution_global(points_init, n_pop=40, n_gen=100, rng=None, deadline=None):
    """Simple differential evolution for global search."""
    if rng is None:
        rng = np.random.default_rng(123)
    
    n = points_init.shape[0]
    pop = [points_init.copy()]
    for _ in range(n_pop - 1):
        # Random initialization inside triangle
        while True:
            u = rng.random(n)
            v = rng.random(n)
            if np.all(u + v < 1.0) and np.all(u > 0.01) and np.all(v > 0.01):
                break
        pts = np.column_stack([u + 0.5*v, H*v])
        pop.append(pts)
    
    best = points_init.copy()
    best_score = min_area_score(best)
    
    for gen in range(n_gen):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        for i in range(n_pop):
            # DE/rand/1
            a, b, c = [x for x in range(n_pop) if x != i]
            a, b, c = rng.choice(a, size=3, replace=False)
            F = 0.8
            trial = pop[a] + F * (pop[b] - pop[c])
            
            # Ensure inside triangle
            valid = True
            for p in range(n):
                x, y = trial[p]
                if y < 0 or x < 0 or x > 1 or y > 2*H*x or y > 2*H*(1-x):
                    valid = False
                    break
            if not valid:
                trial = pop[i]  # keep original
            
            score = min_area_score(trial)
            if score > min_area_score(pop[i]):
                pop[i] = trial
                if score > best_score:
                    best = trial.copy()
                    best_score = score
    
    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(7)
    
    # Initialize with multiple random starts
    best_points = None
    best_score = -np.inf
    
    # Start with the seed-like random initialization
    points = []
    rng_local = np.random.default_rng(7)
    while len(points) < N:
        u, v = rng_local.random(), rng_local.random()
        if u + v < 0.98 and u > 0.01 and v > 0.01:
            points.append([u + 0.5 * v, H * v])
    points = np.array(points)
    
    # Try gradient-based optimization from multiple starts
    starts = [points]
    
    # Add a few more random starts
    for seed in [42, 123, 456, 789]:
        rng_s = np.random.default_rng(seed)
        pts = []
        while len(pts) < N:
            u, v = rng_s.random(), rng_s.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                pts.append([u + 0.5 * v, H * v])
        starts.append(np.array(pts))
    
    # Also try a structured grid-like start
    # Place points roughly evenly
    grid_pts = []
    # 3x3 grid in barycentric + 2 extra
    for i in range(3):
        for j in range(3):
            u = i / 4.0
            v = j / 4.0
            if u + v <= 1.0:
                grid_pts.append([u + 0.5 * v, H * v])
    # Add 2 more
    grid_pts.append([0.25, 0.25 * H])
    grid_pts.append([0.75, 0.25 * H])
    if len(grid_pts) >= N:
        starts.append(np.array(grid_pts[:N]))
    
    for start_pts in starts:
        if time.time() > deadline - 20.0:
            break
        
        # Gradient-based optimization with smooth min
        beta = 20.0
        res = minimize(
            negative_smooth_min,
            start_pts,
            jac=lambda x: gradient_smooth_min(x, beta),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < 0:
            candidate = res.x
            if in_triangle(candidate, margin=-0.005):
                score = min_area_score(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate.copy()
        
        # Nudge search from the result
        if best_points is not None and time.time() < deadline - 15.0:
            nudged, nudged_score = nudge_search(best_points, n_iter=300, step_scale=0.03, rng=rng)
            if nudged_score > best_score:
                best_score = nudged_score
                best_points = nudged
    
    # Differential evolution for global exploration
    if time.time() < deadline - 30.0:
        de_best, de_score = differential_evolution_global(
            best_points if best_points is not None else starts[0],
            n_pop=30, n_gen=80, rng=rng, deadline=deadline
        )
        if de_score > best_score:
            best_score = de_score
            best_points = de_best
    
    # Final polish with nudge search
    if best_points is not None and time.time() < deadline - 10.0:
        polished, polished_score = nudge_search(best_points, n_iter=500, step_scale=0.01, rng=rng)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished
    
    # One more gradient pass if time allows
    if best_points is not None and time.time() < deadline - 5.0:
        beta = 50.0
        res = minimize(
            negative_smooth_min,
            best_points,
            jac=lambda x: gradient_smooth_min(x, beta),
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if res.success or res.fun < 0:
            candidate = res.x
            if in_triangle(candidate, margin=-0.005):
                score = min_area_score(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate
    
    if best_points is None:
        # Fallback: return the initial random points
        best_points = starts[0]
    
    return best_points
