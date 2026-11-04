"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

def _min_triangle_area(pts):
    """Compute minimum triangle area for n points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1.0

def _score(pts):
    """Score = min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 1e-15:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_a

def _all_triangle_areas(pts):
    """Compute all triangle areas as a flat array."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                areas.append(area)
    return np.array(areas)

def _soft_min_score(pts, beta):
    """
    Smooth approximation of min triangle area using log-sum-exp.
    soft_min(x) = -log(sum(exp(-x/beta))) / beta
    As beta -> 0, this approaches min(x).
    """
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    # Use log-sum-exp trick for numerical stability
    scaled = -areas / beta
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum_exp * beta
    return soft_min

def _neg_soft_min_score(x, beta):
    """Objective to minimize: negative soft-min score."""
    pts = x.reshape((-1, 2))
    return -_soft_min_score(pts, beta)

def _grad_neg_soft_min(x, beta):
    """
    Gradient of negative soft-min score with respect to point coordinates.
    
    soft_min = -beta * log(sum(exp(-A_i/beta)))
    d(soft_min)/dA_i = -exp(-A_i/beta) / sum(exp(-A_j/beta))
    d(-soft_min)/dA_i = exp(-A_i/beta) / sum(exp(-A_j/beta))
    
    Then dA_i/dx_j for each triangle area.
    """
    pts = x.reshape((-1, 2))
    n = len(pts)
    
    # Compute all triangle areas and their gradients
    areas = np.zeros(1)
    grads = []
    
    tri_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                areas = np.append(areas, area)
                
                # Gradient of area with respect to coordinates
                # area = 0.5 * |cross|
                # For simplicity, use the signed area gradient and take abs later
                # Actually, let's compute gradient of the unsigned area directly
                
                # d(area)/d(ax) = -0.5 * (cy - ay)
                # d(area)/d(ay) = 0.5 * (cx - ax)
                # d(area)/d(bx) = 0.5 * (cy - ay)
                # d(area)/d(by) = -0.5 * (cx - ax)
                # d(area)/d(cx) = 0.5 * (by - ay)
                # d(area)/d(cy) = -0.5 * (bx - ax)
                
                # But we need gradient of |area|, so multiply by sign(area)
                # Since area >= 0 always (we take abs), the gradient is:
                # Actually, the absolute value makes this non-differentiable when area=0
                # For now, assume area > 0 and use the gradient of the signed area * sign
                
                cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
                sign = 1.0 if cross >= 0 else -1.0
                
                g_ax = -0.5 * sign * (cy - ay)
                g_ay = 0.5 * sign * (cx - ax)
                g_bx = 0.5 * sign * (cy - ay)
                g_by = -0.5 * sign * (cx - ax)
                g_cx = 0.5 * sign * (by - ay)
                g_cy = -0.5 * sign * (bx - ax)
                
                grad_vec = np.zeros(2 * n)
                grad_vec[2*i] = g_ax
                grad_vec[2*i+1] = g_ay
                grad_vec[2*j] = g_bx
                grad_vec[2*j+1] = g_by
                grad_vec[2*k] = g_cx
                grad_vec[2*k+1] = g_cy
                grads.append(grad_vec)
    
    areas = areas[1:]  # Remove the initial placeholder
    grads = np.array(grads)
    
    # Compute soft-min weights
    scaled = -areas / beta
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp  # These are d(-soft_min)/dA_i
    
    # Total gradient = sum over i of weights[i] * grads[i]
    total_grad = np.sum(weights[:, np.newaxis] * grads, axis=0)
    
    return total_grad

def _regular_polygon(n, radius=0.5, center=(0.5, 0.5)):
    """Generate points on a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack((x, y))

def _center_ring(n, inner_radius=0.15, outer_radius=0.5, center=(0.5, 0.5)):
    """Generate points: one at center, rest on a ring."""
    center_pt = np.array([center])
    ring_n = n - 1
    angles = np.linspace(0, 2 * np.pi, ring_n, endpoint=False)
    radius = (inner_radius + outer_radius) / 2
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    ring_pts = np.column_stack((x, y))
    return np.vstack([center_pt, ring_pts])

def _halton_sequence(n, base):
    """Generate Halton sequence in [0, 1)."""
    result = []
    f = 1.0
    r = 0.0
    for i in range(1, n + 1):
        f = 1.0 / base
        a = i
        while a > 0:
            r += f * (a % base)
            a //= base
            f /= base
        result.append(r)
    return result

def _halton_points(n, center=(0.5, 0.5), radius=0.45):
    """Generate points using Halton sequence mapped to a disk."""
    x_seq = _halton_sequence(n, 2)
    y_seq = _halton_sequence(n, 3)
    # Map to disk using polar coordinates
    theta = 2 * np.pi * np.array(x_seq)
    r = radius * np.sqrt(np.array(y_seq))
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack((x, y))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Initial guesses
    best_pts = _regular_polygon(N)
    best_score = _score(best_pts)
    
    starts = [
        _regular_polygon(N, radius=0.5, center=(0.5, 0.5)),
        _regular_polygon(N, radius=0.45, center=(0.5, 0.5)),
        _regular_polygon(N, radius=0.55, center=(0.5, 0.5)),
        _center_ring(N, inner_radius=0.1, outer_radius=0.5),
        _center_ring(N, inner_radius=0.2, outer_radius=0.5),
        _halton_points(N),
    ]
    
    for start in starts:
        if time.time() > deadline - 10:
            break
        s = _score(start)
        if s > best_score:
            best_score = s
            best_pts = start.copy()
    
    # L-BFGS-B optimization with soft-min surrogate
    bounds = [(0.0, 1.0)] * (N * 2)
    
    # Annealing schedule: start with large beta (smooth), decrease to small beta (sharp)
    beta_schedule = [100, 50, 25, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    x0 = best_pts.flatten()
    
    for beta in beta_schedule:
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(
                _neg_soft_min_score,
                x0,
                args=(beta,),
                jac=_grad_neg_soft_min,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 30}
            )
            
            new_pts = result.x.reshape((-1, 2))
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                x0 = new_pts.flatten()
            else:
                # Still update x0 to continue from this point
                x0 = result.x.copy()
                
        except Exception:
            pass
    
    # Final polish with very small beta and more iterations
    if time.time() < deadline - 10:
        for beta in [0.0005, 0.0002, 0.0001]:
            if time.time() > deadline - 10:
                break
            try:
                result = minimize(
                    _neg_soft_min_score,
                    x0,
                    args=(beta,),
                    jac=_grad_neg_soft_min,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 20}
                )
                new_pts = result.x.reshape((-1, 2))
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                    x0 = new_pts.flatten()
            except Exception:
                pass
    
    # Try a few random restarts if time remains
    if time.time() < deadline - 30:
        rng = np.random.RandomState(42)
        for _ in range(3):
            if time.time() > deadline - 15:
                break
            x0 = rng.uniform(0, 1, size=N * 2)
            # Quick optimization from random start
            for beta in [10, 1, 0.1, 0.01]:
                if time.time() > deadline - 15:
                    break
                try:
                    result = minimize(
                        _neg_soft_min_score,
                        x0,
                        args=(beta,),
                        jac=_grad_neg_soft_min,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 20}
                    )
                    new_pts = result.x.reshape((-1, 2))
                    new_score = _score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = new_pts.copy()
                        x0 = new_pts.flatten()
                    else:
                        x0 = result.x.copy()
                except Exception:
                    break
    
    return best_pts
# EVOLVE-BLOCK-END
