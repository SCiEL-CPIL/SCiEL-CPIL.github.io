"""Heilbronn triangles: place 11 points to maximize min triangle area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize, differential_evolution

N = 11
_H = np.sqrt(3) / 2.0
_SQRT3 = np.sqrt(3)

# Precompute all C(11,3) = 165 triangle index triples
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the N points."""
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    # Area = 0.5 * | (xj-xi)(yk-yi) - (yj-yi)(xk-xi) |
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _project_to_triangle(pts):
    """Project points to stay inside the equilateral triangle."""
    pts = np.array(pts, dtype=float)
    # Simple iterative projection
    for _ in range(10):
        # y >= 0
        pts[:, 1] = np.maximum(pts[:, 1], 0)
        # x >= 0
        pts[:, 0] = np.maximum(pts[:, 0], 0)
        # x <= 1
        pts[:, 0] = np.minimum(pts[:, 0], 1.0)
        
        # y <= sqrt(3) * x  =>  y - sqrt(3)x <= 0
        viol_left = pts[:, 1] - _SQRT3 * pts[:, 0]
        mask_left = viol_left > 0
        if np.any(mask_left):
            # Project onto line y = sqrt(3)x
            # Normal is (-sqrt(3), 1). Unit normal n = (-sqrt(3), 1)/2
            # p_new = p - viol * n ? 
            # Actually, solving for intersection:
            # If we just clamp, it's not a projection.
            # Let's use a simple relaxation:
            # y_new = sqrt(3) x
            # x stays same? No, that changes distance.
            # Proper projection onto halfspace:
            # d = dot(p, n) - c. If d > 0, p_proj = p - d * n
            # Line: sqrt(3)x - y = 0. Normal n = (sqrt(3), -1). |n|=2.
            # d = (sqrt(3)x - y)/2. If d > 0, p = p - d*n.
            d = (_SQRT3 * pts[mask_left, 0] - pts[mask_left, 1]) / 2.0
            pts[mask_left, 0] -= d * _SQRT3
            pts[mask_left, 1] += d
            
        # y <= sqrt(3) * (1 - x) => y + sqrt(3)x - sqrt(3) <= 0
        viol_right = pts[:, 1] + _SQRT3 * pts[:, 0] - _SQRT3
        mask_right = viol_right > 0
        if np.any(mask_right):
            # Line: sqrt(3)x + y - sqrt(3) = 0. Normal n = (sqrt(3), 1). |n|=2.
            d = (_SQRT3 * pts[mask_right, 0] + pts[mask_right, 1] - _SQRT3) / 2.0
            pts[mask_right, 0] -= d * _SQRT3
            pts[mask_right, 1] -= d
            
    return pts

def _soft_min(values, beta):
    """Smooth approximation of the minimum using log-sum-exp."""
    min_val = np.min(values)
    shifted = values - min_val
    # Prevent overflow
    max_shifted = np.max(shifted)
    if max_shifted > 50:
        shifted = shifted - max_shifted
        min_val = min_val + max_shifted # Adjust min_val if we shifted? 
        # Actually: min(A) = min(A - m) + m.
        # logsumexp(-beta * (A - m)) = log(sum exp(-beta A + beta m))
        # = beta m + log(sum exp(-beta A))
        # -1/beta * logsumexp(-beta (A-m)) = -1/beta * (log(sum exp(-beta A)) + beta m) = -1/beta logsumexp(-beta A) - m.
        # So if we shift A by m, the result is the same.
        # Let's just stick to the stable version:
        # sm = min_val - log(sum(exp(-beta * (values - min_val)))) / beta
        # If values - min_val is large, exp(-beta * large) is 0.
        # This is safe.
    sum_exp = np.sum(np.exp(-beta * shifted))
    return min_val - np.log(sum_exp) / beta

def _penalty(points):
    """Penalty for points outside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    d1 = np.maximum(0, -y)
    viol2 = np.maximum(0, y - _SQRT3 * x)
    viol3 = np.maximum(0, y - (_SQRT3 - _SQRT3 * x))
    return np.sum(d1**2) + np.sum(viol2**2) + np.sum(viol3**2)

def _smooth_objective(x, beta=100.0):
    """Smooth surrogate: negative soft-min of triangle areas + penalty."""
    pts = x.reshape(N, 2)
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    sm = _soft_min(areas, beta)
    pen = _penalty(pts)
    
    return -(sm) + 10.0 * pen

def _true_objective(x):
    """Negative min triangle area (to be minimized)."""
    pts = x.reshape(N, 2)
    pts = _project_to_triangle(pts)
    return -_min_triangle_area(pts)

def _generate_starts(num_starts=20, seed=0):
    """Generate diverse initial configurations."""
    starts = []
    rng = np.random.RandomState(seed)
    
    # 1. Random points
    for _ in range(5):
        u, v = rng.uniform(0, 1, size=(N, 2))
        mask = u + v > 1
        while np.any(mask):
            u[mask], v[mask] = rng.uniform(0, 1, size=(np.sum(mask), 2))
            mask = u + v > 1
        x = u + 0.5 * v
        y = _H * v
        pts = np.column_stack([x, y])
        starts.append(_project_to_triangle(pts).flatten())

    # 2. Triangular Grid Perturbations
    # 3x3 grid + center + corners?
    # Let's use a 3x3 barycentric grid (9 points) + 2 extra
    grid_pts = []
    for i in range(3):
        for j in range(3):
            k = 2 - i - j
            if k >= 0:
                bary = np.array([i/2.0, j/2.0, k/2.0])
                x = bary[1] + 0.5 * bary[2]
                y = _H * bary[2]
                grid_pts.append([x, y])
    # Add center and midpoints of sides
    grid_pts.append([0.5, _H/3])
    grid_pts.append([0.25, _H/6])
    grid_pts.append([0.75, _H/6])
    grid_pts.append([0.5, _H*2/3])
    grid_pts = np.array(grid_pts[:N])
    
    for _ in range(5):
        noise = rng.normal(0, 0.05, size=(N, 2))
        pts = grid_pts + noise
        starts.append(_project_to_triangle(pts).flatten())

    # 3. Symmetric "Star" patterns
    for r in [0.2, 0.3, 0.4]:
        cx, cy = 0.5, _H/3
        angles = np.linspace(0, 2*np.pi, 6)[:-1]
        pts = [[cx, cy]]
        for a in angles:
            pts.append([cx + r*np.cos(a), cy + r*np.sin(a)])
        # Add 4 more points to make 11
        # Push some to corners/edges
        pts.append([0.1, 0.05])
        pts.append([0.9, 0.05])
        pts.append([0.5, 0.7])
        pts.append([0.3, 0.3])
        pts = np.array(pts[:N])
        noise = rng.normal(0, 0.02, size=(N, 2))
        pts = pts + noise
        starts.append(_project_to_triangle(pts).flatten())

    return starts

def _active_set_gradient_ascent(x0, max_time=10.0, step_size=0.01, decay=0.95):
    """
    Gradient ascent on the minimum area triangle.
    This is a subgradient method. We push the points to increase the area of the currently smallest triangle.
    """
    start = time.time()
    pts = x0.reshape(N, 2).copy()
    best_val = _true_objective(pts.flatten())
    best_x = pts.flatten().copy()
    
    current_step = step_size
    
    while time.time() < start + max_time:
        # Compute all areas
        pi = pts[_idx_i]
        pj = pts[_idx_j]
        pk = pts[_idx_k]
        cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
                (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
        areas = 0.5 * np.abs(cross)
        
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        # Identify active set: triangles with area < min_area + epsilon
        # To avoid noise, just use the single minimum or a small set
        active_mask = areas < (min_area + 1e-4)
        
        # Compute gradient of the sum of active areas
        # Area(i,j,k) = 0.5 * | (xj-xi)(yk-yi) - (yj-yi)(xk-xi) |
        # Let's define signed area S = 0.5 * ( (xj-xi)(yk-yi) - (yj-yi)(xk-xi) )
        # We want to increase |S|.
        # If S > 0, grad |S| = grad S. If S < 0, grad |S| = -grad S.
        # grad_xi S = 0.5 * ( -(yk-yi) - (yj-yi) * 0 ) = -0.5(yk-yi) ? 
        # Let's derive properly.
        # S = 0.5 ( xj yk - xj yi - xi yk + xi yi - yj xk + yj xi + yi xk - yi xi )
        # dS/dxi = 0.5 ( -yk + yi + yj )
        # dS/dyi = 0.5 ( -xk + xi + xj )
        # dS/dxj = 0.5 ( yk - yi )
        # dS/dyj = 0.5 ( -xk + xi )
        # dS/dxk = 0.5 ( -yj + yi )
        # dS/dyk = 0.5 ( xj - xi )
        
        grad = np.zeros_like(pts)
        
        # We only care about active triangles
        i_idx = _idx_i[active_mask]
        j_idx = _idx_j[active_mask]
        k_idx = _idx_k[active_mask]
        
        xi = pts[i_idx]
        yi = pts[i_idx, 1] # placeholder, need arrays
        # Let's just loop for simplicity and correctness, N is small
        for a in range(len(i_idx)):
            i, j, k = i_idx[a], j_idx[a], k_idx[a]
            xi, yi = pts[i]
            xj, yj = pts[j]
            xk, yk = pts[k]
            
            # Signed Area * 2
            S2 = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
            
            # Gradient of S2
            gxi = -yk + yi + yj
            gyi = -xk + xi + xj
            gxj = yk - yi
            gyj = -xk + xi
            gxk = -yj + yi
            gyk = xj - xi
            
            # We want to increase |S2|.
            # If S2 > 0, move in direction of grad S2.
            # If S2 < 0, move in direction of -grad S2.
            sign = 1.0 if S2 > 0 else -1.0
            
            grad[i] += 0.5 * sign * np.array([gxi, gyi])
            grad[j] += 0.5 * sign * np.array([gxj, gyj])
            grad[k] += 0.5 * sign * np.array([gxk, gyk])
            
        # Normalize gradient to have unit norm for stable step size
        norm = np.linalg.norm(grad)
        if norm > 1e-6:
            grad = grad / norm
            
        # Update points
        new_pts = pts + current_step * grad
        new_pts = _project_to_triangle(new_pts)
        
        new_val = _true_objective(new_pts.flatten())
        
        if new_val < best_val: # Note: objective is negative area, so smaller is better
            best_x = new_pts.flatten().copy()
            best_val = new_val
            current_step *= 1.05 # Increase step if improving
        else:
            current_step *= decay # Decrease step if not improving
            
        # Cap step size
        current_step = np.clip(current_step, 1e-4, 0.1)
        
        pts = new_pts
        
    return best_x, best_val

def construct():
    deadline = time.time() + 145.0
    
    # Phase 1: Multi-start with Smooth Surrogate (L-BFGS-B)
    starts = _generate_starts(20, seed=42)
    
    best_x = None
    best_val = np.inf
    
    for i, x0 in enumerate(starts):
        if time.time() > deadline - 60:
            break
            
        # Use different betas for diversity
        beta = 50.0 + 10.0 * (i % 5)
        
        bounds = [(0, 1), (0, _H)] * N
        
        try:
            res = minimize(
                lambda x: _smooth_objective(x, beta),
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            if res.fun < np.inf:
                cand_x = res.x
                cand_val = _true_objective(cand_x)
                if cand_val < best_val:
                    best_val = cand_val
                    best_x = cand_x
        except Exception:
            pass

    if best_x is None:
        # Fallback
        best_x = starts[0]
        best_val = _true_objective(best_x)

    # Phase 2: Active Set Gradient Ascent
    if time.time() < deadline - 30:
        x, val = _active_set_gradient_ascent(best_x, max_time=20.0)
        if val < best_val:
            best_x = x
            best_val = val

    # Phase 3: SLSQP Polish
    if time.time() < deadline - 10:
        bounds = [(0, 1), (0, _H)] * N
        try:
            res = minimize(
                _true_objective,
                best_x,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            )
            if res.success or res.fun < best_val:
                best_x = res.x
                best_val = res.fun
        except Exception:
            pass

    # Final check and projection
    pts = _project_to_triangle(best_x.reshape(N, 2))
    return pts

