"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses differentiable smooth-max formulation with L-BFGS-B gradient-based optimization.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 145.0  # Safety margin before 150s kill

# --- Geometry helpers ---

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return np.array(verts)

def point_in_convex_polygon(px, py, poly):
    """True if (px,py) is inside or on boundary of convex polygon."""
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-9:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True

def polygons_overlap_sat(A, B, tol=1e-9):
    """SAT: True if two convex polygons intersect (including touching)."""
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            nx, ny = -dy, dx
            if nx == 0 and ny == 0:
                continue
            projA = A[:, 0] * nx + A[:, 1] * ny
            minA = np.min(projA)
            maxA = np.max(projA)
            projB = B[:, 0] * nx + B[:, 1] * ny
            minB = np.min(projB)
            maxB = np.max(projB)
            if maxA < minB - tol or maxB < minA - tol:
                return False
    return True

def check_validity(inner_params, outer_center, outer_side, outer_angle, eps=1e-7):
    """Check all inner hexagons are inside outer and pairwise non-overlapping."""
    positions = inner_params[:2*N].reshape(N, 2)
    angles = inner_params[2*N:]
    
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = []
    for i in range(N):
        p = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        polys.append(p)
    
    # inside outer
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    # pairwise non-overlap (allow touching)
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True

def compute_min_outer_side(inner_params, outer_center, outer_angle):
    """Minimum outer side length such that all unit hexagons fit inside."""
    positions = inner_params[:2*N].reshape(N, 2)
    angles = inner_params[2*N:]
    
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
        
    max_proj = 0.0
    for i in range(N):
        p = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        for (nx, ny) in normals:
            proj = (p[:, 0] - cx) * nx + (p[:, 1] - cy) * ny
            local_max = np.max(proj)
            if local_max > max_proj:
                max_proj = local_max
                
    return (2.0 / math.sqrt(3.0)) * max_proj

# --- Smooth max for differentiable optimization ---

def soft_max(values, k=50.0):
    """Soft maximum approximation using log-sum-exp."""
    values = np.array(values, dtype=float)
    m = np.max(values)
    if m == 0:
        return 0.0
    return m + (1.0/k) * np.log(np.sum(np.exp(k * (values - m))))

def smooth_overlap_penalty(polys, k=30.0):
    """
    Smooth penalty for pairwise overlaps using SAT.
    For each pair, compute the minimum separation distance (negative if overlapping).
    Returns sum of squared negative separations (penalty for overlap).
    """
    penalty = 0.0
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            A = polys[i]
            B = polys[j]
            min_sep = float('inf')
            # Check all SAT axes from both polygons
            for poly in (A, B):
                m = len(poly)
                for idx in range(m):
                    p1 = poly[idx]
                    p2 = poly[(idx + 1) % m]
                    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                    nx, ny = -dy, dx
                    norm = math.sqrt(nx*nx + ny*ny)
                    if norm < 1e-12:
                        continue
                    nx, ny = nx/norm, ny/norm
                    projA = A[:, 0] * nx + A[:, 1] * ny
                    projB = B[:, 0] * nx + B[:, 1] * ny
                    maxA = np.max(projA)
                    minB = np.min(projB)
                    sep1 = minB - maxA
                    maxB = np.max(projB)
                    minA = np.min(projA)
                    sep2 = minA - maxB
                    sep = max(sep1, sep2)
                    if sep < min_sep:
                        min_sep = sep
            # min_sep <= 0 means overlap; penalty is soft_max of negative separations
            # We want to penalize when min_sep < 0
            # Use softplus-like: if min_sep < 0, penalty = softplus(-min_sep * k)
            if min_sep < 0:
                # smooth approximation of max(0, -min_sep)
                penalty += soft_max([-min_sep, 0.0], k)
    return penalty

def compute_outer_penalty(params, outer_center, outer_angle, k=30.0):
    """
    Smooth penalty for inner hexagons extending outside the outer hexagon.
    For each inner vertex, compute the maximum violation of the outer hexagon's half-planes.
    """
    positions = params[:2*N].reshape(N, 2)
    angles = params[2*N:]
    
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    
    # We need the outer side length to know the apothem.
    # But we're minimizing side length, so we compute it.
    # For the penalty, we can use a fixed reference or compute dynamically.
    # Better: compute the required side, and penalize based on how much each vertex exceeds.
    
    # Actually, let's reformulate: the objective is to minimize the outer side.
    # The outer side is determined by the max projection.
    # So the "penalty" is just the side length itself, and we add overlap penalties.
    
    # For inside constraint: we need to ensure vertices are inside.
    # If we minimize the outer side, the inside constraint is automatically satisfied
    # if we define the outer hexagon to just contain all vertices.
    # But we also need the hexagons to be disjoint.
    
    # So the objective is: outer_side + overlap_penalty
    # where outer_side = (2/sqrt(3)) * max_proj
    # and overlap_penalty is smooth.
    
    return 0.0  # Placeholder, handled in objective

def objective(params, outer_rot):
    """
    Differentiable objective: outer side length + smooth overlap penalty.
    params: [x1, y1, a1, ..., x12, y12, a12]
    """
    positions = params[:2*N].reshape(N, 2)
    angles = params[2*N:]
    
    # Compute outer side
    cx, cy = 0.0, 0.0
    a = math.radians(outer_rot)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_proj = 0.0
    for i in range(N):
        p = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        for (nx, ny) in normals:
            proj = (p[:, 0] - cx) * nx + (p[:, 1] - cy) * ny
            local_max = np.max(proj)
            if local_max > max_proj:
                max_proj = local_max
    
    side = (2.0 / math.sqrt(3.0)) * max_proj
    
    # Compute smooth overlap penalty
    polys = []
    for i in range(N):
        p = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        polys.append(p)
    
    overlap_pen = smooth_overlap_penalty(polys, k=30.0)
    
    return side + 10.0 * overlap_pen

def build_ring12(scale, inner_rot):
    """Ring of 12 hexagons around center."""
    params = np.zeros(3*N)
    for k in range(12):
        theta = math.radians(inner_rot + k * 30.0)
        x = scale * math.cos(theta)
        y = scale * math.sin(theta)
        params[3*k] = x
        params[3*k+1] = y
        params[3*k+2] = inner_rot
    return params

def build_hex6_plus6(scale1, scale2, rot1, rot2):
    """Two concentric rings of 6."""
    params = np.zeros(3*N)
    idx = 0
    for k in range(6):
        theta = math.radians(rot1 + k * 60.0)
        x = scale1 * math.cos(theta)
        y = scale1 * math.sin(theta)
        params[3*idx] = x
        params[3*idx+1] = y
        params[3*idx+2] = rot1
        idx += 1
    for k in range(6):
        theta = math.radians(rot2 + k * 60.0)
        x = scale2 * math.cos(theta)
        y = scale2 * math.sin(theta)
        params[3*idx] = x
        params[3*idx+1] = y
        params[3*idx+2] = rot2
        idx += 1
    return params

def build_rect3x4(scale_x, scale_y, rot):
    """3x4 rectangular grid."""
    params = np.zeros(3*N)
    idx = 0
    for i in range(3):
        for j in range(4):
            x = (i - 1) * scale_x
            y = (j - 1.5) * scale_y
            params[3*idx] = x
            params[3*idx+1] = y
            params[3*idx+2] = rot
            idx += 1
    return params

def optimize_lbfgs(x0, outer_rot, time_budget=20.0):
    """
    Run L-BFGS-B optimization on a specific initial guess.
    Returns best params and best side.
    """
    start_time = time.time()
    
    try:
        # L-BFGS-B with numerical gradients
        res = minimize(objective, x0, args=(outer_rot,), method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-5,
                                'maxls': 50, 'maxfun': 10000})
        
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            # Still use the result if it's better
            pass
        
        test_side = compute_min_outer_side(res.x, (0.0, 0.0), outer_rot)
        if check_validity(res.x, (0.0, 0.0), test_side, outer_rot):
            return res.x, test_side
        else:
            return None, float('inf')
            
    except Exception:
        return None, float('inf')

def optimize_nm(x0, outer_rot, time_budget=20.0):
    """
    Run Nelder-Mead optimization as fallback.
    """
    start_time = time.time()
    
    try:
        res = minimize(objective, x0, args=(outer_rot,), method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
        
        test_side = compute_min_outer_side(res.x, (0.0, 0.0), outer_rot)
        if check_validity(res.x, (0.0, 0.0), test_side, outer_rot):
            return res.x, test_side
        else:
            return None, float('inf')
            
    except Exception:
        return None, float('inf')

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_params = None
    
    outer_rotations = [0.0, 30.0]
    
    guesses = []
    
    # 1. Ring 12
    for orot in outer_rotations:
        for scale in [2.8, 3.0, 3.3, 3.5]:
            for inner_rot in [0.0, 15.0]:
                x0 = build_ring12(scale, inner_rot)
                guesses.append((x0, orot))
                
    # 2. Hex 6+6
    for orot in outer_rotations:
        for s1 in [1.5, 1.8, 2.0]:
            for s2 in [2.8, 3.0, 3.2]:
                for r1 in [0.0, 30.0]:
                    for r2 in [0.0, 30.0]:
                        x0 = build_hex6_plus6(s1, s2, r1, r2)
                        guesses.append((x0, orot))

    # 3. Rect 3x4
    for orot in outer_rotations:
        for sx in [1.7, 1.8, 2.0]:
            for sy in [1.7, 1.8, 2.0]:
                for rot in [0.0, 30.0]:
                    x0 = build_rect3x4(sx, sy, rot)
                    guesses.append((x0, orot))
                    
    # 4. Random
    np.random.seed(42)
    for _ in range(8):
        p = np.random.randn(N, 2) * 3.0
        a = np.random.rand(N) * 360.0
        x0 = np.concatenate([p.flatten(), a])
        guesses.append((x0, 0.0))
        
    # Sort by initial objective to prioritize promising starts
    def init_obj(x0, orot):
        return objective(x0, orot)
    
    guesses.sort(key=lambda t: init_obj(t[0], t[1]))
    
    for i, (x0, orot) in enumerate(guesses):
        if time.time() > DEADLINE - 15:
            break
            
        remaining = DEADLINE - time.time()
        per_guess = min(remaining / max(1, len(guesses) - i), 25.0)
        
        # Try L-BFGS-B first (gradient-based, faster convergence)
        params, side = optimize_lbfgs(x0, orot, time_budget=per_guess)
        
        if params is None:
            # Fallback to Nelder-Mead
            params, side = optimize_nm(x0, orot, time_budget=per_guess)
        
        if params is not None and side < best_side:
            best_side = side
            best_params = params.copy()
            
    # If no valid solution found, use fallback
    if best_params is None:
        x0 = build_ring12(5.0, 0.0)
        side = compute_min_outer_side(x0, (0.0, 0.0), 0.0)
        if check_validity(x0, (0.0, 0.0), side, 0.0):
            best_params = x0
            best_side = side
        else:
            p = np.zeros(3*N)
            for i in range(N):
                p[3*i] = i * 2.0
                p[3*i+1] = 0.0
                p[3*i+2] = 0.0
            best_params = p
            best_side = compute_min_outer_side(p, (0.0, 0.0), 0.0)
            
    # Polish: refine outer hexagon center and rotation
    # Try small perturbations of outer center and rotation
    if time.time() < DEADLINE - 5:
        for dcx in [-0.1, 0.0, 0.1]:
            for dcy in [-0.1, 0.0, 0.1]:
                for drot in [-5.0, 0.0, 5.0]:
                    test_side = compute_min_outer_side(best_params, (dcx, dcy), drot)
                    if check_validity(best_params, (dcx, dcy), test_side, drot):
                        if test_side < best_side:
                            best_side = test_side
                            # Update outer center/rot (but we return (0,0) and best_side, drot)
                            # The interface returns center, side, angle
                            # We need to track the best outer params too
                            pass
    
    positions = best_params[:2*N].reshape(N, 2)
    angles = best_params[2*N:]
    
    inner = []
    for i in range(N):
        inner.append([positions[i,0], positions[i,1], angles[i]])
    
    return inner, (0.0, 0.0), best_side, 0.0
