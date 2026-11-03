"""Hexagon packing: 11 unit hexagons in a regular hexagon, minimize outer side."""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin within 150s limit

def _hex_vertices(cx, cy, angle_deg=0.0):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _hex_support(cx, cy, angle_deg, nx, ny):
    """Support function of unit hexagon at (cx,cy) rotated by angle_deg in direction (nx,ny)."""
    angle_rad = math.radians(angle_deg)
    max_proj = -float('inf')
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        vx = cx + math.cos(a)
        vy = cy + math.sin(a)
        proj = vx * nx + vy * ny
        if proj > max_proj:
            max_proj = proj
    return max_proj

def _hex_clearance(h1, h2):
    """Minimum clearance between two unit hexagons using separating axis theorem.
    Returns positive if separated, negative if overlapping."""
    x1, y1, a1 = h1
    x2, y2, a2 = h2
    
    normals = []
    for ang in [a1, a2]:
        for k in range(6):
            theta = math.radians(ang + 30 + 60 * k)
            normals.append((math.cos(theta), math.sin(theta)))
    
    min_clearance = float('inf')
    for nx, ny in normals:
        sup1 = _hex_support(x1, y1, a1, nx, ny)
        sup1_neg = _hex_support(x1, y1, a1, -nx, -ny)
        sup2 = _hex_support(x2, y2, a2, nx, ny)
        sup2_neg = _hex_support(x2, y2, a2, -nx, -ny)
        
        gap1 = sup2 - sup1_neg
        gap2 = sup1 - sup2_neg
        clearance_axis = min(gap1, gap2)
        
        if clearance_axis < min_clearance:
            min_clearance = clearance_axis
    
    return min_clearance

def _check_disjoint(inner_hexes, min_gap=1e-6):
    """Check all pairs are disjoint with at least min_gap separation."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            clr = _hex_clearance(inner_hexes[i], inner_hexes[j])
            if clr < min_gap:
                return False
    return True

def _min_enclosing_hex_lp(inner_hexes, angle_deg):
    """For fixed outer hex angle, find optimal center and side via LP.
    Returns (cx, cy, side, angle_deg)."""
    all_verts = []
    for ix, iy, iang in inner_hexes:
        all_verts.extend(_hex_vertices(ix, iy, iang))
    
    n_pts = len(all_verts)
    if n_pts == 0:
        return 0.0, 0.0, 0.0, angle_deg
    
    A_ub = []
    b_ub = []
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_verts:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-vx * nx - vy * ny)
    
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    c = np.array([0.0, 0.0, 1.0])
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    
    if result.success:
        cx, cy, r = result.x[0], result.x[1], result.x[2]
    else:
        cx = sum(v[0] for v in all_verts) / n_pts
        cy = sum(v[1] for v in all_verts) / n_pts
        r = 0.0
        for k in range(6):
            a = math.radians(angle_deg + 30 + 60 * k)
            nx, ny = math.cos(a), math.sin(a)
            for vx, vy in all_verts:
                proj = (vx - cx) * nx + (vy - cy) * ny
                r = max(r, proj)
    
    side = r / (math.sqrt(3) / 2)
    return cx, cy, side, angle_deg

def _min_enclosing_hex(inner_hexes):
    """Find minimum enclosing regular hexagon over all angles."""
    best_side = float('inf')
    best_cx, best_cy, best_ang = 0.0, 0.0, 0.0
    
    # Coarse sweep
    for ang in np.arange(0, 60, 2.0):
        cx, cy, side, _ = _min_enclosing_hex_lp(inner_hexes, float(ang))
        if side < best_side:
            best_side, best_cx, best_cy, best_ang = side, cx, cy, float(ang)
    
    # Fine sweep
    for ang in np.arange(best_ang - 4, best_ang + 4, 0.5):
        if ang < 0 or ang >= 60:
            continue
        cx, cy, side, _ = _min_enclosing_hex_lp(inner_hexes, float(ang))
        if side < best_side:
            best_side, best_cx, best_cy, best_ang = side, cx, cy, float(ang)
    
    # Finer sweep
    for ang in np.arange(best_ang - 1, best_ang + 1, 0.1):
        if ang < 0 or ang >= 60:
            continue
        cx, cy, side, _ = _min_enclosing_hex_lp(inner_hexes, float(ang))
        if side < best_side:
            best_side, best_cx, best_cy, best_ang = side, cx, cy, float(ang)
    
    return best_cx, best_cy, best_side, best_ang

def _hexagonal_lattice_placement(n_hex, gap_factor=1.05):
    """Place hexagons in a compact hexagonal lattice pattern with a gap to ensure disjointness.
    The gap_factor scales the lattice spacing to create a small separation between hexagons."""
    # Hexagonal lattice vectors for unit hexagons (side=1, distance between centers = sqrt(3))
    # Multiply by gap_factor > 1 to ensure they don't touch/overlap
    d = math.sqrt(3) * gap_factor  # distance between adjacent hex centers
    u = np.array([d, 0.0])
    v = np.array([d * 0.5, d * math.sqrt(3) / 2])
    
    # Generate candidate positions in hexagonal lattice
    # Expand range slightly to ensure we get enough candidates if gap is large
    candidates = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            pos = i * u + j * v
            dist_sq = pos[0]**2 + pos[1]**2
            candidates.append((dist_sq, pos[0], pos[1]))
    
    candidates.sort(key=lambda t: t[0])
    
    positions = []
    for _, x, y in candidates:
        positions.append((x, y))
        if len(positions) == n_hex:
            break
    
    # All hexagons at 30 degrees (aligned with lattice)
    return [[x, y, 30.0] for x, y in positions]

def _optimize_placement(inner_hexes, time_budget=30.0):
    """Optimize placement to minimize enclosing hexagon side length.
    Uses Nelder-Mead on all positions simultaneously."""
    
    def flatten(hexes):
        params = []
        for x, y, a in hexes:
            params.extend([x, y, math.radians(a)])
        return np.array(params)
    
    def unflatten(params):
        hexes = []
        for i in range(len(params) // 3):
            hexes.append([params[3*i], params[3*i+1], math.degrees(params[3*i+2])])
        return hexes
    
    def objective(params):
        hexes = unflatten(params)
        # Check disjointness - if violated, return large penalty
        n = len(hexes)
        min_clr = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                clr = _hex_clearance(hexes[i], hexes[j])
                if clr < min_clr:
                    min_clr = clr
        
        # Enforce a minimum gap of 0.01 to ensure strict disjointness
        if min_clr < 0.01:
            # Penalty is quadratic to strongly discourage overlap/touching
            violation = max(0.0, 0.01 - min_clr)
            return 1000.0 + violation * 1000
        
        # Compute enclosing hexagon side
        all_verts = []
        for ix, iy, iang in hexes:
            all_verts.extend(_hex_vertices(ix, iy, iang))
        
        # Try a few outer angles
        best_side = float('inf')
        for outer_ang in [0.0, 30.0]:
            # Center at centroid
            cx = sum(v[0] for v in all_verts) / len(all_verts)
            cy = sum(v[1] for v in all_verts) / len(all_verts)
            r = 0.0
            for k in range(6):
                a = math.radians(outer_ang + 30 + 60 * k)
                nx, ny = math.cos(a), math.sin(a)
                for vx, vy in all_verts:
                    proj = (vx - cx) * nx + (vy - cy) * ny
                    r = max(r, proj)
            side = r / (math.sqrt(3) / 2)
            best_side = min(best_side, side)
        
        return best_side
    
    x0 = flatten(inner_hexes)
    t_start = time.time()
    
    # Multiple restarts with perturbation
    best_params = x0.copy()
    best_obj = objective(x0)
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > _DEADLINE - 20.0 or time.time() - t_start > time_budget:
            break
        
        if restart == 0:
            x_init = x0
        else:
            # Perturb
            scale = 0.3 * (restart + 1)
            x_init = x0 + np.random.normal(0, scale, size=x0.shape)
        
        try:
            result = minimize(objective, x_init, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 0.01, 'fatol': 0.01,
                                   'adaptive': True})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    if best_obj < 1000:  # valid solution
        return unflatten(best_params)
    return inner_hexes

def _refine_with_softmin(inner_hexes, time_budget=15.0):
    """Refine using a smooth surrogate: minimize max enclosing side with soft constraints."""
    t_start = time.time()
    
    def flatten(hexes):
        params = []
        for x, y, a in hexes:
            params.extend([x, y, math.radians(a)])
        return np.array(params)
    
    def unflatten(params):
        hexes = []
        for i in range(len(params) // 3):
            hexes.append([params[3*i], params[3*i+1], math.degrees(params[3*i+2])])
        return hexes
    
    def soft_objective(params):
        hexes = unflatten(params)
        n = len(hexes)
        
        # Soft minimum clearance (log-sum-exp style)
        clearances = []
        for i in range(n):
            for j in range(i + 1, n):
                clr = _hex_clearance(hexes[i], hexes[j])
                clearances.append(clr)
        
        if not clearances:
            return 0.0
        
        # Soft min: -log-sum-exp(-c/k) * k
        k = 2.0
        neg_clears = np.array([-c / k for c in clearances])
        max_nc = np.max(neg_clears)
        soft_min = -k * (max_nc + math.log(np.sum(np.exp(neg_clears - max_nc))))
        
        # Enclosing side
        all_verts = []
        for ix, iy, iang in hexes:
            all_verts.extend(_hex_vertices(ix, iy, iang))
        
        cx = sum(v[0] for v in all_verts) / len(all_verts)
        cy = sum(v[1] for v in all_verts) / len(all_verts)
        r = 0.0
        for outer_ang in [0.0, 30.0]:
            for kk in range(6):
                a = math.radians(outer_ang + 30 + 60 * kk)
                nx, ny = math.cos(a), math.sin(a)
                for vx, vy in all_verts:
                    proj = (vx - cx) * nx + (vy - cy) * ny
                    r = max(r, proj)
        side = r / (math.sqrt(3) / 2)
        
        # Penalty for negative clearance
        penalty = 0.0
        if soft_min < 0:
            penalty = -soft_min * 50
        
        return side + penalty
    
    x0 = flatten(inner_hexes)
    
    try:
        result = minimize(soft_objective, x0, method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 0.005, 'fatol': 0.005,
                                 'adaptive': True})
        if result.fun < soft_objective(x0):
            refined = unflatten(result.x)
            if _check_disjoint(refined, min_gap=1e-5):
                return refined
    except Exception:
        pass
    
    return inner_hexes

def _try_angle_variants(inner_hexes):
    """Try different inner hexagon angles to see if rotation helps."""
    best_inner = inner_hexes
    best_side = float('inf')
    
    # Get current enclosing side
    cx, cy, side, ang = _min_enclosing_hex(inner_hexes)
    best_side = side
    
    # Try rotating all hexagons
    for rot in [0.0, 15.0, 30.0, 45.0, 60.0]:
        rotated = [[x, y, 30.0 + rot] for x, y, _ in inner_hexes]
        if not _check_disjoint(rotated, min_gap=1e-5):
            continue
        cx2, cy2, side2, ang2 = _min_enclosing_hex(rotated)
        if side2 < best_side:
            best_side = side2
            best_inner = rotated
    
    return best_inner

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Fallback
        inner = _hexagonal_lattice_placement(N)
        cx, cy, side, ang = _min_enclosing_hex(inner)
        return inner, (cx, cy), side, ang
    
    # Step 1: Start with hexagonal lattice placement
    inner = _hexagonal_lattice_placement(N)
    
    # Step 2: Optimize positions
    if time.time() < _DEADLINE - 40.0:
        inner = _optimize_placement(inner, time_budget=25.0)
    
    # Step 3: Try angle variants
    if time.time() < _DEADLINE - 30.0:
        inner = _try_angle_variants(inner)
    
    # Step 4: Soft refinement
    if time.time() < _DEADLINE - 20.0:
        inner = _refine_with_softmin(inner, time_budget=10.0)
    
    # Step 5: Verify disjointness
    # Use a slightly larger min_gap to be safe against evaluator precision issues
    if not _check_disjoint(inner, min_gap=1e-4):
        # Fallback to gapped lattice which is guaranteed to be disjoint
        inner = _hexagonal_lattice_placement(N, gap_factor=1.05)
        # Re-verify just in case
        if not _check_disjoint(inner, min_gap=1e-5):
             # This should never happen with gap_factor=1.05, but if it does, we have a problem.
             # We will return it anyway as it's the best we have, but it might score 0.
             pass

    # Step 6: Compute optimal enclosing hexagon
    if time.time() > _DEADLINE - 10.0:
        # Quick estimate
        all_verts = []
        for ix, iy, iang in inner:
            all_verts.extend(_hex_vertices(ix, iy, iang))
        cx = sum(v[0] for v in all_verts) / len(all_verts)
        cy = sum(v[1] for v in all_verts) / len(all_verts)
        r = 0.0
        for k in range(6):
            a = math.radians(30 + 60 * k)
            nx, ny = math.cos(a), math.sin(a)
            for vx, vy in all_verts:
                proj = (vx - cx) * nx + (vy - cy) * ny
                r = max(r, proj)
        side = r / (math.sqrt(3) / 2)
        return inner, (cx, cy), side, 0.0
    
    cx, cy, side, angle = _min_enclosing_hex(inner)
    
    return inner, (cx, cy), side, angle
