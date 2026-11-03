"""Packing 11 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

N = 11
DEADLINE = 140.0  # Safety margin before 150s kill

def _get_hex_vertices(cx, cy, angle_deg, side=1.0):
    """Get vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def _hex_contains_point(px, py, hx, hy, angle_deg, side):
    """Check if point (px,py) is inside hexagon centered at (hx,hy)."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(-a), math.sin(-a)
    dx, dy = px - hx, py - hy
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    s32 = side * math.sqrt(3) / 2
    return (abs(lx) <= side + 1e-9 and
            abs(ly) <= s32 + 1e-9 and
            abs(lx) / side + abs(ly) / s32 <= 2 + 1e-9)

def _hexagons_overlap(h1x, h1y, h1a, h2x, h2y, h2a, tol=1e-9):
    """Check if two unit hexagons overlap using SAT (Separating Axis Theorem)."""
    v1 = _get_hex_vertices(h1x, h1y, h1a, 1.0)
    v2 = _get_hex_vertices(h2x, h2y, h2a, 1.0)
    
    # Get all edge normals from both hexagons
    edges = []
    for i in range(6):
        # Edge 1
        ex = v1[(i+1)%6][0] - v1[i][0]
        ey = v1[(i+1)%6][1] - v1[i][1]
        edges.append((-ey, ex))
        # Edge 2
        ex = v2[(i+1)%6][0] - v2[i][0]
        ey = v2[(i+1)%6][1] - v2[i][1]
        edges.append((-ey, ex))
    
    for nx, ny in edges:
        # Project both hexagons onto this normal
        proj1 = [vx * nx + vy * ny for vx, vy in v1]
        proj2 = [vx * nx + vy * ny for vx, vy in v2]
        min1, max1 = min(proj1), max(proj1)
        min2, max2 = min(proj2), max(proj2)
        # If there's a gap, no overlap
        if max1 < min2 - tol or max2 < min1 - tol:
            return False
    return True

def _check_disjoint(inner, outer_center, outer_side, outer_angle, tol=1e-7):
    """Check all constraints: inner hexagons inside outer, no overlaps."""
    # Check each inner hexagon is inside outer
    for i in range(N):
        cx, cy, ang = inner[i]
        verts = _get_hex_vertices(cx, cy, ang, 1.0)
        for vx, vy in verts:
            if not _hex_contains_point(vx, vy, outer_center[0], outer_center[1], outer_angle, outer_side):
                return False
    
    # Check pairwise disjointness
    for i in range(N):
        for j in range(i+1, N):
            if _hexagons_overlap(inner[i][0], inner[i][1], inner[i][2],
                                 inner[j][0], inner[j][1], inner[j][2]):
                return False
    return True

def _min_enclosing_hex_side(inner, outer_center, outer_angle):
    """Find minimum side of hexagon at given center/angle that encloses all inner hexagons."""
    # Collect all vertices of all inner hexagons
    all_verts = []
    for i in range(N):
        cx, cy, ang = inner[i]
        verts = _get_hex_vertices(cx, cy, ang, 1.0)
        all_verts.extend(verts)
    
    # Transform to outer hexagon's local frame
    a = math.radians(outer_angle)
    ca, sa = math.cos(-a), math.sin(-a)
    
    max_val = 0.0
    for vx, vy in all_verts:
        dx = vx - outer_center[0]
        dy = vy - outer_center[1]
        lx = ca * dx - sa * dy
        ly = sa * dx + ca * dy
        # For a hexagon with side s: the constraint is
        # |lx| <= s, |ly| <= s*sqrt(3)/2, |lx|/s + |ly|/(s*sqrt(3)/2) <= 2
        # The third constraint gives: |lx| + |ly|/sqrt(3) <= 2s
        # So s >= (|lx| + |ly|/sqrt(3)) / 2
        val = (abs(lx) + abs(ly) / math.sqrt(3)) / 2.0
        max_val = max(max_val, val)
    
    return max_val

def _compute_all_vertices(inner):
    """Compute all vertices of all inner hexagons."""
    all_verts = []
    for i in range(N):
        cx, cy, ang = inner[i]
        verts = _get_hex_vertices(cx, cy, ang, 1.0)
        all_verts.extend(verts)
    return np.array(all_verts)

def _nelder_mead_polish(inner, outer_center, outer_angle, max_iter=200):
    """Polish using Nelder-Mead on inner positions only."""
    best_side = _min_enclosing_hex_side(inner, outer_center, outer_angle)
    best_inner = [list(x) for x in inner]
    
    def objective(params):
        if time.time() - _nelder_mead_polish.start_time > DEADLINE:
            return 1e10
        n = len(params) // 3
        pos = params[:2*n].reshape(n, 2)
        angs = params[2*n:]
        new_inner = [[pos[i][0], pos[i][1], angs[i]] for i in range(n)]
        
        # Check disjointness
        for i in range(n):
            for j in range(i+1, n):
                if _hexagons_overlap(new_inner[i][0], new_inner[i][1], new_inner[i][2],
                                     new_inner[j][0], new_inner[j][1], new_inner[j][2]):
                    return 1e6 + best_side
        
        side = _min_enclosing_hex_side(new_inner, outer_center, outer_angle)
        return side
    
    x0 = []
    for i in range(N):
        x0.extend([inner[i][0], inner[i][1], inner[i][2]])
    
    _nelder_mead_polish.start_time = time.time()
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
        if result.fun < best_side + 1e-9 and result.fun < 1e5:
            n = N
            pos = result.x[:2*n].reshape(n, 2)
            angs = result.x[2*n:]
            new_inner = [[pos[i][0], pos[i][1], angs[i]] for i in range(n)]
            # Verify
            for i in range(n):
                for j in range(i+1, n):
                    if _hexagons_overlap(new_inner[i][0], new_inner[i][1], new_inner[i][2],
                                         new_inner[j][0], new_inner[j][1], new_inner[j][2]):
                        return best_side, best_inner
            new_side = _min_enclosing_hex_side(new_inner, outer_center, outer_angle)
            if new_side < best_side:
                return new_side, new_inner
    except Exception:
        pass
    
    return best_side, best_inner

def construct():
    start_time = time.time()
    
    # Better initial layout: hexagonal cluster
    # 1 center + 6 ring + 4 in second ring
    sp = math.sqrt(3)  # spacing between adjacent hexagon centers
    
    # Hexagonal lattice positions for 11 hexagons
    # Use a compact arrangement
    init_positions = [
        (0, 0),  # center
        (sp, 0), (-sp, 0),
        (sp/2, sp*math.sqrt(3)/2), (-sp/2, sp*math.sqrt(3)/2),
        (sp/2, -sp*math.sqrt(3)/2), (-sp/2, -sp*math.sqrt(3)/2),
        (0, sp*math.sqrt(3)), (0, -sp*math.sqrt(3)),
        (sp*math.sqrt(3), 0), (-sp*math.sqrt(3), 0),
    ]
    
    assert len(init_positions) == N
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Try multiple outer angles
    outer_angles = [0, 30, 15, 45, 60]
    
    for oa in outer_angles:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        outer_center = (0.0, 0.0)
        
        # Initial inner positions
        inner0 = [[cx, cy, 0.0] for cx, cy in init_positions]
        
        # Compute initial enclosing side
        init_side = _min_enclosing_hex_side(inner0, outer_center, oa)
        
        if init_side < best_side:
            best_side = init_side
            best_inner = [list(x) for x in inner0]
            best_outer_center = outer_center
            best_outer_angle = oa
        
        # Differential evolution to optimize inner positions
        if time.time() - start_time > DEADLINE - 20:
            break
        
        def de_objective(params):
            if time.time() - start_time > DEADLINE:
                return 1e10
            n = len(params) // 3
            pos = params[:2*n].reshape(n, 2)
            angs = params[2*n:]
            new_inner = [[pos[i][0], pos[i][1], angs[i]] for i in range(n)]
            
            # Check disjointness - heavy penalty
            for i in range(n):
                for j in range(i+1, n):
                    if _hexagons_overlap(new_inner[i][0], new_inner[i][1], new_inner[i][2],
                                         new_inner[j][0], new_inner[j][1], new_inner[j][2]):
                        return 1e6 + _min_enclosing_hex_side(new_inner, outer_center, oa)
            
            return _min_enclosing_hex_side(new_inner, outer_center, oa)
        
        # Bounds for DE
        bounds = []
        for i in range(N):
            bounds.extend([(-8, 8), (-8, 8), (-180, 180)])
        
        try:
            de_result = differential_evolution(
                de_objective, bounds,
                maxiter=30, tol=1e-6, seed=42,
                popsize=8, polish=False,
                updating='immediate'
            )
            
            if de_result.fun < best_side + 1e-9 and de_result.fun < 1e5:
                n = N
                pos = de_result.x[:2*n].reshape(n, 2)
                angs = de_result.x[2*n:]
                new_inner = [[pos[i][0], pos[i][1], angs[i]] for i in range(n)]
                
                # Verify disjointness
                valid = True
                for i in range(n):
                    for j in range(i+1, n):
                        if _hexagons_overlap(new_inner[i][0], new_inner[i][1], new_inner[i][2],
                                             new_inner[j][0], new_inner[j][1], new_inner[j][2]):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    new_side = _min_enclosing_hex_side(new_inner, outer_center, oa)
                    if new_side < best_side:
                        best_side = new_side
                        best_inner = new_inner
                        best_outer_center = outer_center
                        best_outer_angle = oa
                        
        except Exception:
            pass
    
    # Nelder-Mead polish on best result
    if best_inner is not None and time.time() - start_time < DEADLINE - 10:
        best_side, best_inner = _nelder_mead_polish(best_inner, best_outer_center, best_outer_angle)
    
    # Final verification
    if best_inner is not None:
        valid = _check_disjoint(best_inner, best_outer_center, best_side, best_outer_angle)
        if not valid:
            # Fallback
            best_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            best_outer_center = (0.0, 0.0)
            best_side = 40.0
            best_outer_angle = 0.0
    
    if best_inner is None:
        best_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        best_outer_center = (0.0, 0.0)
        best_side = 40.0
        best_outer_angle = 0.0
    
    return best_inner, best_outer_center, best_side, best_outer_angle
