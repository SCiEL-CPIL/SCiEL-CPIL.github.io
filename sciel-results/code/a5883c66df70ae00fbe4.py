"""Pack N unit hexagons into the smallest enclosing regular hexagon."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
import math

N = 11
_START_TIME = time.time()
DEADLINE = 120.0  # seconds, with margin below 150s

def _time_remaining():
    return DEADLINE - (time.time() - _START_TIME)

def hexagon_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon with side length 'side', center (cx,cy), rotation angle_deg."""
    rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        a = rad + i * math.pi / 3
        # Regular hexagon vertex at angle a, distance = side from center
        x = cx + side * math.cos(a)
        y = cy + side * math.sin(a)
        verts.append((x, y))
    return verts

def hex_in_hex_distance(cx, cy, side, angle_deg, outer_cx, outer_cy, outer_side, outer_angle_deg):
    """
    Compute minimum outer_side such that unit hexagon at (cx,cy,angle) fits in outer hexagon.
    Returns the required outer side (or large number if not feasible).
    Actually, we want to find, for a given outer hexagon, whether all inner hexagons fit.
    Better: for a set of inner hexagons, find min outer side.
    """
    # This is complex. Instead, for a fixed outer hexagon, check if each inner hex fits.
    pass

def required_outer_side(inner_hexes, outer_angle_deg, outer_cx=0.0, outer_cy=0.0):
    """
    Given list of (x, y, angle_deg) unit hexagons, find minimum side length of a regular
    hexagon centered at (outer_cx, outer_cy) with rotation outer_angle_deg that contains all.
    """
    min_side = 0.0
    outer_rad = math.radians(outer_angle_deg)
    # The 6 edge normals of the outer hexagon
    for i in range(6):
        a = outer_rad + i * math.pi / 3
        nx = -math.sin(a)
        ny = math.cos(a)
        # For each inner hexagon, compute max extent along this normal
        for (cx, cy, ang_deg) in inner_hexes:
            ang_rad = math.radians(ang_deg)
            # Unit hexagon (side=1) vertices relative to its center
            max_proj = 0.0
            for j in range(6):
                va = ang_rad + j * math.pi / 3
                vx = math.cos(va)
                vy = math.sin(vy) if False else math.sin(va)  # bug fix below
                # Proper:
            # Let's compute all 6 vertices
            max_proj = 0.0
            for j in range(6):
                va = ang_rad + j * math.pi / 3
                vx = math.cos(va)
                vy = math.sin(va)
                proj = (cx + vx) * nx + (cy + vy) * ny
                if proj > max_proj:
                    max_proj = proj
            # The outer hexagon edge in direction (nx,ny) is at distance s from center
            # along normal: dot((x-outer_cx), (y-outer_cy), (nx,ny)) <= s
            # So s >= max_proj - (outer_cx*nx + outer_cy*ny)
            req = max_proj - (outer_cx * nx + outer_cy * ny)
            if req > min_side:
                min_side = req
    return min_side

def inner_overlap_penalty(inner_hexes):
    """
    Penalty for overlapping inner hexagons.
    Two unit hexagons overlap if their centers are closer than ~2 (for same orientation)
    but this depends on relative angle. Approximate: if center distance < 2.0, penalty.
    More precisely: for same orientation, min center distance for no overlap is 2*sqrt(3)/2*... 
    Actually for two unit hexagons with same orientation, the inscribed circle radius is sqrt(3)/2,
    so they touch when centers are 2*sqrt(3)/2 * 2 = ... 
    Let me think: regular hex side 1, distance from center to vertex = 1, center to edge = sqrt(3)/2.
    Two hexagons with same orientation, parallel sides: they overlap if center dist < 2*sqrt(3)/2 = sqrt(3)? No.
    Actually for two regular hexagons of side 1, same orientation, the minimum center distance 
    to avoid overlap is 2 (vertex-to-vertex along an axis of symmetry). Let me just use distance < 1.95.
    """
    pen = 0.0
    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            x1, y1, a1 = inner_hexes[i]
            x2, y2, a2 = inner_hexes[j]
            d = math.hypot(x1 - x2, y1 - y2)
            # Minimum distance for non-overlap depends on relative rotation.
            # Worst case (same orientation): need d >= 2*inradius = sqrt(3) ≈ 1.732 for edge-to-edge? 
            # Actually for same-orientation hexagons, they just touch when d = 2 * apothem = sqrt(3)? No.
            # Apothem = sqrt(3)/2 for unit hex. Two hexagons side by side share an edge: center dist = 2*apothem = sqrt(3) ≈ 1.732.
            # But if rotated, they can get closer. Use a conservative threshold.
            min_d = sqrt3 = 1.7320508
            if d < min_d - 0.01:
                pen += (min_d - 0.01 - d) ** 2
    return pen

import math
sqrt3 = 1.7320508

def objective(params, outer_angle_deg):
    """
    params: flat array [x0,y0,a0, x1,y1,a1, ...]
    Returns total cost = required_outer_side + overlap_penalty * large_weight
    """
    inner = []
    for i in range(N):
        x = params[i * 3]
        y = params[i * 3 + 1]
        a = params[i * 3 + 2]
        inner.append((x, y, a))
    s_req = required_outer_side(inner, outer_angle_deg)
    pen = inner_overlap_penalty(inner)
    return s_req + 100.0 * pen

def make_initial_guess():
    """Create a compact hexagonal cluster initial guess."""
    hexes = []
    # Center
    hexes.append((0.0, 0.0, 0.0))
    # Ring 1: 6 hexagons at distance sqrt(3) (edge-adjacent)
    for i in range(6):
        a = math.radians(60 * i)
        x = sqrt3 * math.cos(a)
        y = sqrt3 * math.sin(a)
        hexes.append((x, y, 0.0))
    # Need 4 more (total 11). Add from ring 2.
    # Ring 2 positions at distance 2*sqrt(3) or at (2,0) type positions
    # Actually in hex grid, second ring positions... let me just place 4 more around
    for i in range(4):
        a = math.radians(30 + 60 * i)
        x = 2.0 * sqrt3 * math.cos(a)
        y = 2.0 * sqrt3 * math.sin(a)
        hexes.append((x, y, 0.0))
    return hexes

def construct():
    global _START_TIME, DEADLINE
    
    # Deadline guard
    if _time_remaining() < 5.0:
        # Return a safe valid configuration
        hexes = make_initial_guess()
        inner = [[h[0], h[1], h[2]] for h in hexes]
        s_req = required_outer_side(hexes, 0.0)
        return inner, (0.0, 0.0), s_req + 1e-6, 0.0
    
    # Initial compact guess
    init_hexes = make_initial_guess()
    
    # Try a few outer angles
    best_result = None
    best_cost = float('inf')
    
    for outer_angle in [0.0, 30.0]:
        if _time_remaining() < 10.0:
            break
            
        init_params = []
        for h in init_hexes:
            init_params.extend([h[0], h[1], h[2]])
        
        # Use differential evolution
        bounds = [(-8, 8), (-8, 8), (0, 360)] * N
        
        try:
            res = differential_evolution(
                lambda p: objective(p, outer_angle),
                bounds,
                x0=init_params,
                seed=42,
                maxiter=30,
                popsize=12,
                tol=1e-3,
                polish=True
            )
            best_params = res.x
        except Exception:
            best_params = init_params
        
        # Evaluate
        inner = []
        for i in range(N):
            inner.append((best_params[i*3], best_params[i*3+1], best_params[i*3+2]))
        
        pen = inner_overlap_penalty(inner)
        if pen > 1e-4:
            # Fall back to initial guess
            inner = init_hexes
        
        s_req = required_outer_side(inner, outer_angle)
        cost = s_req + 100 * inner_overlap_penalty(inner)
        
        if cost < best_cost:
            best_cost = cost
            best_inner = inner
            best_outer_angle = outer_angle
            best_side = s_req
    
    if best_result is None:
        hexes = make_initial_guess()
        inner = [[h[0], h[1], h[2]] for h in hexes]
        s_req = required_outer_side(hexes, 0.0)
        return inner, (0.0, 0.0), s_req + 1e-6, 0.0
    
    inner_list = [[h[0], h[1], h[2]] for h in best_inner]
    return inner_list, (0.0, 0.0), best_side + 1e-6, best_outer_angle

# Fix the bug in required_outer_side
def required_outer_side(inner_hexes, outer_angle_deg, outer_cx=0.0, outer_cy=0.0):
    min_side = 0.0
    outer_rad = math.radians(outer_angle_deg)
    for i in range(6):
        a = outer_rad + i * math.pi / 3
        nx = -math.sin(a)
        ny = math.cos(a)
        for (cx, cy, ang_deg) in inner_hexes:
            ang_rad = math.radians(ang_deg)
            max_proj = -float('inf')
            for j in range(6):
                va = ang_rad + j * math.pi / 3
                vx = math.cos(va)
                vy = math.sin(va)
                proj = (cx + vx) * nx + (cy + vy) * ny
                if proj > max_proj:
                    max_proj = proj
            req = max_proj - (outer_cx * nx + outer_cy * ny)
            if req > min_side:
                min_side = req
    return max(min_side, 0.0)

def construct():
    global _START_TIME
    _START_TIME = time.time()
    DEADLINE_LOCAL = 115.0  # safety margin below 150s
    
    def time_remaining():
        return DEADLINE_LOCAL - (time.time() - _START_TIME)
    
    sqrt3_val = math.sqrt(3)
    
    def hexes_init():
        hexes = []
        hexes.append((0.0, 0.0, 0.0))
        for i in range(6):
            a = math.radians(60 * i)
            hexes.append((sqrt3_val * math.cos(a), sqrt3_val * math.sin(a), 0.0))
        for i in range(4):
            a = math.radians(30 + 60 * i)
            hexes.append((2 * sqrt3_val * math.cos(a), 2 * sqrt3_val * math.sin(a), 0.0))
        return hexes
    
    def req_side(inner, oa_deg):
        min_s = 0.0
        oa_rad = math.radians(oa_deg)
        for i in range(6):
            a = oa_rad + i * math.pi / 3
            nx = -math.sin(a)
            ny = math.cos(a)
            for (cx, cy, ang_d) in inner:
                ar = math.radians(ang_d)
                mp = -1e18
                for j in range(6):
                    va = ar + j * math.pi / 3
                    proj = (cx + math.cos(va)) * nx + (cy + math.sin(va)) * ny
                    if proj > mp:
                        mp = proj
                r = mp - (0.0 * nx + 0.0 * ny)
                if r > min_s:
                    min_s = r
        return max(min_s, 0.0)
    
    def overlap_pen(inner):
        p = 0.0
        md = sqrt3_val - 0.02
        for i in range(N):
            for j in range(i + 1, N):
                d = math.hypot(inner[i][0] - inner[j][0], inner[i][1] - inner[j][1])
                if d < md:
                    p += (md - d) ** 2
        return p
    
    def cost(p, oa):
        inn = []
        for i in range(N):
            inn.append((p[i*3], p[i*3+1], p[i*3+2]))
        return req_side(inn, oa) + 100.0 * overlap_pen(inn)
    
    init = hexes_init()
    best_cost = 1e10
    best_inner = None
    best_oa = 0.0
    best_side = 0.0
    
    for oa in [0.0, 30.0]:
        if time_remaining() < 8.0:
            break
        p0 = []
        for h in init:
            p0.extend([h[0], h[1], h[2]])
        bounds = [(-6, 6), (-6, 6), (0, 360)] * N
        try:
            res = differential_evolution(lambda p: cost(p, oa), bounds, x0=p0,
                                          seed=42, maxiter=25, popsize=10, tol=1e-3, polish=True)
            bp = res.x
        except:
            bp = p0
        inn = [(bp[i*3], bp[i*3+1], bp[i*3+2]) for i in range(N)]
        pen = overlap_pen(inn)
        if pen > 1e-3:
            inn = init
        s = req_side(inn, oa)
        c = s + 100 * overlap_pen(inn)
        if c < best_cost:
            best_cost = c
            best_inner = inn
            best_oa = oa
            best_side = s
    
    if best_inner is None:
        best_inner = init
        best_side = req_side(best_inner, 0.0)
        best_oa = 0.0
    
    inner_list = [[h[0], h[1], h[2]] for h in best_inner]
    return inner_list, (0.0, 0.0), best_side + 1e-6, best_oa
