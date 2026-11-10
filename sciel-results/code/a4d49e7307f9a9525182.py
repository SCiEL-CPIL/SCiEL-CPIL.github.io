# EVOLVE-BLOCK-START
import numpy as np
import time
import math

_START_TIME = time.time()
_TIME_LIMIT = 610.0  # safety margin before 630s kill


def _hex_vertices(cx, cy, s, a_deg):
    """Return (6,2) array of vertices for a regular hexagon."""
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])


def _edge_normals(poly):
    """Return (6,2) array of outward unit normals for a convex polygon (CCW vertices)."""
    n = len(poly)
    normals = np.zeros((n, 2))
    centroid = np.mean(poly, axis=0)
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        edge = p2 - p1
        # Normal: rotate edge 90 deg clockwise for outward normal in CCW polygon
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-15:
            normal /= norm
        # Ensure it points outward
        mid = (p1 + p2) / 2.0
        if np.dot(normal, mid - centroid) < 0:
            normal = -normal
        normals[i] = normal
    return normals


def _min_gap_sat(P, Q):
    """
    Minimum gap between two convex polygons P and Q using SAT.
    Returns 0.0 if they overlap or touch, positive value if separated.
    P, Q are (6,2) arrays of vertices in CCW order.
    """
    norms_P = _edge_normals(P)
    norms_Q = _edge_normals(Q)
    
    min_gap = float('inf')
    
    for n in np.vstack([norms_P, norms_Q]):
        proj_P = P @ n
        proj_Q = Q @ n
        
        min_P, max_P = proj_P.min(), proj_P.max()
        min_Q, max_Q = proj_Q.min(), proj_Q.max()
        
        if max_P < min_Q:
            gap = min_Q - max_P
        elif max_Q < min_P:
            gap = min_P - max_Q
        else:
            gap = 0.0
        
        if gap < min_gap:
            min_gap = gap
    
    return min_gap


def _check_separations(inner):
    """Check all pairs of inner hexagons are separated by > 1e-6."""
    n = len(inner)
    for i in range(n):
        v1 = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for j in range(i + 1, n):
            v2 = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            gap = _min_gap_sat(v1, v2)
            if gap < 1e-6:
                return False, gap, (i, j)
    return True, float('inf'), None


def _inner_fits_outer(inner, outer_side, outer_angle=0.0):
    """Check if all inner hexagon vertices are inside the outer hexagon (centered at origin)."""
    ov = _hex_vertices(0.0, 0.0, outer_side, outer_angle)
    ov_normals = _edge_normals(ov)
    
    for h in inner:
        v = _hex_vertices(h[0], h[1], 1.0, h[2])
        for p in v:
            for i in range(6):
                p1 = ov[i]
                p2 = ov[(i + 1) % 6]
                e = p2 - p1
                cross = e[0] * (p[1] - p1[1]) - e[1] * (p[0] - p1[0])
                if cross < -1e-6:
                    return False
    return True


def _min_outer_side(inner, max_side=20.0):
    """Binary search for minimum outer side length (centered at origin, angle 0)."""
    lo, hi = 3.0, max_side
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _inner_fits_outer(inner, mid):
            hi = mid
        else:
            lo = mid
    return hi


def _objective(x, n=11):
    """Objective: outer side length + penalty for overlaps."""
    inner = x.reshape(n, 3)
    
    # Penalty for overlaps
    pen = 0.0
    for i in range(n):
        v1 = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for j in range(i + 1, n):
            v2 = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            gap = _min_gap_sat(v1, v2)
            if gap < 1e-6:
                pen += (1e-6 - gap) ** 2 * 1000.0
    
    side = _min_outer_side(inner)
    return side + pen


def _initial_packing():
    """Good initial hexagonal grid packing of 11 unit hexagons."""
    # Center hexagon at origin, surrounded by 6, plus 4 more
    s3 = math.sqrt(3) / 2.0
    # Hex grid: center at (0,0)
    # Neighbors at distance 2*s3 = sqrt(3) ~ 1.732 for flat-top, or 2 for pointy-top
    # For unit side hexagons, center-to-center distance for touching = 2 * apothem = sqrt(3)
    # But actually for two unit hexagons to touch, center distance = 2 * apothem = sqrt(3)
    
    # Let's use a standard hexagonal arrangement
    # Flat-top hexagons: horizontal spacing = 1.5, vertical spacing = sqrt(3)
    # Pointy-top: horizontal spacing = sqrt(3), vertical spacing = 1.5
    
    # Using pointy-top (angle 0 means vertex at right)
    # Actually with our vertex formula, angle 0 puts vertex at (s, 0)
    # So it's "pointy-right"
    
    # For unit hexagons, the distance between centers of touching hexagons
    # depends on orientation. Let's just use a known good packing.
    
    # 7-hex cluster + 4 more
    dx = 3.0  # horizontal spacing for pointy-top
    dy = 2.0  # vertical spacing
    
    # Actually let me think more carefully.
    # A unit regular hexagon has inradius (apothem) = sqrt(3)/2, circumradius = 1
    # Two hexagons with the same orientation touch when center distance = 2 * apothem = sqrt(3)
    # if they share an edge, or 2 * circumradius = 2 if vertex-to-vertex.
    
    # For a hexagonal grid of same-orientation hexagons:
    # The 6 neighbors are at distance 2*apothem = sqrt(3) in the 6 edge-normal directions.
    
    # Let me just place them in a hex grid pattern
    # Using the fact that for same-orientation unit hexagons,
    # neighbors are at distance sqrt(3) in directions 0, 60, 120, 180, 240, 300
    
    d = math.sqrt(3)  # center-to-center distance for touching
    inner = np.zeros((11, 3))
    
    # Center
    inner[0] = [0, 0, 0]
    
    # 6 neighbors
    for i in range(6):
        angle = np.deg2rad(60.0 * i)
        inner[i + 1, 0] = d * np.cos(angle)
        inner[i + 1, 1] = d * np.sin(angle)
        inner[i + 1, 2] = 0
    
    # 4 more: place at second ring positions
    # Second ring in hex grid is at distance 2*d in 6 directions, plus 6 positions at d*sqrt(3)
    # Actually, let me just add 4 more at reasonable positions
    # Positions at distance 2*d in 4 of the 6 directions
    for i in range(4):
        angle = np.deg2rad(60.0 * (2 * i + 1))  # odd multiples: 60, 180, 300, 420=60
        # Let me use 0, 120, 240, 360=0 -> that's only 3 unique
        # Use 60, 120, 180, 240
        pass
    
    # Better: just use a known compact arrangement
    # Place remaining 4 at:
    # (2*d, 0), (-2*d, 0), (d*cos(60), d*sin(60) + d*sin(60))... 
    
    # Let me just use a simple grid-like arrangement that's known to work
    # Hexagonal close packing:
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 3 hexagons (offset)
    # Total: 10... need 11
    
    # Alternative: 
    # Row 0: 2
    # Row 1: 4
    # Row 2: 4
    # Row 3: 1
    # Total: 11
    
    # For pointy-top hexagons with side 1:
    # Horizontal spacing between centers in same row: sqrt(3)
    # Vertical spacing between rows: 1.5
    # Offset alternate rows by sqrt(3)/2
    
    h_sp = math.sqrt(3)  # horizontal spacing
    v_sp = 1.5  # vertical spacing
    offset = math.sqrt(3) / 2.0
    
    positions = []
    # Row 0: 2 hexagons
    positions.append((-h_sp/2, 0))
    positions.append((h_sp/2, 0))
    # Row 1: 4 hexagons
    for i in range(4):
        positions.append((-1.5*h_sp + i*h_sp, -v_sp))
    # Row 2: 4 hexagons
    for i in range(4):
        positions.append((-1.5*h_sp + i*h_sp, v_sp))
    # Row 3: 1 hexagon
    positions.append((0, 2*v_sp))
    
    for i in range(11):
        inner[i, 0] = positions[i][0]
        inner[i, 1] = positions[i][1]
        inner[i, 2] = 0
    
    return inner


def _try_optimize(x0, n=11, max_iter=2000):
    """Try to optimize from initial position x0."""
    from scipy.optimize import minimize
    
    def obj(x):
        if time.time() - _START_TIME > _TIME_LIMIT:
            return 1e10
        return _objective(x, n)
    
    try:
        res = minimize(obj, x0.flatten(), method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-5,
                                'adaptive': True})
        x = res.x.reshape(n, 3)
        return x
    except Exception:
        return None


def hexagon_packing_11():
    n = 11
    inner0 = _initial_packing()
    
    # Check if initial is valid
    valid, gap, _ = _check_separations(inner0)
    best_x = inner0.copy()
    best_side = _min_outer_side(inner0)
    
    if not valid:
        # Initial packing has overlaps, need to fix
        # Try to spread them out
        for attempt in range(20):
            if time.time() - _START_TIME > _TIME_LIMIT - 30:
                break
            rng = np.random.default_rng(attempt)
            x_try = inner0 + rng.normal(0, 0.3, inner0.shape)
            x_try[:, 2] = 0  # keep angles at 0 for now
            valid, gap, _ = _check_separations(x_try)
            if valid:
                side = _min_outer_side(x_try)
                if side < best_side:
                    best_side = side
                    best_x = x_try.copy()
    
    # Now optimize
    starts = [best_x.copy()]
    
    # Add some jittered starts
    rng = np.random.default_rng(42)
    for _ in range(8):
        if time.time() - _START_TIME > _TIME_LIMIT - 60:
            break
        starts.append(best_x + rng.normal(0, 0.2, best_x.shape))
    
    # Also try with some angle variation
    for _ in range(4):
        if time.time() - _START_TIME > _TIME_LIMIT - 60:
            break
        x_angle = best_x.copy()
        x_angle[:, 2] = rng.uniform(-30, 30, size=n)
        starts.append(x_angle)
    
    for s0 in starts:
        if time.time() - _START_TIME > _TIME_LIMIT - 30:
            break
        
        x_opt = _try_optimize(s0, n, max_iter=3000)
        if x_opt is None:
            continue
        
        # Check validity
        valid, gap, _ = _check_separations(x_opt)
        if valid:
            side = _min_outer_side(x_opt)
            if side < best_side:
                best_side = side
                best_x = x_opt.copy()
    
    # Final refinement with tighter tolerances
    if time.time() - _START_TIME < _TIME_LIMIT - 30:
        x_ref = _try_optimize(best_x, n, max_iter=5000)
        if x_ref is not None:
            valid, gap, _ = _check_separations(x_ref)
            if valid:
                side = _min_outer_side(x_ref)
                if side < best_side:
                    best_side = side
                    best_x = x_ref.copy()
    
    # Verify edge lengths are within tolerance
    for h in best_x:
        v = _hex_vertices(h[0], h[1], 1.0, h[2])
        for i in range(6):
            e = v[(i + 1) % 6] - v[i]
            length = np.linalg.norm(e)
            if abs(length - 1.0) > 1e-8:
                # Shouldn't happen with our vertex formula, but just in case
                pass
    
    inner = best_x
    side = best_side
    outer = np.array([0.0, 0.0, 0.0])
    
    return inner, outer, float(side)


# EVOLVE-BLOCK-END
