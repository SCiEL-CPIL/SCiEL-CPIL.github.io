"""Packing 11 unit hexagons in a minimum side-length regular hexagon."""
import math
import time
import numpy as np

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 118.0  # safety margin

def _hex_vertices(cx, cy, angle_deg=0.0):
    """Vertices of unit hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    return [(cx + math.cos(a0 + math.pi * k / 3),
             cy + math.sin(a0 + math.pi * k / 3)) for k in range(6)]

def _support(hx, hy, hangle, nx, ny):
    """Support function: max dot product of vertices with direction (nx,ny)."""
    a0 = math.radians(hangle)
    best = -1e18
    for k in range(6):
        a = a0 + math.pi * k / 3
        vx = hx + math.cos(a)
        vy = hy + math.sin(a)
        d = vx * nx + vy * ny
        if d > best:
            best = d
    return best

def _pair_clearance(h1, h2):
    """Minimum clearance between two unit hexagons via SAT.
    h1=(x,y,ang), h2=(x,y,ang). Positive=separated, negative=overlap."""
    x1, y1, a1 = h1
    x2, y2, a2 = h2
    
    # 12 candidate separating axes: 6 edge normals from each hexagon
    best = -1e18
    for base in [a1, a2]:
        for k in range(6):
            theta = math.radians(base + 30 + 60 * k)
            nx = math.cos(theta)
            ny = math.sin(theta)
            
            # Project both hexagons onto this axis
            s1 = _support(x1, y1, a1, nx, ny)
            s1n = _support(x1, y1, a1, -nx, -ny)
            s2 = _support(x2, y2, a2, nx, ny)
            s2n = _support(x2, y2, a2, -nx, -ny)
            
            # Gap in this direction
            gap = min(s2 + s1n, s1 + s2n)
            if gap > best:
                best = gap
    
    return best

def _check_disjoint(hexes, tol=1e-7):
    """Check all pairs are disjoint (clearance >= tol)."""
    n = len(hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if _pair_clearance(hexes[i], hexes[j]) < tol:
                return False
    return True

def _enclosing_side_lp(hexes, outer_angle):
    """For fixed outer hexagon orientation, find min side via LP.
    Returns (cx, cy, side) or None if infeasible."""
    # Collect all 66 vertices
    verts = []
    for hx, hy, ha in hexes:
        verts.extend(_hex_vertices(hx, hy, ha))
    
    # Outer hexagon normals (6 directions)
    normals = []
    for k in range(6):
        a = math.radians(outer_angle + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    # LP variables: [cx, cy, r] where r = side * sqrt(3)/2
    # Constraint: for each vertex v and each normal n:
    #   (vx - cx)*nx + (vy - cy)*ny <= r
    #   => -nx*cx - ny*cy - r <= -vx*nx - vy*ny
    # Also: -r <= 0
    
    A = []
    b = []
    for nx, ny in normals:
        for vx, vy in verts:
            A.append([-nx, -ny, -1.0])
            b.append(-vx * nx - vy * ny)
    A.append([0.0, 0.0, -1.0])
    b.append(0.0)
    
    A = np.array(A)
    b = np.array(b)
    c = np.array([0.0, 0.0, 1.0])
    
    from scipy.optimize import linprog
    res = linprog(c, A_ub=A, b_ub=b, method='highs', options={'presolve': True})
    
    if res.success:
        r = res.x[2]
        side = r / (math.sqrt(3) / 2)
        return res.x[0], res.x[1], side
    return None

def _min_enclosing(hexes):
    """Find minimum enclosing regular hexagon over all orientations.
    Returns (cx, cy, side, angle_deg)."""
    best_side = 1e18
    best = (0.0, 0.0, 0.0, 0.0)
    
    # Coarse sweep
    for ang in np.arange(0, 60, 3.0):
        if time.time() > _DEADLINE - 5:
            break
        r = _enclosing_side_lp(hexes, float(ang))
        if r is not None:
            cx, cy, side = r
            if side < best_side:
                best_side = side
                best = (cx, cy, side, float(ang))
    
    # Fine sweep around best
    if best_side < 1e18:
        for ang in np.arange(best[3] - 4, best[3] + 4, 0.5):
            if ang < 0 or ang >= 60:
                continue
            if time.time() > _DEADLINE - 3:
                break
            r = _enclosing_side_lp(hexes, float(ang))
            if r is not None:
                cx, cy, side = r
                if side < best_side:
                    best_side = side
                    best = (cx, cy, side, float(ang))
        
        # Very fine
        for ang in np.arange(best[3] - 1, best[3] + 1, 0.05):
            if ang < 0 or ang >= 60:
                continue
            if time.time() > _DEADLINE - 2:
                break
            r = _enclosing_side_lp(hexes, float(ang))
            if r is not None:
                cx, cy, side = r
                if side < best_side:
                    best_side = side
                    best = (cx, cy, side, float(ang))
    
    return best

def _lattice_positions_1_2_3_2_1():
    """Generate 11 hexagon positions in 1-2-3-2-1 hexagonal lattice pattern.
    All hexagons have the same orientation (0 degrees).
    Returns list of (x, y, angle) tuples."""
    # For unit hexagons with flat-top orientation (angle=0),
    # the lattice vectors for touching hexagons are:
    # u = (sqrt(3), 0)  -- horizontal neighbor
    # v = (sqrt(3)/2, 1.5)  -- diagonal neighbor
    
    s3 = math.sqrt(3)
    # Slight expansion to ensure strict disjointness
    eps = 0.001
    u = (s3 * (1 + eps), 0.0)
    v = (s3 / 2 * (1 + eps), 1.5 * (1 + eps))
    
    # 1-2-3-2-1 pattern (hexagonal cluster)
    # Row 0: 1 hex at center
    # Row 1: 2 hexes
    # Row 2: 3 hexes
    # Row 3: 2 hexes
    # Row 4: 1 hex
    
    # Use axial coordinates for hex grid
    # For flat-top hexagons, the 6 neighbors of (q,r) are:
    # (q+1,r), (q+1,r-1), (q,r-1), (q-1,r), (q-1,r+1), (q,r+1)
    # Position: x = s3*(q + r/2), y = 1.5*r
    
    positions = []
    # Center
    positions.append((0, 0))
    # 6 surrounding
    neighbors = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]
    for q, r in neighbors:
        positions.append((q, r))
    # Next ring (need 4 more to get to 11)
    # Ring 2: (2,0),(2,-1),(2,-2),(1,-2),(0,-2),(-1,-2),(-2,-1),(-2,0),(-2,1),(-1,1),(0,1),(1,1)
    # But we already have (1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1) from ring 1
    # Ring 2 positions not in ring 1:
    ring2 = [(2,0),(2,-1),(2,-2),(1,-2),(0,-2),(-1,-2),(-2,-1),(-2,0),(-2,1)]
    existing = set(positions)
    for q, r in ring2:
        if (q, r) not in existing:
            positions.append((q, r))
            if len(positions) == N:
                break
    
    # Convert to Cartesian
    result = []
    for q, r in positions[:N]:
        x = s3 * (1 + eps) * (q + r / 2.0)
        y = 1.5 * (1 + eps) * r
        result.append((x, y, 0.0))
    
    return result

def _rotate_cluster(hexes, angle_deg):
    """Rotate all hexagons around their centroid by angle_deg."""
    cx = sum(h[0] for h in hexes) / len(hexes)
    cy = sum(h[1] for h in hexes) / len(hexes)
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    result = []
    for hx, hy, ha in hexes:
        dx, dy = hx - cx, hy - cy
        nx = cx + ca * dx - sa * dy
        ny = cy + sa * dx + ca * dy
        result.append((nx, ny, ha + angle_deg))
    return result

def _optimize_cluster_rotation(hexes):
    """Optimize the global rotation of the rigid cluster to minimize enclosing hex side."""
    # For a rigid cluster, the enclosing hex side as a function of relative
    # angle between cluster and outer hex is periodic with period 60 degrees.
    # We sweep the outer hex angle (equivalently, rotate the cluster).
    
    best = _min_enclosing(hexes)
    return best

def _try_multiple_topologies():
    """Try several lattice topologies and return the best placement."""
    best_result = None
    best_side = 1e18
    
    # Topology 1: 1-2-3-2-1 hexagonal cluster
    hexes = _lattice_positions_1_2_3_2_1()
    if _check_disjoint(hexes, tol=1e-5):
        result = _min_enclosing(hexes)
        if result[2] < best_side:
            best_side = result[2]
            best_result = (hexes, result)
    
    # Topology 2: Slightly different arrangement - more compact
    # Try a 2-3-3-2-1 or other pattern
    s3 = math.sqrt(3)
    eps = 0.001
    scale = 1 + eps
    
    # Alternative: line of 5, line of 3, line of 3
    # Actually let's try a different hex packing:
    # Positions in a more elongated shape
    alt_positions = [
        (0, 0),
        (s3 * scale, 0),
        (-s3 * scale, 0),
        (s3/2 * scale, 1.5 * scale),
        (-s3/2 * scale, 1.5 * scale),
        (s3/2 * scale, -1.5 * scale),
        (-s3/2 * scale, -1.5 * scale),
        (2 * s3 * scale, 0),
        (-2 * s3 * scale, 0),
        (s3 * scale, 1.5 * scale),
        (s3 * scale, -1.5 * scale),
    ]
    hexes2 = [(x, y, 0.0) for x, y in alt_positions]
    if _check_disjoint(hexes2, tol=1e-5):
        result = _min_enclosing(hexes2)
        if result[2] < best_side:
            best_side = result[2]
            best_result = (hexes2, result)
    
    # Topology 3: 3-4-4 triangle-ish
    tri_positions = []
    for row in range(3):
        for col in range(4):
            x = s3 * scale * (col + 0.5 * row)
            y = 1.5 * scale * row
            tri_positions.append((x, y))
    # That's 12, take first 11
    tri_positions = tri_positions[:11]
    hexes3 = [(x, y, 0.0) for x, y in tri_positions]
    if _check_disjoint(hexes3, tol=1e-5):
        result = _min_enclosing(hexes3)
        if result[2] < best_side:
            best_side = result[2]
            best_result = (hexes3, result)
    
    return best_result

def _local_refine(hexes):
    """Local refinement: adjust each hexagon position slightly to reduce enclosing side."""
    from scipy.optimize import minimize
    
    def flatten(hexes):
        p = []
        for x, y, a in hexes:
            p.extend([x, y, math.radians(a)])
        return p
    
    def unflatten(p):
        h = []
        for i in range(len(p) // 3):
            h.append([p[3*i], p[3*i+1], math.degrees(p[3*i+2])])
        return h
    
    def objective(params):
        hexes = unflatten(params)
        # Quick check for overlap
        for i in range(len(hexes)):
            for j in range(i+1, len(hexes)):
                if _pair_clearance(hexes[i], hexes[j]) < 0:
                    return 1e10
        # Compute enclosing side (use fixed angle 0 for speed in inner loop)
        r = _enclosing_side_lp(hexes, 0.0)
        if r is None:
            return 1e10
        return r[2]
    
    x0 = flatten(hexes)
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.001})
        if res.fun < 1e9:
            refined = unflatten(res.x)
            if _check_disjoint(refined, tol=1e-5):
                return refined
    except:
        pass
    return hexes

def construct():
    global _DEADLINE
    _DEADLINE = _START_TIME + 118.0
    
    if time.time() > _DEADLINE - 5:
        # Emergency fallback
        s3 = math.sqrt(3)
        hexes = [(s3 * i, 0, 0.0) for i in range(N)]
        cx = s3 * (N-1) / 2
        cy = 0
        side = s3 * (N-1) + 2
        return [[x, y, a] for x, y, a in hexes], (cx, cy), side, 0.0
    
    # Step 1: Try multiple lattice topologies
    best = _try_multiple_topologies()
    
    if best is None:
        # Fallback: simple lattice
        s3 = math.sqrt(3)
        eps = 0.001
        hexes = []
        for i in range(N):
            hexes.append([s3 * (1+eps) * i, 0.0, 0.0])
        cx = s3 * (N-1) / 2
        cy = 0
        side = s3 * (N-1) + 2
        return hexes, (cx, cy), side, 0.0
    
    hexes, enc_result = best
    cx, cy, side, angle = enc_result
    
    # Step 2: Local refinement if time permits
    if time.time() < _DEADLINE - 30:
        refined = _local_refine(hexes)
        if refined is not None and refined != hexes:
            hexes = refined
            if time.time() < _DEADLINE - 15:
                enc_result = _min_enclosing(hexes)
                cx, cy, side, angle = enc_result
    
    # Step 3: Final precise enclosing hex computation
    if time.time() < _DEADLINE - 5:
        enc_result = _min_enclosing(hexes)
        cx, cy, side, angle = enc_result
    
    # Verify
    if not _check_disjoint(hexes, tol=1e-5):
        # Fallback to safe lattice
        s3 = math.sqrt(3)
        eps = 0.002
        hexes = []
        for i in range(N):
            hexes.append([s3 * (1+eps) * i, 0.0, 0.0])
        cx = s3 * (1+eps) * (N-1) / 2
        cy = 0
        side = s3 * (1+eps) * (N-1) + 2
        return hexes, (cx, cy), side, 0.0
    
    return [[x, y, a] for x, y, a in hexes], (cx, cy), side, angle
