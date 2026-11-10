# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side s, rotation a_deg."""
    a = np.radians(a_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _hex_face_normals(a_deg):
    """Return 6 outward face normals (unit vectors) for a hexagon with rotation a_deg."""
    a = np.radians(a_deg)
    # Face normals are perpendicular to edges. For a regular hexagon, the face normals
    # point in directions a + 30 + 60*i degrees
    angles = a + np.radians(30) + np.radians(60) * np.arange(6)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _pair_separated(v1, n1, v2, n2, tol=1e-6):
    """
    Check if two hexagons are separated.
    v1, n1: vertices and normals of hex1
    v2, n2: vertices and normals of hex2
    Returns True if separated (valid), False if overlapping.
    
    Two convex polygons are separated if there exists a separating axis.
    For two regular hexagons, we check all 12 edge normals (6 from each).
    For each normal, project both polygons onto that axis and check if intervals are apart.
    """
    # Check normals from hex1
    for i in range(6):
        n = n1[i]
        proj1 = v1 @ n
        proj2 = v2 @ n
        if proj1.max() - proj2.min() < -tol or proj2.max() - proj1.min() < -tol:
            # Intervals are apart -> separating axis found
            return True
    
    # Check normals from hex2
    for i in range(6):
        n = n2[i]
        proj1 = v1 @ n
        proj2 = v2 @ n
        if proj1.max() - proj2.min() < -tol or proj2.max() - proj1.min() < -tol:
            return True
    
    return False


def _check_inner_separation(inner_data, tol=1e-6):
    """Check all pairs of inner hexagons are separated."""
    n = len(inner_data)
    vertices = []
    normals = []
    for i in range(n):
        cx, cy, a = inner_data[i]
        v = _hex_vertices(cx, cy, 1.0, a)
        nm = _hex_face_normals(a)
        vertices.append(v)
        normals.append(nm)
    
    for i in range(n):
        for j in range(i + 1, n):
            if not _pair_separated(vertices[i], normals[i], vertices[j], normals[j], tol):
                return False
    return True


def _check_inside_outer(inner_data, outer_data, tol=1e-6):
    """Check all inner hexagons are inside the outer hexagon."""
    cx, cy, a = outer_data
    # We need to know the side length; but for containment check we use the outer hex vertices
    # Actually, the containment check is: every inner vertex p satisfies
    # cross(p2 - p1, p - p1) >= -1e-6 for each outer edge from p1 to p2
    # But we need the outer side length to get vertices. Let's pass it.
    pass


def _check_inside_outer_full(inner_data, outer_data, side, tol=1e-6):
    """Check all inner hexagons are inside the outer hexagon with given side length."""
    cx, cy, a = outer_data
    outer_verts = _hex_vertices(cx, cy, side, a)
    n_outer = 6
    
    for i in range(len(inner_data)):
        ix, iy, ia = inner_data[i]
        inner_verts = _hex_vertices(ix, iy, 1.0, ia)
        for p in inner_verts:
            for e in range(n_outer):
                p1 = outer_verts[e]
                p2 = outer_verts[(e + 1) % n_outer]
                edge = p2 - p1
                vec = p - p1
                cross_val = edge[0] * vec[1] - edge[1] * vec[0]
                if cross_val < -tol:
                    return False
    return True


def _validate(inner_data, outer_data, side, tol=1e-6):
    """Full validation."""
    # Check finiteness
    if not (np.all(np.isfinite(inner_data)) and np.all(np.isfinite(outer_data)) and np.isfinite(side)):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    
    # Check inner separation
    if not _check_inner_separation(inner_data, tol):
        return False
    
    # Check containment
    if not _check_inside_outer_full(inner_data, outer_data, side, tol):
        return False
    
    # Check edge lengths of shifted inner hexagons
    outer_center = np.array([outer_data[0], outer_data[1]])
    for i in range(len(inner_data)):
        ix, iy, ia = inner_data[i]
        shifted = np.array([ix, iy]) - outer_center
        # Reconstruct with shifted center
        v = _hex_vertices(shifted[0], shifted[1], 1.0, ia)
        for e in range(6):
            d = v[(e + 1) % 6] - v[e]
            length = np.linalg.norm(d)
            if abs(length - 1.0) > 1e-8:
                return False
    
    return True


def _objective(params, outer_side):
    """
    Objective function: minimize outer_side, but with penalties for violations.
    Actually, we'll minimize a penalty that is 0 if valid and large if not,
    plus the outer side length.
    
    Better: we'll use a two-phase approach. First find a valid packing, then minimize side.
    For now, let's use a penalty-based objective.
    
    params: 33 values = 11 * (x, y, angle_deg)
    """
    inner = params.reshape(11, 3)
    
    # Compute required outer side to contain all inner hexagons
    # The outer hexagon is centered at origin with angle 0 for simplicity in optimization
    # Actually, let's fix outer at (0,0,0) and just compute the minimum side needed
    
    # For a hexagon centered at origin with angle 0, the containment constraint is:
    # For each inner vertex, compute the "hex distance" and the side must be >= max hex distance
    
    # The hex distance of a point (x,y) from center for a hexagon with side s, angle 0:
    # The hexagon has vertices at (s*cos(60i), s*sin(60i))
    # The supporting function in direction theta is s * max over edges...
    
    # Simpler: for each inner hexagon, compute its 6 vertices, and for each vertex,
    # compute the minimum side length of a hexagon centered at origin, angle 0, that contains it.
    
    # A point (px, py) is inside a regular hexagon (center 0,0, angle 0, side s) iff:
    # |px| <= s, |py| <= s*sqrt(3)/2, and |px| + |py|/sqrt(3) <= s
    # Wait, let me think more carefully.
    
    # For a regular hexagon with flat top (angle 0 means vertex at 0 degrees? or flat top?)
    # The problem says: vertices at (cx + s*cos(a + 60i deg), cy + s*sin(a + 60i deg))
    # So with a=0, vertex 0 is at (s, 0), vertex 1 at (s/2, s*sqrt(3)/2), etc.
    # This is a "pointy" hexagon (vertex on the right).
    
    # The half-width in x direction is s (from -s to s)
    # The half-height in y direction is s*sqrt(3)/2
    # The constraint is: |x| + |y|/sqrt(3) <= s  ... let me verify.
    # At vertex (s, 0): |s| + 0 = s. OK.
    # At vertex (s/2, s*sqrt(3)/2): s/2 + (s*sqrt(3)/2)/sqrt(3) = s/2 + s/2 = s. OK.
    # So the constraint is |x| + |y|/sqrt(3) <= s.
    
    required_side = 0.0
    for i in range(11):
        ix, iy, ia = inner[i]
        v = _hex_vertices(ix, iy, 1.0, ia)
        for p in v:
            hex_dist = abs(p[0]) + abs(p[1]) / np.sqrt(3)
            required_side = max(required_side, hex_dist)
    
    # Add small margin
    required_side += 1e-7
    
    # Check if inner hexagons are separated
    separated = _check_inner_separation(inner, tol=1e-6)
    
    if not separated:
        # Penalty: sum of overlaps (approximate)
        # For simplicity, use a large penalty
        return 1000.0 + required_side
    
    return required_side


def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Initial guess: hexagonal grid arrangement
    # Unit hexagons in a honeycomb pattern
    # Center at (0,0), neighbors at distance 2 in 6 directions
    # Hexagon centers in a hex grid:
    # The distance between centers of adjacent unit hexagons is 2 (since each has "radius" 1)
    # Actually, the inradius of a unit hexagon is sqrt(3)/2, so adjacent centers are 2*sqrt(3)/2 = sqrt(3) apart?
    # No. Two unit hexagons sharing an edge: the distance between centers is 2 * (inradius) = 2 * sqrt(3)/2 = sqrt(3).
    # Wait, let me reconsider. A regular hexagon with side 1 has inradius sqrt(3)/2 and circumradius 1.
    # Two hexagons sharing an edge: the centers are separated by 2 * inradius = sqrt(3).
    
    # Hex grid with spacing sqrt(3):
    # Actually, for a honeycomb of unit hexagons, the center-to-center distance is 2 * (sqrt(3)/2) = sqrt(3) for adjacent hexagons.
    # But the hex grid has axes at 60 degrees.
    
    # Let's use axial coordinates. For a hex grid, the positions are:
    # (q, r) -> (sqrt(3) * (q + r/2), 3/2 * r)  ... this is for pointy-top hexes.
    # For our hexagons (vertex at 0 deg), the grid is rotated.
    
    # Let me just use a simple arrangement and optimize.
    
    # 11 hexagons: center + 6 around + 4 more
    # Arrangement:
    #   Row 0: 3 hexagons
    #   Row 1: 4 hexagons  
    #   Row 2: 4 hexagons
    # Or:
    #   Center, 6 neighbors, then 4 more in a ring
    
    # Let's use the arrangement from the seed but optimize it.
    inner_data = np.array([
        [0, 0, 0],
        [-2.5, 0, 0],
        [2.5, 0, 0],
        [-1.25, 2.17, 0],
        [1.25, 2.17, 0],
        [-1.25, -2.17, 0],
        [1.25, -2.17, 0],
        [-3.75, 2.17, 0],
        [3.75, 2.17, 0],
        [-3.75, -2.17, 0],
        [3.75, -2.17, 0],
    ])
    
    # Compute initial side
    required_side = 0.0
    for i in range(11):
        ix, iy, ia = inner_data[i]
        v = _hex_vertices(ix, iy, 1.0, ia)
        for p in v:
            hex_dist = abs(p[0]) + abs(p[1]) / np.sqrt(3)
            required_side = max(required_side, hex_dist)
    
    best_inner = inner_data.copy()
    best_side = required_side
    best_score = 1.0 / best_side if best_side > 0 else 0
    
    # Verify initial
    outer_data = np.array([0.0, 0.0, 0.0])
    if _validate(inner_data, outer_data, required_side):
        best_inner = inner_data.copy()
        best_side = required_side
        best_score = 1.0 / best_side
    
    # Now optimize using differential evolution
    # Parameters: 11 * 3 = 33
    # Bounds for x, y: [-10, 10], angle: [0, 60]
    bounds = [(-10, 10)] * 11 + [(-10, 10)] * 11 + [(0, 60)] * 11
    
    # We need to check time
    if time.time() > deadline - 120:
        return best_inner, outer_data, best_side
    
    try:
        result = differential_evolution(
            _objective,
            bounds,
            args=(None,),
            maxiter=200,
            popsize=15,
            tol=1e-4,
            seed=42,
            polish=True,
            init='sobol',
            update_interval=10
        )
        
        if result.success or result.fun < best_score * best_side:
            inner_opt = result.x.reshape(11, 3)
            # Compute side
            req_side = 0.0
            for i in range(11):
                ix, iy, ia = inner_opt[i]
                v = _hex_vertices(ix, iy, 1.0, ia)
                for p in v:
                    hex_dist = abs(p[0]) + abs(p[1]) / np.sqrt(3)
                    req_side = max(req_side, hex_dist)
            req_side += 1e-7
            
            if _validate(inner_opt, outer_data, req_side):
                if req_side < best_side:
                    best_inner = inner_opt
                    best_side = req_side
                    best_score = 1.0 / req_side
    except Exception:
        pass
    
    # Try local refinement with Nelder-Mead on best solution
    if time.time() < deadline - 60:
        try:
            def local_obj(params):
                inner = params.reshape(11, 3)
                req_side = 0.0
                for i in range(11):
                    ix, iy, ia = inner[i]
                    v = _hex_vertices(ix, iy, 1.0, ia)
                    for p in v:
                        hex_dist = abs(p[0]) + abs(p[1]) / np.sqrt(3)
                        req_side = max(req_side, hex_dist)
                req_side += 1e-7
                
                if not _check_inner_separation(inner, tol=1e-6):
                    return 1000.0 + req_side
                return req_side
            
            local_result = minimize(
                local_obj,
                best_inner.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if local_result.fun < best_side:
                inner_lm = local_result.x.reshape(11, 3)
                req_side = 0.0
                for i in range(11):
                    ix, iy, ia = inner_lm[i]
                    v = _hex_vertices(ix, iy, 1.0, ia)
                    for p in v:
                        hex_dist = abs(p[0]) + abs(p[1]) / np.sqrt(3)
                        req_side = max(req_side, hex_dist)
                req_side += 1e-7
                
                if _validate(inner_lm, outer_data, req_side):
                    if req_side < best_side:
                        best_inner = inner_lm
                        best_side = req_side
                        best_score = 1.0 / req_side
        except Exception:
            pass
    
    # Final validation
    outer_data = np.array([0.0, 0.0, 0.0])
    if not _validate(best_inner, outer_data, best_side):
        # Fallback to seed
        best_inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ])
        best_side = 8.0
    
    return best_inner, outer_data, best_side


# EVOLVE-BLOCK-END
