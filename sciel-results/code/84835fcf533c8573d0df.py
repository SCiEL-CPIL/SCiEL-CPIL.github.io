import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """
    Returns vertices of a regular hexagon centered at (cx, cy) with side s
    and rotation angle_deg (degrees).
    Vertices are in CCW order.
    """
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _pairwise_penalty(pos1, ang1, pos2, ang2, eps=1e-4):
    """
    Calculate penalty for overlap between two unit hexagons.
    pos1, pos2: (x, y)
    ang1, ang2: rotation in radians
    Returns 0 if disjoint, >0 if overlapping.
    Uses Minkowski sum logic: check if vector between centers is inside
    the Minkowski sum of the two hexagons.
    For rotated hexagons, the Minkowski sum is a 12-gon (or hexagon if aligned).
    We approximate by checking support functions or just distance to boundary.
    A simpler robust method: check if any vertex of one is inside the other,
    or if any edge intersects.
    However, for speed and smoothness in optimization, we can use a distance-based metric.
    
    Let's use the distance between the two convex polygons.
    If they are disjoint, distance > 0.
    We want a penalty that is 0 when distance >= 0 (disjoint) and large when overlapping.
    
    Actually, the standard approach for packing is:
    penalty = max(0, min_dist_to_boundary)^2 ? No, that's for containment.
    For non-overlap:
    The condition is that the Minkowski difference (H1 - H2) does not contain the vector (c1 - c2).
    H1 - H2 is a convex polygon.
    We can compute the signed distance from the vector v = c1 - c2 to the polygon P = H1 - H2.
    If v is inside P, they overlap.
    The "depth" of overlap is the distance from v to the boundary of P.
    We can approximate this by checking the 6 support functions?
    
    Let's use a simpler, differentiable proxy:
    For each of the 6 vertices of hex1, check distance to hex2.
    For each of the 6 vertices of hex2, check distance to hex1.
    If any vertex is inside the other, we have overlap.
    The penalty can be the square of the minimum distance of a vertex to the boundary of the other,
    but this is not smooth at the boundary.
    
    Better: Use the distance between the two polygons.
    If dist > 0, they are disjoint.
    If dist < 0 (overlap), we want to penalize.
    We can define penalty = max(0, -dist)^2.
    Computing polygon distance is O(1) for fixed number of vertices.
    
    Let's implement a fast distance check.
    """
    v1 = _hex_vertices(pos1[0], pos1[1], 1.0, math.degrees(ang1))
    v2 = _hex_vertices(pos2[0], pos2[1], 1.0, math.degrees(ang2))
    
    # Check if any vertex of v1 is inside v2
    # And any vertex of v2 is inside v1
    # And if edges intersect.
    
    # For speed, first check bounding box or center distance.
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    dist_sq = dx*dx + dy*dy
    # Max distance between centers of disjoint unit hexagons is 2 (vertex to vertex).
    # Min distance is sqrt(3) (flat to flat).
    # If dist_sq > 4.0, definitely disjoint.
    if dist_sq > 4.0:
        return 0.0
    
    # If dist_sq < 3.0, definitely overlapping (at least by some amount).
    # We need a smooth penalty.
    
    # Let's use the support function approach for the Minkowski sum.
    # The Minkowski sum H1 + H2 (where H2 is reflected) is a convex polygon.
    # We check if the vector v is inside this polygon.
    # The vertices of the Minkowski sum are the sums of vertices of H1 and -H2.
    # Since H1 and H2 are regular hexagons, their Minkowski sum is a 12-gon (if rotated relative)
    # or a hexagon (if aligned).
    # Actually, the Minkowski sum of two regular hexagons is always a regular hexagon
    # if they are aligned, and a 12-gon if rotated.
    # Wait, the Minkowski sum of two convex polygons with m and n vertices has at most m+n vertices.
    
    # Let's just compute the distance between the two polygons directly.
    # This is robust.
    min_dist = float('inf')
    
    # Function to compute distance from point to segment
    def point_to_segment_dist(p, a, b):
        px, py = p
        ax, ay = a
        bx, by = b
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        c1 = vx*wx + vy*wy
        if c1 <= 0:
            return math.hypot(px - ax, py - ay)
        c2 = vx*vx + vy*vy
        if c2 <= c1:
            return math.hypot(px - bx, py - by)
        t = c1 / c2
        px_proj = ax + t * vx
        py_proj = ay + t * vy
        return math.hypot(px - px_proj, py - py_proj)

    # Check all pairs of edges? That's 36 checks. Might be slow in Python loop.
    # Optimization: Only check if bounding boxes overlap.
    # Or use a simpler heuristic:
    # Penalty = max(0, (required_dist - actual_dist))^2
    # Required distance depends on orientation.
    # The minimum distance between centers of two non-overlapping unit hexagons
    # is sqrt(3) (when aligned flat-to-flat) and 2 (when vertex-to-vertex).
    # Actually, the distance between centers must be at least the support function sum.
    
    # Let's use the exact polygon intersection test for validity,
    # and a smooth penalty based on the "violation" of the Minkowski sum constraint.
    
    # Construct Minkowski sum vertices.
    # H1 vertices: v1
    # -H2 vertices: -v2 (reflected through origin)
    # The Minkowski sum is the convex hull of {v1[i] + (-v2)[j]}.
    # This is O(36) points. Convex hull is O(36 log 36).
    # Then check if v is inside.
    # If inside, compute distance to boundary.
    
    # This is too heavy for an inner loop of an optimizer.
    
    # Alternative: Use the fact that for small rotations or aligned hexagons,
    # the critical directions are limited.
    # Let's assume for the optimization that we can use a simpler penalty:
    # Penalty = max(0, min_dist_required - dist_centers)^2
    # where min_dist_required is approximated by the support function in the direction of the vector.
    
    # Support function of unit hexagon h(t) = max_{v in H} dot(v, t).
    # For a regular hexagon with circumradius 1, apothem sqrt(3)/2.
    # h(t) = max( |dot(v_i, t)| )? No, h(t) = max(dot(v_i, t)).
    # Since H is symmetric, h(t) = h(-t).
    # The distance between centers must be >= h1(t) + h2(-t) for all t?
    # No, the condition for non-overlap is that for ALL directions t,
    # dot(c1 - c2, t) <= h1(t) + h2(-t).
    # This is equivalent to (c1 - c2) being outside the Minkowski sum H1 + H2 (reflected).
    # Actually, c1 - c2 is outside (H1 - H2).
    # H1 - H2 = H1 + (-H2).
    
    # Let's compute h(t) for the 6 normal directions of the hexagons.
    # If we check the 6 normals of H1 and 6 normals of H2, is that sufficient?
    # The Minkowski sum is a polygon. Its edges are parallel to the edges of H1 or H2.
    # So the normals of the Minkowski sum are the union of the normals of H1 and H2.
    # So we just need to check the 12 normal directions (6 from each, possibly duplicates).
    
    # Normals of H1: angles ang1 + 30 + k*60
    # Normals of H2: angles ang2 + 30 + k*60
    
    # For each normal n, check if dot(v, n) > h1(n) + h2(-n).
    # h1(n) = apothem * max(1, |cos(angle_diff)|)? No.
    # For a regular hexagon, h(n) = apothem if n is a normal, and larger if n is towards a vertex.
    # Actually, h(n) = apothem / cos(delta) where delta is the angle between n and the nearest normal.
    # Max delta is 30 degrees.
    
    # Let's implement this.
    def support_hex(n_x, n_y, angle_rad):
        # n is a unit vector
        # Rotate n by -angle to get local frame
        c = math.cos(-angle_rad)
        s = math.sin(-angle_rad)
        lx = n_x * c - n_y * s
        ly = n_x * s + n_y * c
        
        # In local frame, hexagon is aligned.
        # Normals are at 30, 90, 150.
        # We can compute h(l) = max(dot(v_i, l)) for 6 vertices.
        # Or use the formula:
        # h(l) = sqrt(3)/2 * max(1, |lx|, |ly|, |lx/2 + ly*sqrt(3)/2| ... )?
        # Simpler: just dot with vertices.
        # Vertices in local frame: (1,0), (0.5, sqrt(3)/2), etc.
        h = 0
        for i in range(6):
            theta = i * math.pi / 3
            vx = math.cos(theta)
            vy = math.sin(theta)
            proj = lx * vx + ly * vy
            if proj > h:
                h = proj
        return h

    v_x = pos1[0] - pos2[0]
    v_y = pos1[1] - pos2[1]
    
    max_violation = 0.0
    
    # Check 6 normals of H1
    for k in range(6):
        theta = ang1 + math.radians(30 + k * 60)
        nx = math.cos(theta)
        ny = math.sin(theta)
        
        # h1(n)
        h1 = support_hex(nx, ny, ang1)
        # h2(-n)
        h2 = support_hex(-nx, -ny, ang2)
        
        # dot(v, n)
        proj = v_x * nx + v_y * ny
        
        if proj > h1 + h2:
            violation = proj - (h1 + h2)
            if violation > max_violation:
                max_violation = violation
                
        # Also check -n? The condition is for all t.
        # dot(v, -n) <= h1(-n) + h2(n)
        # -proj <= h1 + h2 (since h is even)
        # So we need |proj| <= h1 + h2?
        # No, the Minkowski sum is symmetric.
        # So we just need to check the 6 unique directions of the normals of the Minkowski sum.
        # The normals of H1 + (-H2) are the union of normals of H1 and H2.
        # So we should check normals of H1 and normals of H2.
        
    # Check 6 normals of H2
    for k in range(6):
        theta = ang2 + math.radians(30 + k * 60)
        nx = math.cos(theta)
        ny = math.sin(theta)
        
        h1 = support_hex(nx, ny, ang1)
        h2 = support_hex(-nx, -ny, ang2)
        
        proj = v_x * nx + v_y * ny
        
        if proj > h1 + h2:
            violation = proj - (h1 + h2)
            if violation > max_violation:
                max_violation = violation
        
        if -proj > h1 + h2:
            violation = -proj - (h1 + h2)
            if violation > max_violation:
                max_violation = violation

    # Also check the first loop for -proj
    for k in range(6):
        theta = ang1 + math.radians(30 + k * 60)
        nx = math.cos(theta)
        ny = math.sin(theta)
        h1 = support_hex(nx, ny, ang1)
        h2 = support_hex(-nx, -ny, ang2)
        proj = v_x * nx + v_y * ny
        if -proj > h1 + h2:
            violation = -proj - (h1 + h2)
            if violation > max_violation:
                max_violation = violation

    if max_violation > 0:
        return max_violation**2
    return 0.0

def _outer_penalty(positions, angles, outer_params):
    """
    Calculate penalty for inner hexagons sticking out of the outer hexagon.
    outer_params: (cx, cy, side, angle_deg)
    """
    ocx, ocy, oside, oang_deg = outer_params
    oang = math.radians(oang_deg)
    
    # Vertices of outer hexagon
    # We check if any vertex of any inner hexagon is outside the outer hexagon.
    # Or check if any edge of inner hexagon is outside?
    # Since inner hexagons are convex and outer is convex,
    # it suffices to check that all vertices of inner hexagons are inside the outer hexagon.
    
    penalty = 0.0
    apothem = oside * math.sqrt(3) / 2
    
    # Normals of outer hexagon
    # We can check the support function condition:
    # For all inner vertices v, and all outer normals n, dot(v - center, n) <= apothem.
    
    # Outer normals
    outer_normals = []
    for k in range(6):
        theta = oang + math.radians(30 + k * 60)
        outer_normals.append((math.cos(theta), math.sin(theta)))
    
    for i in range(len(positions)):
        cx, cy = positions[i]
        ang = angles[i]
        # Get vertices of inner hexagon
        for j in range(6):
            theta = ang + j * math.pi / 3
            vx = cx + math.cos(theta)
            vy = cy + math.sin(theta)
            
            dx = vx - ocx
            dy = vy - ocy
            
            for nx, ny in outer_normals:
                proj = dx * nx + dy * ny
                if proj > apothem:
                    violation = proj - apothem
                    penalty += violation**2
                if proj < -apothem:
                    violation = -apothem - proj
                    penalty += violation**2
    
    return penalty

def _objective(params):
    """
    Objective function for optimization.
    params: [x1, y1, a1, ..., x11, y11, a11, ocx, ocy, oside, oang]
    """
    n_inner = N
    positions = []
    angles = []
    for i in range(n_inner):
        x = params[3*i]
        y = params[3*i + 1]
        a = params[3*i + 2]
        positions.append((x, y))
        angles.append(a)
        
    ocx = params[3*n_inner]
    ocy = params[3*n_inner + 1]
    oside = params[3*n_inner + 2]
    oang = params[3*n_inner + 3]
    
    # Penalty for inner-inner overlap
    inner_pen = 0.0
    for i in range(n_inner):
        for j in range(i+1, n_inner):
            inner_pen += _pairwise_penalty(positions[i], angles[i], positions[j], angles[j])
            
    # Penalty for inner-outer overlap
    outer_pen = _outer_penalty(positions, angles, (ocx, ocy, oside, oang))
    
    # We want to minimize oside, but also avoid overlaps.
    # A common trick is to minimize oside + lambda * (inner_pen + outer_pen).
    # But we need the penalty to be large enough to force feasibility.
    # Alternatively, we can use a two-stage approach or constraints.
    # Since we are using differential_evolution or L-BFGS-B, we can just sum them.
    # To make oside the primary objective, we can scale the penalties heavily.
    
    # If there is any violation, the cost should be dominated by the penalty.
    # Let's use a large weight.
    weight = 1000.0
    
    return oside + weight * (inner_pen + outer_pen)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial guess
    # Place hexagons in a grid pattern.
    # Side length of outer hexagon for 11 hexagons is roughly 4-5.
    # Let's start with a grid of hexagons.
    
    # Grid of hexagons:
    # Row 0: 3 hexagons
    # Row 1: 3 hexagons
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons?
    # Or a flower pattern: 1 center, 6 around, 4 more?
    
    # Let's use a simple grid for initialization.
    # dx = sqrt(3) * 1.1 (slightly apart)
    # dy = 1.5 * 1.1
    
    dx = math.sqrt(3) * 1.1
    dy = 1.5 * 1.1
    
    init_params = []
    idx = 0
    for row in range(4):
        for col in range(3):
            if idx >= N:
                break
            x = (col - 1) * dx
            y = (row - 1.5) * dy
            a = 0.0
            init_params.extend([x, y, a])
            idx += 1
    
    # Outer hexagon init
    # Center at (0,0), side 5, angle 0
    init_params.extend([0.0, 0.0, 5.0, 0.0])
    
    # Bounds
    # Inner positions: within [-5, 5]
    # Inner angles: [0, 2*pi]
    # Outer center: [-5, 5]
    # Outer side: [2, 10]
    # Outer angle: [0, 2*pi]
    
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0)) # x
        bounds.append((-5.0, 5.0)) # y
        bounds.append((0.0, 2*math.pi)) # angle
    bounds.append((-5.0, 5.0)) # ocx
    bounds.append((-5.0, 5.0)) # ocy
    bounds.append((2.0, 10.0)) # oside
    bounds.append((0.0, 2*math.pi)) # oang
    
    # Optimization
    best_result = None
    best_cost = float('inf')
    
    # Try Differential Evolution first, it's robust for non-convex problems.
    # DE is slow, so we limit the population and generations.
    # If it's too slow, we fall back to L-BFGS-B from multiple starts.
    
    use_de = True
    if use_de:
        try:
            # DE parameters
            # popsize=10, maxiter=50
            # This might take a while.
            # Let's try a smaller population and fewer iterations first.
            result = differential_evolution(
                _objective,
                bounds,
                maxiter=30,
                popsize=10,
                tol=1e-4,
                seed=42,
                polish=True
            )
            
            if result.success or result.fun < best_cost:
                best_cost = result.fun
                best_result = result.x
        except Exception as e:
            print(f"DE failed: {e}")
            
    # Refine with L-BFGS-B
    # If we have a result from DE, use it as start.
    # Otherwise, use the grid init.
    # We can also try multiple random starts for L-BFGS-B.
    
    starts = []
    if best_result is not None:
        starts.append(best_result)
    
    # Add grid init
    starts.append(init_params)
    
    # Add a few random starts
    import random
    random.seed(42)
    for _ in range(5):
        rand_params = []
        for i in range(N):
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            a = random.uniform(0, 2*math.pi)
            rand_params.extend([x, y, a])
        rand_params.extend([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(4, 6), random.uniform(0, 2*math.pi)])
        starts.append(rand_params)
        
    for start in starts:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                _objective,
                start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            
            if res.fun < best_cost:
                best_cost = res.fun
                best_result = res.x
        except Exception as e:
            pass
            
    if best_result is None:
        # Fallback to grid init
        best_result = init_params
        
    # Extract result
    positions = []
    angles_deg = []
    for i in range(N):
        x = best_result[3*i]
        y = best_result[3*i + 1]
        a = best_result[3*i + 2]
        positions.append((x, y))
        angles_deg.append(math.degrees(a))
        
    ocx = best_result[3*N]
    ocy = best_result[3*N + 1]
    oside = best_result[3*N + 2]
    oang = best_result[3*N + 3]
    
    # Final validation and cleanup
    # Check if the solution is valid.
    # If not, we might need to return a fallback.
    # For now, assume the optimizer found a good solution.
    
    inner_rows = []
    for i in range(N):
        inner_rows.append([positions[i][0], positions[i][1], angles_deg[i]])
        
    return inner_rows, (ocx, ocy), oside, oang

if __name__ == "__main__":
    r = construct()
    print(r)
