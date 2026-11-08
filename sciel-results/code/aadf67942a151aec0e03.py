import numpy as np
from scipy.optimize import minimize
import time
import math
import random
import itertools

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _min_center_distance(a1, a2):
    """
    Minimum center-to-center distance for two unit regular hexagons with angles a1, a2 (degrees)
    to be non-overlapping. Use conservative bound: sum of circumradii = 2.0
    but tighter: for two unit hexagons, the minimum distance depends on relative angle.
    For simplicity, use 2*apothem = sqrt(3) when aligned, up to 2.0 when vertex-to-vertex.
    We use a safe lower bound that guarantees no overlap: 2.0 is always safe.
    But for tighter packing, we can use the actual minimum.
    """
    # For two unit hexagons, the minimum distance between centers for non-overlap
    # depends on relative rotation. The worst case (minimum distance) is when they
    # are aligned: distance = sqrt(3) (edge-to-edge touching).
    # The maximum distance needed is 2.0 (vertex-to-vertex).
    # For safety, we'll compute based on relative angle.
    da = abs(a1 - a2) % 60
    da = min(da, 60 - da)  # Symmetric around 0
    # The support function of a unit hexagon in direction theta:
    # For a hexagon with vertex at angle 0, the support in direction phi is:
    # max over vertices of v·u where u = (cos phi, sin phi)
    # This is piecewise linear. The minimum distance for non-overlap is:
    # min over all directions u of (support_1(u) + support_2(-u))
    # For two identical hexagons with relative rotation delta, this is:
    # 2 * min over u of support_hex(u) where support accounts for rotation
    # Actually, the minimum center distance for non-overlap of two convex bodies A, B is:
    # min over u of (h_A(u) + h_B(-u)) where h is the support function
    # For a unit regular hexagon, h(u) = max_k (v_k · u) where v_k are vertices
    # With rotation, v_k is rotated by the angle.
    
    # For a unit hexagon with vertex at angle alpha, vertices are at alpha + k*pi/3
    # h(u) = max_k cos(u_angle - alpha - k*pi/3)
    # This equals cos(min_k |u_angle - alpha - k*pi/3|) for the closest vertex direction
    # But actually it's the max of cosines, which is cos of the minimum angular distance
    
    # The minimum distance is min over u of (h1(u) + h2(-u))
    # h1(u) = max_k cos(u - a1 - k*pi/3)
    # h2(-u) = max_k cos(-u - a2 - k*pi/3) = max_k cos(u + a2 + k*pi/3)
    
    # For numerical simplicity and safety, use a lookup or conservative bound.
    # The minimum possible distance is sqrt(3) ≈ 1.732 (edge-to-edge, aligned)
    # The maximum is 2.0 (vertex-to-vertex)
    # We'll use: distance = sqrt(3) + (2.0 - sqrt(3)) * (da/30)
    # This is a linear interpolation, not exact but safe enough for initialization.
    # Actually, let's just use 2.0 for safety in the relaxation, and rely on SAT for final check.
    return 2.0  # Conservative: guarantees no overlap

def _hexagons_overlap_fast(c1, a1, c2, a2, eps=1e-9):
    """
    Fast check if two unit hexagons overlap using center distance.
    c1, c2 are (x,y) centers, a1, a2 are angles in degrees.
    Returns True if they overlap.
    """
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    dist = math.sqrt(dx*dx + dy*dy)
    
    # Compute minimum center distance for non-overlap
    # For two unit hexagons, the minimum distance depends on relative angle
    # Use a conservative but tighter bound than 2.0
    da = abs(a1 - a2) % 60
    da = min(da, 60 - da)
    
    # The support function of a unit hexagon with vertex at angle 0:
    # h(phi) = max_k cos(phi - k*pi/3) for k=0..5
    # This equals cos(min_k |phi - k*pi/3|_wrapped)
    # where |x|_wrapped = min(|x|, 2*pi - |x|)
    
    # For two hexagons with relative rotation delta = da in radians:
    # min_dist = min over phi of (h(phi) + h(phi + pi + delta))
    # where h is the support function of a unit hexagon with vertex at 0
    
    # Let's compute this numerically for the given da
    delta = math.radians(da)
    
    # h(phi) for unit hexagon with vertex at angle 0:
    # vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # h(phi) = max(cos(phi), cos(phi-pi/3), cos(phi-2pi/3), cos(phi-pi), cos(phi-4pi/3), cos(phi-5pi/3))
    # = max over k of cos(phi - k*pi/3)
    
    # We need min over phi of h(phi) + h(phi + pi + delta)
    # Note: h(phi + pi) = h(-phi) by symmetry of hexagon
    # So h(phi + pi + delta) = h(-phi + delta)
    
    # For numerical computation, sample phi
    min_dist = float('inf')
    for k in range(360):
        phi = math.radians(k)
        # h(phi)
        h1 = max(math.cos(phi - m * math.pi / 3) for m in range(6))
        # h(phi + pi + delta) = h(-(phi) + delta) by symmetry
        phi2 = -phi + delta
        h2 = max(math.cos(phi2 - m * math.pi / 3) for m in range(6))
        d = h1 + h2
        if d < min_dist:
            min_dist = d
    
    return dist < min_dist - eps

def _check_overlap_sat(v1, v2, eps=1e-9):
    """
    Check if two hexagons (as vertex arrays) have overlapping interiors using SAT.
    Returns True if they overlap.
    """
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        edge = p2 - p1
        n = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n = n / norm
        
        proj1_min = np.min(v1 @ n)
        proj1_max = np.max(v1 @ n)
        proj2_min = np.min(v2 @ n)
        proj2_max = np.max(v2 @ n)
        
        if proj1_max <= proj2_min + eps or proj2_max <= proj1_min + eps:
            return False
    
    return True

def _point_in_hex(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px,py) is inside regular hexagon centered at (cx,cy)."""
    angle = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    for i in range(6):
        theta_n = i * math.pi / 3 + math.pi / 6
        nx = math.cos(theta_n)
        ny = math.sin(theta_n)
        if nx * lx + ny * ly > side * math.sqrt(3) / 2 + 1e-9:
            return False
    return True

def _enclosing_side(positions, outer_center, outer_angle):
    """
    Compute the minimum side length of a regular hexagon centered at outer_center
    with rotation outer_angle that contains all inner hexagons (each of side 1).
    
    For each inner hexagon with center c and angle a_inner, and for each outer edge
    normal n_k, the constraint is:
    n_k · c + support_inner(n_k, a_inner) <= s * sqrt(3) / 2
    
    where support_inner(n_k, a_inner) is the support function of the unit hexagon
    with rotation a_inner in direction n_k.
    """
    angle = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = angle + math.pi / 6 + i * math.pi / 3
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_val = 0.0
    for pos in positions:
        x, y, ang = pos
        inner_angle = math.radians(ang)
        for nx, ny in normals:
            # Support function of unit hexagon with vertex at angle inner_angle in direction (nx, ny)
            # = max over vertices v of v · (nx, ny)
            # Vertices are at inner_angle + k*pi/3, k=0..5
            # v_k = (cos(inner_angle + k*pi/3), sin(inner_angle + k*pi/3))
            # v_k · (nx, ny) = cos(inner_angle + k*pi/3 - atan2(ny, nx))
            n_angle = math.atan2(ny, nx)
            support = max(math.cos(inner_angle + k * math.pi / 3 - n_angle) for k in range(6))
            
            val = nx * x + ny * y + support
            if val > max_val:
                max_val = val
    
    return 2.0 * max_val / math.sqrt(3)

def _optimize_outer(positions, time_limit=5.0):
    """Optimize the outer hexagon center and angle to minimize side length."""
    start = time.time()
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Compute centroid of all inner hexagon centers
    centers = np.array([[p[0], p[1]] for p in positions])
    centroid = np.mean(centers, axis=0)
    
    # Try different angles
    for angle_deg in range(0, 180, 10):
        if time.time() - start > time_limit:
            break
        
        angle = math.radians(angle_deg)
        normals = []
        for i in range(6):
            theta = angle + math.pi / 6 + i * math.pi / 3
            normals.append(np.array([math.cos(theta), math.sin(theta)]))
        
        # For this angle, we need to find center c that minimizes
        # max over all inner hexagons and all normals of (n · c_inner + support(n, inner_angle))
        # subject to: n_k · c + support_k <= s * sqrt(3) / 2
        # i.e., s = (2/sqrt(3)) * max_k (n_k · c + support_k)
        # where support_k = max over inner hexagons of (n_k · c_inner + support_inner(n_k, a_inner))
        
        # Wait, the center c of the outer hexagon shifts all constraints:
        # n_k · (v - c) <= s * sqrt(3) / 2 for all vertices v
        # n_k · v - n_k · c <= s * sqrt(3) / 2
        # s >= (2/sqrt(3)) * (n_k · v - n_k · c)
        
        # For each inner hexagon, the max over its vertices of n_k · v is:
        # n_k · c_inner + support_inner(n_k, a_inner)
        
        # So: s >= (2/sqrt(3)) * max over inner hexagons of (n_k · c_inner + support_inner(n_k, a_inner) - n_k · c)
        
        # To minimize s, we want to maximize n_k · c for the worst-case k.
        # This is: minimize s subject to s * sqrt(3)/2 + n_k · c >= n_k · c_inner + support_k for all k
        
        # Let's solve this with scipy
        def objective(params):
            return params[2]
        
        def constraints_fn(params):
            cx, cy, s = params
            cons = []
            for k, n in enumerate(normals):
                # Compute max over inner hexagons of (n · c_inner + support_inner(n, a_inner))
                max_proj = float('-inf')
                for pos in positions:
                    x, y, ang = pos
                    inner_angle = math.radians(ang)
                    n_angle = math.atan2(n[1], n[0])
                    support = max(math.cos(inner_angle + m * math.pi / 3 - n_angle) for m in range(6))
                    proj = n[0] * x + n[1] * y + support
                    if proj > max_proj:
                        max_proj = proj
                
                # Constraint: s * sqrt(3)/2 + n[0]*cx + n[1]*cy >= max_proj
                val = s * math.sqrt(3) / 2 + n[0] * cx + n[1] * cy - max_proj
                cons.append({'type': 'ineq', 'fun': lambda val=val: val})
            return cons
        
        x0 = np.array([centroid[0], centroid[1], 5.0])
        
        try:
            result = minimize(objective, x0, method='SLSQP', constraints=constraints_fn,
                             options={'maxiter': 30, 'ftol': 1e-6})
            if result.success and result.fun < best_side:
                best_side = result.fun
                best_center = (result.x[0], result.x[1])
                best_angle = angle_deg
        except:
            pass
    
    if best_side == float('inf'):
        # Fallback
        best_center = (float(centroid[0]), float(centroid[1]))
        best_angle = 0.0
        best_side = _enclosing_side(positions, best_center, best_angle)
    
    return best_center, best_side, best_angle

def _check_validity(positions, outer_center, outer_side, outer_angle, eps=1e-7):
    """Check all inner hexagons are inside outer and mutually disjoint."""
    # Check all inside outer using the proper containment test
    angle = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = angle + math.pi / 6 + i * math.pi / 3
        normals.append((math.cos(theta), math.sin(theta)))
    
    for pos in positions:
        x, y, ang = pos
        inner_angle = math.radians(ang)
        for nx, ny in normals:
            n_angle = math.atan2(ny, nx)
            support = max(math.cos(inner_angle + k * math.pi / 3 - n_angle) for k in range(6))
            val = nx * (x - outer_center[0]) + ny * (y - outer_center[1]) + support
            if val > outer_side * math.sqrt(3) / 2 + eps:
                return False
    
    # Check mutual disjointness using fast check
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            c1 = (positions[i][0], positions[i][1])
            c2 = (positions[j][0], positions[j][1])
            if _hexagons_overlap_fast(c1, positions[i][2], c2, positions[j][2]):
                # Double-check with SAT
                v1 = _hex_vertices(positions[i][0], positions[i][1], 1.0, positions[i][2])
                v2 = _hex_vertices(positions[j][0], positions[j][1], 1.0, positions[j][2])
                if _check_overlap_sat(v1, v2):
                    return False
    
    return True

def _force_directed_relax(positions, max_iter=100):
    """Force-directed relaxation to resolve overlaps."""
    positions = [list(p) for p in positions]
    
    for iteration in range(max_iter):
        moved = False
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                c1 = (positions[i][0], positions[i][1])
                c2 = (positions[j][0], positions[j][1])
                
                if _hexagons_overlap_fast(c1, positions[i][2], c2, positions[j][2]):
                    # Push apart along the line connecting centers
                    dx = positions[j][0] - positions[i][0]
                    dy = positions[j][1] - positions[i][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 1e-9:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    
                    # Compute required separation
                    da = abs(positions[i][2] - positions[j][2]) % 60
                    da = min(da, 60 - da)
                    delta = math.radians(da)
                    min_dist = float('inf')
                    for k in range(360):
                        phi = math.radians(k)
                        h1 = max(math.cos(phi - m * math.pi / 3) for m in range(6))
                        phi2 = -phi + delta
                        h2 = max(math.cos(phi2 - m * math.pi / 3) for m in range(6))
                        d = h1 + h2
                        if d < min_dist:
                            min_dist = d
                    
                    if dist < min_dist:
                        push = (min_dist - dist) / 2.0
                        positions[i][0] -= dx * push
                        positions[i][1] -= dy * push
                        positions[j][0] += dx * push
                        positions[j][1] += dy * push
                        moved = True
        
        if not moved:
            break
    
    return [tuple(p) for p in positions]

def _get_valid_fallback():
    """Return a valid arrangement with large outer hexagon."""
    positions = []
    for i in range(N):
        x = i * 2.5
        y = 0.0
        positions.append([x, y, 0.0])
    
    # Compute proper enclosing side
    outer_center = (sum(p[0] for p in positions) / N, 0.0)
    outer_side = _enclosing_side(positions, outer_center, 0.0) + 0.1
    outer_angle = 0.0
    
    return positions, outer_center, outer_side, outer_angle

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    sqrt3 = math.sqrt(3)
    
    # Initial arrangement: hexagonal lattice pattern
    d = sqrt3
    dy_row = d * sqrt3 / 2
    
    positions = []
    for i in range(4):
        positions.append([i * d - 1.5 * d, 0.0, 0.0])
    for i in range(3):
        positions.append([i * d - d, dy_row, 0.0])
    for i in range(4):
        positions.append([i * d - 1.5 * d, 2 * dy_row, 0.0])
    
    # Force-directed relaxation
    positions = _force_directed_relax(positions, max_iter=100)
    
    # Optimize outer hexagon
    outer_center, outer_side, outer_angle = _optimize_outer(positions, time_limit=10.0)
    
    # Check validity
    if not _check_validity(positions, outer_center, outer_side, outer_angle):
        # Try with slightly larger side
        outer_side += 0.01
        if not _check_validity(positions, outer_center, outer_side, outer_angle):
            return _get_valid_fallback()
    
    best_positions = [list(p) for p in positions]
    best_score = outer_side
    
    # Simulated annealing
    remaining_time = deadline - time.time()
    if remaining_time > 15.0:
        max_iter = min(3000, int(remaining_time * 50))
        
        current_positions = [list(p) for p in best_positions]
        current_score = best_score
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10.0:
                break
            
            t = 0.5 * (0.01 / 0.5) ** (iteration / max_iter)
            
            new_positions = [list(p) for p in current_positions]
            idx = random.randint(0, len(new_positions) - 1)
            new_positions[idx][0] += random.gauss(0, t * 0.3)
            new_positions[idx][1] += random.gauss(0, t * 0.3)
            new_positions[idx][2] += random.gauss(0, t * 5)
            
            nc, ns, na = _optimize_outer(new_positions, time_limit=2.0)
            
            if _check_validity(new_positions, nc, ns, na):
                new_score = ns
            else:
                new_score = ns + 5.0
            
            if new_score < current_score:
                current_positions = new_positions
                current_score = new_score
                if new_score < best_score:
                    best_positions = [list(p) for p in new_positions]
                    best_score = new_score
                    outer_center, outer_side, outer_angle = nc, ns, na
            else:
                delta = new_score - current_score
                if delta < 0 or random.random() < math.exp(-delta / max(t, 1e-10)):
                    current_positions = new_positions
                    current_score = new_score
        
        # Recompute outer for best
        outer_center, outer_side, outer_angle = _optimize_outer(best_positions, time_limit=5.0)
        if not _check_validity(best_positions, outer_center, outer_side, outer_angle):
            outer_side += 0.01
            if not _check_validity(best_positions, outer_center, outer_side, outer_angle):
                return _get_valid_fallback()
    
    # Final safety check
    if not _check_validity(best_positions, outer_center, outer_side, outer_angle):
        return _get_valid_fallback()
    
    return best_positions, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-END
