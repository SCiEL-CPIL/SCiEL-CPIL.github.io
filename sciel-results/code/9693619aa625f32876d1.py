"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side length and rotation."""
    angle = np.deg2rad(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside or on boundary of hexagon."""
    dx, dy = px - cx, py - cy
    angle = -np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    s = side
    sqrt3 = np.sqrt(3)
    # Hexagon with flat top (angle 0) has vertices at (s,0), (s/2, s*sqrt3/2)...
    # The inequalities for a hexagon with vertices at angles 0, 60, ... are:
    # |x| <= s
    # |y| <= s*sqrt3/2
    # |x*sqrt3/2 + y/2| <= s
    # |x*sqrt3/2 - y/2| <= s
    # Wait, let's verify.
    # Vertex 0: (s, 0).
    # Vertex 1: (s/2, s*sqrt3/2).
    # Edge 0-1 vector: (-s/2, s*sqrt3/2). Normal: (s*sqrt3/2, s/2) -> (sqrt3/2, 1/2).
    # Dot product of point (x,y) with normal <= s * |normal|? No.
    # The half-width in direction (sqrt3/2, 1/2) is s.
    # So |x*sqrt3/2 + y/1/2|? No.
    # Let's use the standard form for a regular hexagon with circumradius R=s.
    # The apothem is a = s * sqrt(3)/2.
    # The hexagon is the intersection of 6 half-planes.
    # For a hexagon with vertices at angles k*pi/3:
    # The supporting lines are at distance a from the center, with normals at angles k*pi/3 + pi/6?
    # Let's just use the convex hull check or simple inequalities.
    # For angle 0 (flat top/bottom? No, vertex at right):
    # Vertices: (s,0), (s/2, s*sqrt3/2), (-s/2, s*sqrt3/2), (-s,0), (-s/2, -s*sqrt3/2), (s/2, -s*sqrt3/2).
    # Inequalities:
    # 1. y <= s*sqrt3/2
    # 2. y >= -s*sqrt3/2
    # 3. x*sqrt3/2 + y/2 <= s
    # 4. x*sqrt3/2 - y/2 <= s
    # 5. -x*sqrt3/2 + y/2 <= s
    # 6. -x*sqrt3/2 - y/2 <= s
    # Let's check vertex (s,0):
    # 3: s*sqrt3/2 <= s ? No. s*0.866 <= s. True.
    # Wait, the max value of x*sqrt3/2 + y/2 on the hexagon is s.
    # At (s,0): s*sqrt3/2. This is < s.
    # At (s/2, s*sqrt3/2): s/2*sqrt3/2 + s*sqrt3/2/2 = s*sqrt3/4 + s*sqrt3/4 = s*sqrt3/2.
    # So the constraint is |x*sqrt3/2 + y/2| <= s*sqrt3/2? No.
    # Let's calculate the distance from origin to the line passing through (s,0) and (s/2, s*sqrt3/2).
    # Line eq: y - 0 = (s*sqrt3/2 - 0) / (s/2 - s) * (x - s) => y = (-sqrt3) (x - s) => sqrt3 x + y = s*sqrt3.
    # Distance from origin: |0 + 0 - s*sqrt3| / sqrt(3+1) = s*sqrt3 / 2.
    # So the inequality is sqrt3*x + y <= s*sqrt3.
    # Similarly for other sides.
    # The 6 inequalities for a hexagon with vertex at (s,0) are:
    # 1. sqrt3*x + y <= s*sqrt3
    # 2. -sqrt3*x + y <= s*sqrt3
    # 3. -y <= s*sqrt3/2  (Top flat edge? No, vertex is at 60 deg. Top edge is between 60 and 120. y = s*sqrt3/2)
    #    So y <= s*sqrt3/2.
    # 4. y >= -s*sqrt3/2
    # 5. sqrt3*x - y <= s*sqrt3
    # 6. -sqrt3*x - y <= s*sqrt3
    
    # Let's re-verify with vertices.
    # V0: (s, 0).
    # 1: sqrt3*s + 0 = s*sqrt3. OK.
    # 2: -sqrt3*s + 0 = -s*sqrt3 <= s*sqrt3. OK.
    # 3: 0 <= s*sqrt3/2. OK.
    # 4: 0 >= -s*sqrt3/2. OK.
    # 5: sqrt3*s - 0 = s*sqrt3. OK.
    # 6: -sqrt3*s - 0 = -s*sqrt3 <= s*sqrt3. OK.
    
    # V1: (s/2, s*sqrt3/2).
    # 1: sqrt3*(s/2) + s*sqrt3/2 = s*sqrt3. OK.
    # 3: s*sqrt3/2 <= s*sqrt3/2. OK.
    
    # So the constraints are:
    # abs(lx * sqrt3 + ly) <= s * sqrt3
    # abs(-lx * sqrt3 + ly) <= s * sqrt3
    # abs(ly) <= s * sqrt3 / 2
    # This simplifies to:
    # |ly| <= s*sqrt3/2
    # |lx*sqrt3 + ly| <= s*sqrt3
    # |-lx*sqrt3 + ly| <= s*sqrt3
    
    return (np.abs(ly) <= s * sqrt3 / 2 and
            np.abs(lx * sqrt3 + ly) <= s * sqrt3 and
            np.abs(-lx * sqrt3 + ly) <= s * sqrt3)

def check_hexagon_overlap(v1, v2):
    """
    Check if two convex polygons (hexagons) overlap using Separating Axis Theorem.
    v1, v2 are Nx2 arrays of vertices.
    Returns True if they overlap.
    """
    axes = []
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i+1)%6]
        edge = p2 - p1
        # Normal to edge
        normal = np.array([-edge[1], edge[0]])
        axes.append(normal)
        p1 = v2[i]
        p2 = v2[(i+1)%6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        axes.append(normal)
        
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return False
            
    return True

def is_valid_configuration(inner_positions, outer_center, outer_side, outer_angle):
    """
    Check if the configuration is valid:
    1. All inner hexagons are inside the outer hexagon.
    2. All inner hexagons are disjoint.
    """
    outer_verts = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    inner_verts_list = []
    
    # 1. Containment
    for k in range(N):
        ix, iy, iangle = inner_positions[k]
        iv = hex_vertices(ix, iy, 1.0, iangle)
        inner_verts_list.append(iv)
        for v in iv:
            if not point_in_hexagon(v[0], v[1], outer_center[0], outer_center[1], outer_side, outer_angle):
                return False
                
    # 2. Disjointness
    for i in range(N):
        for j in range(i+1, N):
            if check_hexagon_overlap(inner_verts_list[i], inner_verts_list[j]):
                return False
                
    return True

def construct():
    """Place 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: Hexagonal packing of 12 hexagons.
    # Center (0,0).
    # Ring 1: 6 hexagons at distance sqrt(3) (tangent).
    # Ring 2: 5 hexagons? 
    # Actually, a compact cluster of 12 hexagons can be formed by:
    # 1 in center.
    # 6 in first ring (distance sqrt(3)).
    # 5 in second ring? 
    # Or just a 3x4 grid rotated?
    # Let's use a hexagonal grid.
    # Axial coordinates (q, r).
    # Distance between centers is sqrt(3).
    # We want 12 hexagons.
    # A "flower" of 7 is 1+6.
    # Add 5 more.
    
    # Let's generate positions on a hex grid.
    # Spacing d = sqrt(3).
    d = np.sqrt(3)
    positions = []
    # Center
    positions.append([0.0, 0.0, 0.0])
    # Ring 1 (6 hexes)
    for i in range(6):
        angle = i * np.pi / 3
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        positions.append([x, y, 0.0])
    # Ring 2 (5 hexes) - place them at the "corners" of the ring 1 to maximize density?
    # Or just continue the grid.
    # Axial coords for ring 2 are (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1).
    # Let's pick 5 that fit well.
    # Actually, let's just use a rectangular grid for simplicity in initialization, then optimize.
    # A 3x4 grid of hexagons (aligned) has width 3*d + 1? No.
    # Width of 3 hexagons side-by-side: 2*d + 2*s? No.
    # Center to center is d.
    # Total width of 3 centers: 2*d. Plus radius 1 on each side: 2*d + 2.
    # Total height of 4 centers: 3*d. Plus radius 1 on each side: 3*d + 2.
    
    # Let's stick to the radial packing which is more symmetric.
    # Ring 2 positions in axial coords (q,r):
    # (2,0), (1,1), (0,2), (-1,1), (-2,0) -> 5 positions.
    # Convert axial to cartesian:
    # x = d * (q + r/2)
    # y = d * (sqrt(3)/2 * r)
    
    axial_ring2 = [(2,0), (1,1), (0,2), (-1,1), (-2,0)]
    for q, r in axial_ring2:
        x = d * (q + r/2.0)
        y = d * (np.sqrt(3)/2 * r)
        positions.append([x, y, 0.0])
        
    # Now we have 12 positions.
    # We will optimize the outer hexagon to fit these, and slightly adjust inner positions.
    
    best_score = 1e10
    best_inner = None
    best_outer = None
    
    # Coarse search over outer hexagon angle and size
    # The bounding box of the inner cluster is roughly:
    # Max radius from center:
    # Ring 1: d.
    # Ring 2: distance for (2,0) is 2*d.
    # So max center distance is 2*d = 2*sqrt(3) approx 3.46.
    # Plus circumradius 1. So max extent approx 4.46.
    # Outer hexagon side S must be large enough.
    # Apothem of outer = S*sqrt(3)/2.
    # We need apothem >= max extent in normal direction.
    
    # Let's define a function to compute the minimum outer side length required for a given set of inner hexes and outer angle.
    def min_outer_side(inner_pos, outer_angle):
        # For a fixed outer angle and center (assumed 0,0 for now, we can optimize center later),
        # the side length S is determined by the maximum "distance" of any inner vertex from the center
        # projected onto the 6 normals of the outer hexagon.
        # Outer hexagon with side S, angle A, center (0,0).
        # Normals are at angles A + pi/6 + k*pi/3.
        # The constraint is: dot(vertex, normal) <= S * sqrt(3) / 2 for all vertices.
        # So S >= max(dot(vertex, normal)) * 2 / sqrt(3).
        
        max_val = 0.0
        for k in range(N):
            ix, iy, iangle = inner_pos[k]
            iv = hex_vertices(ix, iy, 1.0, iangle)
            for v in iv:
                # Project onto all 6 normals
                for m in range(6):
                    normal_angle = np.deg2rad(outer_angle) + np.pi/6 + m * np.pi/3
                    nx = np.cos(normal_angle)
                    ny = np.sin(normal_angle)
                    dot = v[0]*nx + v[1]*ny
                    if dot > max_val:
                        max_val = dot
        return max_val * 2 / np.sqrt(3)
        
    # We will iterate over a few outer angles.
    # And for each, we will run a local optimization on inner positions to minimize the required outer side.
    
    def objective_inner(params, outer_angle):
        # params: 12*3 (x,y,angle) for inner hexes
        # We fix outer angle, optimize inner positions to minimize required outer side.
        inner_pos = params.reshape(N, 3)
        
        # Check disjointness penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                ix1, iy1, _ = inner_pos[i]
                ix2, iy2, _ = inner_pos[j]
                dist = np.sqrt((ix2-ix1)**2 + (iy2-iy1)**2)
                if dist < np.sqrt(3):
                    penalty += 100 * (np.sqrt(3) - dist)
                    
        if penalty > 0:
            return 1e10 + penalty
            
        # Calculate required outer side
        side = min_outer_side(inner_pos, outer_angle)
        return side
        
    # Try different outer angles
    for outer_angle_deg in [0, 30, 60, 90, 120, 150]:
        if time.time() > deadline - 20:
            break
            
        # Initial guess for inner positions
        p0 = np.array(positions).flatten()
        
        try:
            # Optimize inner positions
            res = minimize(objective_inner, p0, args=(outer_angle_deg,), method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.success or res.fun < best_score:
                inner_opt = res.x.reshape(N, 3)
                side = min_outer_side(inner_opt, outer_angle_deg)
                
                # Now we have a candidate inner config and a required side length.
                # We can also try to shift the outer hexagon center to reduce side length further?
                # For now, assume center is 0,0.
                # Let's verify validity.
                if is_valid_configuration(inner_opt, (0.0, 0.0), side, outer_angle_deg):
                    if side < best_score:
                        best_score = side
                        best_inner = inner_opt
                        best_outer = (0.0, 0.0, side, outer_angle_deg)
                        
        except Exception:
            pass
            
    # If we didn't find a valid one, fallback to a very large hexagon
    if best_inner is None:
        best_inner = np.array(positions)
        best_outer = (0.0, 0.0, 10.0, 0.0)
        
    # Final validation and return
    inner_list = [[float(best_inner[k,0]), float(best_inner[k,1]), float(best_inner[k,2])] for k in range(N)]
    ocx, ocy, oside, oang = best_outer
    
    # Ensure strict validity before returning
    if not is_valid_configuration(inner_list, (ocx, ocy), oside, oang):
        # Emergency fallback: increase size
        oside *= 1.5
        # Re-check
        if not is_valid_configuration(inner_list, (ocx, ocy), oside, oang):
             # Just return something huge
             return [[0.0, 0.0, 0.0]]*N, (0.0, 0.0), 50.0, 0.0

    return inner_list, (float(ocx), float(ocy)), float(oside), float(oang)
