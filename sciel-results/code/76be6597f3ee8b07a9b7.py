import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 148.0  # safety margin before 150s kill
    
    # 1. Define the inner hexagons in a rigid 3-4-3 hexagonal cluster
    # This is a known optimal packing for 11 hexagons (or close to it)
    # Spacing between centers of touching same-orientation hexagons is 2*apothem = sqrt(3)
    sp = math.sqrt(3)
    
    # Coordinates for 3-4-3 cluster
    # Row 0 (3 hexes): y = 0, x = -sp, 0, sp
    # Row 1 (4 hexes): y = 1.5, x = -1.5*sp, -0.5*sp, 0.5*sp, 1.5*sp  (offset by 0.5*sp)
    # Wait, standard hex grid:
    # If flat-top:
    #   distance between centers = 2 * apothem = sqrt(3)
    #   vertical distance between rows = 1.5 * side = 1.5 (for side 1)
    #   horizontal offset = sqrt(3)/2 * side = sqrt(3)/2
    # Let's use pointy-top for standard grid alignment or just define coords carefully.
    # Let's use a simple grid generation for 3-4-3.
    # Side length = 1.
    # Apothem = sqrt(3)/2.
    # Distance between centers of touching hexes = 2 * apothem = sqrt(3).
    
    # Let's place them in a hexagonal lattice.
    # Center at (0,0).
    # Neighbors at distance sqrt(3) in directions 0, 60, 120, 180, 240, 300.
    # Let's generate 11 points in a 3-4-3 pattern.
    # Row 0: 3 points
    # Row 1: 4 points
    # Row 2: 3 points
    
    # Using axial coordinates (q, r) for hex grid:
    # Pointy top:
    # x = size * sqrt(3) * (q + r/2)
    # y = size * 3/2 * r
    # Here size = 1 (side length).
    # Distance between adjacent centers is sqrt(3).
    
    # 3-4-3 pattern in axial coords (q, r):
    # r=0: q in [-1, 0, 1]
    # r=1: q in [-2, -1, 0, 1]  (Wait, standard hex grid adjacency)
    # Let's just hardcode the Cartesian coordinates for a tight 3-4-3 cluster.
    
    # Let's use the "flower" pattern which is 1+6+4? No, 11 is 1+6+4.
    # A 3-4-3 block is 11 hexes.
    
    # Let's calculate positions for a 3x4x3 block of hexes.
    # Actually, a compact cluster of 11 hexes is often the "hexagon of hexagons" of radius 2 minus 5 corners?
    # Radius 1 hex cluster has 7 hexes.
    # Radius 2 hex cluster has 19 hexes.
    # 11 hexes is between.
    
    # Let's stick to the 3-4-3 rows.
    # Row 0 (y=0): x = -sqrt(3), 0, sqrt(3)
    # Row 1 (y=1.5): x = -1.5*sqrt(3), -0.5*sqrt(3), 0.5*sqrt(3), 1.5*sqrt(3)
    # Row 2 (y=3.0): x = -sqrt(3), 0, sqrt(3)
    
    # Check distances:
    # (0,0) to (-0.5*sqrt(3), 1.5): dx = -0.5*sqrt(3), dy = 1.5.
    # dist^2 = 3/4 + 9/4 = 12/4 = 3. dist = sqrt(3). Correct.
    # (0,0) to (sqrt(3), 0): dist = sqrt(3). Correct.
    
    inner = []
    # Row 0
    for q in [-1, 0, 1]:
        x = q * sp
        y = 0.0
        inner.append([x, y, 0.0])
        
    # Row 1
    for q in [-1.5, -0.5, 0.5, 1.5]:
        x = q * sp
        y = 1.5
        inner.append([x, y, 0.0])
        
    # Row 2
    for q in [-1, 0, 1]:
        x = q * sp
        y = 3.0
        inner.append([x, y, 0.0])
        
    # We have 11 hexagons.
    # All are oriented at 0 degrees (flat-top or pointy-top? 
    # The vertex calculation in the seed program uses:
    # theta = a + 60*i. 
    # If a=0, vertices are at 0, 60, 120... which is pointy-top?
    # cos(0)=1, sin(0)=0 -> (1,0).
    # cos(60)=0.5, sin(60)=0.866.
    # This is a "pointy-right" or similar orientation.
    # The grid spacing assumed above (sqrt(3) horizontal, 1.5 vertical) works for 
    # hexagons that share edges.
    # If hexagon A is at (0,0) and B is at (sqrt(3), 0), do they touch?
    # Vertices of A (side 1): (1,0), (0.5, 0.866), (-0.5, 0.866), (-1,0), (-0.5,-0.866), (0.5,-0.866).
    # Vertices of B (centered at sqrt(3), 0): (sqrt(3)+1, 0), ...
    # Distance between centers is sqrt(3) approx 1.732.
    # Sum of apothems is 2 * (sqrt(3)/2) = sqrt(3).
    # So they touch at the edge midpoint if aligned correctly.
    # The orientation 0 degrees means the "flat" sides are not horizontal/vertical necessarily.
    # Vertex at 0 deg is (1,0). The edge opposite is at 180 deg (-1,0).
    # The "width" in x direction is 2. The "height" in y is sqrt(3).
    # If we place B at (sqrt(3), 0), the leftmost point of B is sqrt(3)-1 approx 0.732.
    # The rightmost point of A is 1.
    # They overlap! 1 > 0.732.
    # So the spacing sqrt(3) is for the *apothem* distance, meaning the *edges* touch.
    # But for pointy-top (vertex at 0), the horizontal distance between centers for touching is 2 * apothem?
    # No, for hexagons to touch edge-to-edge, the distance between centers is 2 * apothem = sqrt(3).
    # This assumes the edges are parallel.
    # If orientation is 0 for both, the edges are parallel.
    # The edge normal is at 30, 90, etc?
    # Vertices at 0, 60...
    # Edges are between 0 and 60. Midpoint angle 30.
    # Normal to edge is at 30.
    # Distance from center to edge is apothem = sqrt(3)/2.
    # So if we move center by 2 * apothem in direction 30, the edges touch.
    # My grid used horizontal movement.
    # If I move horizontally (angle 0), the vertices are at 0 and 180.
    # Distance between centers must be 2 (vertex to vertex) to touch? No, that's too far.
    # They would touch at the vertex if distance is 2? No, vertices are sharp.
    # For regular hexagons, the minimum distance between centers for disjointness is 2 * apothem = sqrt(3) IF the hexagons are rotated such that a flat side faces the other.
    # In my grid, Row 0 y=0, Row 1 y=1.5.
    # Vector (0.5*sqrt(3), 1.5). Length sqrt(3/4 + 9/4) = sqrt(3).
    # Angle of this vector: atan2(1.5, 0.866) = 60 degrees.
    # Does a flat side face 60 degrees?
    # Vertices at 0, 60. Edge is between 0 and 60. Midpoint 30. Normal 30.
    # Next edge 60-120. Midpoint 90. Normal 90.
    # There is no flat side facing 60 degrees. There is a vertex at 60 degrees.
    # So the hexagons in my grid (spacing sqrt(3) at 60 deg) will have vertices colliding!
    # The vertex of A at 60 deg is at distance 1 from center A.
    # The vertex of B at 240 deg (towards A) is at distance 1 from center B.
    # Distance between centers is sqrt(3) approx 1.732.
    # Sum of vertex radii is 2.
    # 1.732 < 2. So they overlap.
    
    # To make them disjoint, we need to rotate them or change spacing.
    # If we rotate the inner hexagons by 30 degrees:
    # Vertices at 30, 90, 150...
    # Edges midpoints at 0, 60, 120...
    # Normals at 0, 60, 120...
    # If we place B at distance sqrt(3) in direction 0 (horizontal):
    # Edge of A facing 0 is flat. Edge of B facing 180 is flat.
    # They touch perfectly.
    # So, if we rotate ALL inner hexagons by 30 degrees, the grid spacing of sqrt(3) in directions 0, 60, 120 works.
    
    # Let's set angle = 30 for all inner hexagons.
    for i in range(len(inner)):
        inner[i][2] = 30.0

    # Now we have a valid, tight packing of 11 hexagons.
    # We need to find the minimal outer hexagon side length S.
    # The outer hexagon can be rotated.
    # To minimize S, we should align the outer hexagon's symmetry with the cluster.
    # The cluster has 6-fold symmetry?
    # The 3-4-3 block is not 6-fold symmetric. It's 2-fold (180 deg rotation) or reflectional.
    # However, the "bounding hexagon" of a hexagonal cluster is often minimized when the outer hexagon is aligned with the lattice or rotated by 30.
    
    # Let's compute the required S for a given outer angle.
    # S = max over all 6 outer normals of (max projection of inner vertices on that normal) / apothem_factor
    # apothem_factor = sqrt(3)/2.
    
    def hex_vertices(cx, cy, S, angle_deg):
        a = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + math.radians(60 * i)
            verts.append((cx + S * math.cos(theta), cy + S * math.sin(theta)))
        return verts

    def compute_S(inner_list, outer_angle_deg):
        # Get all vertices of all inner hexagons
        all_verts = []
        for (x, y, ang) in inner_list:
            verts = hex_vertices(x, y, 1.0, ang)
            all_verts.extend(verts)
        
        max_proj = 0.0
        a = math.radians(outer_angle_deg)
        apothem = math.sqrt(3) / 2.0
        
        for i in range(6):
            # Normal direction for face i
            # Face normal is perpendicular to edge.
            # If vertices are at a + 60*i, edge between i and i+1 has midpoint angle a + 60*i + 30.
            # Normal is a + 60*i + 30.
            theta_normal = a + math.radians(30 + 60 * i)
            nx = math.cos(theta_normal)
            ny = math.sin(theta_normal)
            
            proj = 0.0
            for (vx, vy) in all_verts:
                p = vx * nx + vy * ny
                if p > proj:
                    proj = p
            
            if proj > max_proj:
                max_proj = proj
        
        return max_proj / apothem

    # We want to minimize S by choosing the outer angle.
    # Also, we can translate the inner cluster to center it in the outer hexagon.
    # The "center" of the cluster in terms of bounding box might not be the origin.
    # Let's find the centroid of the vertices or just use the bounding box center.
    # Actually, the formula above assumes the outer hexagon is centered at (0,0).
    # We can shift the inner hexagons by (dx, dy) to minimize S.
    # S(dx, dy, angle) = max_n max_v ( (v + d) . n ) / apothem
    # This is equivalent to max_n ( max_v (v . n) + d . n ) / apothem
    # To minimize the max of these 6 terms, we should choose d such that the projections are balanced.
    # Specifically, for each normal n, let M_n = max_v (v . n) and m_n = min_v (v . n).
    # The width in direction n is W_n = M_n - m_n.
    # If we shift by d, the max projection becomes M_n + d . n.
    # The min projection becomes m_n + d . n.
    # We want to minimize S = max_n (M_n + d . n) / apothem.
    # Note that for the opposite normal -n, the term is M_{-n} + d . (-n) = -m_n - d . n.
    # Since the hexagon is centered at 0, the constraint is symmetric.
    # Actually, the outer hexagon contains the points if for all n, max_proj(n) <= apothem * S AND min_proj(n) >= -apothem * S.
    # This is equivalent to |proj(n)| <= apothem * S for all n? No.
    # The condition is that the projection of the set onto any normal n is within [-apothem*S, apothem*S].
    # So we need max_n (max_proj(n) - min_proj(n)) / 2 <= apothem * S?
    # No, the width in direction n is W_n. The hexagon width in direction n is 2 * apothem * S.
    # So we need W_n <= 2 * apothem * S for all n.
    # S >= max_n (W_n / (2 * apothem)).
    # And we can center the cluster such that the midpoint of the projection interval is 0.
    # So d . n = -(M_n + m_n) / 2 for all n? This is only possible if the cluster is centrally symmetric.
    # The 3-4-3 cluster is centrally symmetric (180 deg rotation).
    # So m_n = -M_n.
    # Thus W_n = 2 M_n.
    # S = max_n (M_n / apothem).
    # And the optimal center is (0,0) if the cluster is centered at (0,0).
    # Is my cluster centered at (0,0)?
    # Row 0: y=0. Row 1: y=1.5. Row 2: y=3.0.
    # Centroid y = (3*0 + 4*1.5 + 3*3) / 10? No, 11 hexes.
    # Sum y = 0 + 6 + 9 = 15.
    # Mean y = 15/11 approx 1.36.
    # It is NOT centered at (0,0).
    # So we must shift the inner hexagons by (0, -15/11) to center them?
    # Or just compute the bounding box and center it.
    
    # Let's compute the optimal center and S for a given outer angle.
    # For a given outer angle, we have 6 normals n_i.
    # We want to minimize S subject to:
    # max_v (v + d) . n_i <= apothem * S
    # min_v (v + d) . n_i >= -apothem * S
    # This is equivalent to:
    # (max_v v . n_i - min_v v . n_i) + (max_v v . n_i + min_v v . n_i + 2 d . n_i) ... no.
    # Let P_i = max_v v . n_i, Q_i = min_v v . n_i.
    # Constraints:
    # P_i + d . n_i <= A
    # Q_i + d . n_i >= -A
    # where A = apothem * S.
    # We want to minimize A.
    # This is a linear programming problem in variables d_x, d_y, A.
    # But since we have only 2D d, we can solve it analytically or with a small LP.
    # Given the small size, we can just iterate over outer angles and solve for d and A.
    # Actually, for a fixed outer angle, the optimal d is the one that centers the "widths".
    # For each normal n_i, the interval of projections is [Q_i + d . n_i, P_i + d . n_i].
    # We want to fit this interval into [-A, A].
    # This requires:
    # P_i + d . n_i <= A
    # -Q_i - d . n_i <= A  (from Q_i + d . n_i >= -A)
    # So A >= max( P_i + d . n_i, -Q_i - d . n_i ).
    # We want to minimize A over d.
    # This is minimizing the maximum of linear functions of d.
    # This is a convex optimization problem.
    # We can solve it using a simple gradient descent or just check the vertices of the dual.
    # Alternatively, since we only have 2 variables (d_x, d_y), we can use scipy.optimize.linprog if available, 
    # or just a simple grid search / ternary search on angle and d.
    # But wait, the directive says "analytically computing... avoiding iterative numerical optimization".
    # For a fixed angle, the optimal d is determined by the "tightest" constraints.
    # Usually, the optimal center is the centroid of the "critical" vertices.
    
    # Let's use a simpler approach:
    # 1. Iterate over outer angles (0 to 60).
    # 2. For each angle, find the optimal d by minimizing the max width.
    #    The width in direction n is W_n = P_n - Q_n.
    #    The shift d affects the position, not the width.
    #    Wait, S is determined by the *radius* of the bounding hexagon, not just width?
    #    No, for a hexagon, the side length S is related to the apothem A.
    #    The condition is that the projection of the set onto ANY normal of the hexagon must be within [-A, A].
    #    If we shift by d, the projection interval shifts by d . n.
    #    To minimize A, we should center the interval.
    #    So for each n, we set d . n = -(P_n + Q_n) / 2.
    #    This is a system of 6 equations for 2 unknowns (d_x, d_y).
    #    It is overdetermined. We want to find d that minimizes max_n | (P_n + Q_n)/2 + d . n | + W_n/2?
    #    No.
    #    Constraint: | (P_n + Q_n)/2 + d . n | + W_n/2 <= A.
    #    Let C_n = (P_n + Q_n)/2 and W_n = P_n - Q_n.
    #    We want to minimize A such that there exists d with | C_n + d . n | <= A - W_n/2 for all n.
    #    This implies A >= W_n/2 + | C_n + d . n |.
    #    We want to minimize max_n ( W_n/2 + | C_n + d . n | ).
    #    This is still an optimization over d.
    #    
    #    However, for the 3-4-3 cluster, the symmetry might simplify this.
    #    The cluster is symmetric under 180 deg rotation.
    #    So C_n = 0 for all n?
    #    If the cluster is centered at (0,0), then P_n = -Q_n, so C_n = 0.
    #    Then we just need A >= W_n/2.
    #    And d = (0,0) is optimal.
    #    So, if we center the cluster at (0,0), the optimal outer hexagon is also at (0,0).
    #    And S = max_n (W_n / (2 * apothem)).
    
    # So the plan:
    # 1. Calculate the centroid of the 11 hex centers (or vertices) and shift them to (0,0).
    # 2. For a range of outer angles, calculate W_n for the 6 normals.
    # 3. S = max(W_n) / (2 * apothem).
    # 4. Pick the angle that minimizes S.
    
    # Step 1: Center the cluster
    # Compute centroid of all vertices
    all_verts = []
    for (x, y, ang) in inner:
        verts = hex_vertices(x, y, 1.0, ang)
        all_verts.extend(verts)
    
    cx = sum(v[0] for v in all_verts) / len(all_verts)
    cy = sum(v[1] for v in all_verts) / len(all_verts)
    
    # Shift inner hexagons
    centered_inner = []
    for (x, y, ang) in inner:
        centered_inner.append([x - cx, y - cy, ang])
        
    # Recompute vertices for centered cluster
    all_verts_centered = []
    for (x, y, ang) in centered_inner:
        verts = hex_vertices(x, y, 1.0, ang)
        all_verts_centered.extend(verts)
        
    # Step 2: Find optimal outer angle
    best_S = float('inf')
    best_angle = 0.0
    apothem = math.sqrt(3) / 2.0
    
    # Scan angles from 0 to 60 with fine step
    for deg in np.arange(0, 60, 0.1):
        a = math.radians(deg)
        max_width = 0.0
        for i in range(6):
            theta_normal = a + math.radians(30 + 60 * i)
            nx = math.cos(theta_normal)
            ny = math.sin(theta_normal)
            
            proj = [v[0]*nx + v[1]*ny for v in all_verts_centered]
            w = max(proj) - min(proj)
            if w > max_width:
                max_width = w
                
        S = max_width / (2 * apothem)
        if S < best_S:
            best_S = S
            best_angle = deg
            
    # Final result
    # Return inner hexagons (centered), outer center (0,0), side length best_S, angle best_angle
    # The interface requires: inner rows (x, y, angle_degrees), outer center (x, y), side, angle
    
    return centered_inner, (0.0, 0.0), best_S, best_angle

# EVOLVE-BLOCK-END
