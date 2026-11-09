import numpy as np
import time
import math

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Helper: Get vertices of a unit regular hexagon centered at (cx, cy) with rotation angle (radians)
    def hex_vertices(cx, cy, angle):
        # Base angles for vertices: 0, 60, 120, 180, 240, 300 degrees
        base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
        final_angles = base_angles + angle
        xs = cx + np.cos(final_angles)
        ys = cy + np.sin(final_angles)
        return np.column_stack((xs, ys))

    # Helper: Calculate the minimum side length of a regular hexagon (centered at origin, rotation 0)
    # that contains a set of points.
    # The distance from the center to the edge of a regular hexagon with side length S is R_in = S * sqrt(3) / 2.
    # A point (x, y) is inside the hexagon if its projection onto all 6 edge normals is <= R_in.
    # The edge normals are at angles 30, 90, 150, 210, 270, 330 degrees.
    def min_side_for_points(points):
        # points: (M, 2) array
        if len(points) == 0:
            return 0.0
        
        normals = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
        nx = np.cos(normals)
        ny = np.sin(normals)
        
        # Project all points onto all normals
        # proj[i, k] = x_i * nx_k + y_i * ny_k
        proj = points[:, 0] * nx[None, :] + points[:, 1] * ny[None, :]
        
        # The required inradius R_in must be at least the maximum projection for each normal
        # R_in >= max(proj[:, k]) for each k
        # The hexagon is symmetric, so we just need the max absolute projection?
        # Wait, the hexagon is centered at origin. The condition is |proj| <= R_in?
        # No, the hexagon is defined by 6 half-planes.
        # For a regular hexagon centered at origin with "pointy top" (vertices at 90, 150...),
        # the edges are perpendicular to 0, 60, 120...
        # Let's stick to the definition used in the current program's is_inside_outer:
        # Normals at 30, 90... imply edges are perpendicular to these.
        # The distance from center to edge is R_in.
        # A point is inside if dot(p, n_k) <= R_in for all k?
        # Actually, since the hexagon contains the origin, and is convex,
        # the distance from origin to the supporting line of edge k is R_in.
        # So we need max_k (max_i (dot(p_i, n_k))) <= R_in.
        # Note: We don't need absolute value if we consider all 6 normals covering 360 degrees.
        
        max_projections = np.max(proj, axis=0)
        required_R_in = np.max(max_projections)
        
        # R_in = S * sqrt(3) / 2  =>  S = 2 * R_in / sqrt(3)
        S = 2 * required_R_in / np.sqrt(3)
        return S

    # Helper: Check if a packing is valid (disjoint and inside)
    def is_valid_packing(positions, angles, S):
        # 1. Check inside outer hexagon
        # We can just compute the min side required for all vertices.
        # If min_side <= S, they are inside.
        all_verts = []
        for i in range(N):
            verts = hex_vertices(positions[i, 0], positions[i, 1], angles[i])
            all_verts.append(verts)
        
        all_verts_arr = np.vstack(all_verts)
        required_S = min_side_for_points(all_verts_arr)
        if required_S > S + 1e-9:
            return False
            
        # 2. Check disjointness
        # For N=11, simple O(N^2) vertex-edge check is fast enough.
        # But since we are constructing a known valid topology (Center + Ring), 
        # and optimizing rotations, disjointness is preserved if we start valid and move slowly?
        # No, rotation can cause overlap. We must check.
        
        def point_in_hex(px, py, vertices):
            inside = False
            n = 6
            j = n - 1
            for i in range(n):
                xi, yi = vertices[i]
                xj, yj = vertices[j]
                if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                    inside = not inside
                j = i
            return inside

        def edges_intersect(p1, p2, p3, p4):
            def cross(o, a, b):
                return (a[0]-o[0]) * (b[1]-o[1]) - (a[1]-o[1]) * (b[0]-o[0])
            d1 = cross(p3, p4, p1)
            d2 = cross(p3, p4, p2)
            d3 = cross(p1, p2, p3)
            d4 = cross(p1, p2, p4)
            if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                return True
            return False

        for i in range(N):
            v1 = all_verts[i]
            for j in range(i+1, N):
                v2 = all_verts[j]
                # Check vertex containment
                for v in v1:
                    if point_in_hex(v[0], v[1], v2):
                        return False
                for v in v2:
                    if point_in_hex(v[0], v[1], v1):
                        return False
                # Check edge intersection
                for k in range(6):
                    p1 = v1[k]
                    p2 = v1[(k+1)%6]
                    for l in range(6):
                        p3 = v2[l]
                        p4 = v2[(l+1)%6]
                        if edges_intersect(p1, p2, p3, p4):
                            return False
        return True

    # --- Strategy: Center + Ring Topology ---
    # 1 central hexagon at (0,0)
    # 6 surrounding hexagons at distance d_center_ring
    # 4 outer hexagons? No, 11 total.
    # Standard hexagonal packing:
    # Center (0,0)
    # Ring 1: 6 hexagons at distance sqrt(3) (touching).
    # This uses 7 hexagons.
    # We have 4 more.
    # Where to put them?
    # In a hexagonal lattice, the next ring has 12 hexagons.
    # We only have 4.
    # We can place them in "corners" of the Ring 1, or "sides".
    # To minimize the bounding box, we should probably place them symmetrically.
    # Let's try placing them at the vertices of a larger hexagon?
    # Or just extending the lattice.
    
    # Let's define the positions in a hexagonal lattice coordinate system or Cartesian.
    # Lattice vectors for touching hexagons (side 1):
    # Distance between centers = sqrt(3).
    # Let's align the lattice such that one axis is horizontal.
    # Center C0 at (0,0).
    # C1 at (sqrt(3), 0)
    # C2 at (sqrt(3)/2, 1.5)
    # C3 at (-sqrt(3)/2, 1.5)
    # C4 at (-sqrt(3), 0)
    # C5 at (-sqrt(3)/2, -1.5)
    # C6 at (sqrt(3)/2, -1.5)
    # These 7 form a compact cluster.
    
    # Remaining 4:
    # We can place them at:
    # C7 at (3*sqrt(3)/2, 3*1.5/2)? No, let's look at the lattice.
    # The "corners" of the 7-hex cluster are at distance 2*sqrt(3) from center?
    # No, the distance from C0 to C1 is sqrt(3).
    # The distance from C0 to the next shell (e.g., beyond C1) is 2*sqrt(3).
    # Let's place the remaining 4 at the "tips" of the hexagon formed by Ring 1.
    # Tips are at angles 0, 60, 120, 180, 240, 300.
    # Let's pick 4 of these directions to place the extra 4.
    # To be symmetric, we can't pick just 4 of 6 easily without breaking symmetry?
    # Actually, we can rotate the whole assembly.
    # Let's try placing them at 0, 90, 180, 270? No, lattice is hexagonal.
    
    # Alternative: Use a "Cross" shape?
    # Or just place them in the gaps?
    
    # Let's try a specific configuration:
    # Center (0,0)
    # 6 at radius R1 = sqrt(3).
    # 4 at radius R2 = 2*sqrt(3)?
    # If we place them at 0, 90, 180, 270 degrees relative to the lattice axes?
    
    # Let's generate the 7 core hexagons first.
    R1 = np.sqrt(3)
    core_angles = np.deg2rad(np.arange(0, 360, 60))
    core_positions = np.array([
        [0, 0],
        [R1 * np.cos(0), R1 * np.sin(0)],
        [R1 * np.cos(60*np.pi/180), R1 * np.sin(60*np.pi/180)],
        [R1 * np.cos(120*np.pi/180), R1 * np.sin(120*np.pi/180)],
        [R1 * np.cos(180*np.pi/180), R1 * np.sin(180*np.pi/180)],
        [R1 * np.cos(240*np.pi/180), R1 * np.sin(240*np.pi/180)],
        [R1 * np.cos(300*np.pi/180), R1 * np.sin(300*np.pi/180)],
    ])
    
    # The 4 extra hexagons.
    # In a hexagonal packing, the next layer is at distance 2*sqrt(3) along the lattice vectors?
    # No, the distance from center to the "corners" of the 7-hex cluster is 2 * (distance to vertex of hexagon)?
    # Let's just place them at distance R2 = 2 * R1 = 2*sqrt(3).
    # We need to choose 4 angles.
    # If we choose 0, 90, 180, 270, it's square symmetry, which might not fit a hexagon well.
    # If we choose 0, 60, 120, 180, it's asymmetric.
    # Best bet for a hexagonal outer boundary: Place them symmetrically.
    # Maybe 0, 120, 240 and one more?
    # Actually, for N=11, a known optimal packing often involves 1 center, 6 ring, and 4 more.
    # Let's try placing the 4 extra ones at angles 0, 90, 180, 270 relative to the global frame,
    # but we can rotate the entire assembly.
    
    # Let's try 4 angles: 0, 90, 180, 270.
    R2 = 2 * np.sqrt(3)
    extra_angles = np.deg2rad([0, 90, 180, 270])
    extra_positions = np.array([
        [R2 * np.cos(a), R2 * np.sin(a)] for a in extra_angles
    ])
    
    # Total positions
    positions = np.vstack([core_positions, extra_positions])
    
    # Initial angles for all hexagons: 0 (aligned with global frame)
    angles = np.zeros(N)
    
    # Optimization:
    # 1. Rotate the entire assembly to minimize the bounding hexagon side.
    # 2. Rotate individual hexagons? 
    #    For regular hexagons, rotating them 30 degrees changes the "width".
    #    However, if they are tightly packed, rotating one might cause overlap.
    #    Let's assume all inner hexagons are aligned with the outer hexagon (angle 0) for the first pass.
    #    This is a strong constraint but often optimal for "flat" packings.
    
    # Let's evaluate the current configuration (angle 0 for all)
    def evaluate_config(positions, angles, global_rotation):
        # Apply global rotation
        c = np.cos(global_rotation)
        s = np.sin(global_rotation)
        rot_mat = np.array([[c, -s], [s, c]])
        rotated_positions = positions @ rot_mat.T
        
        # Check validity (disjointness)
        # Since we are just rotating the whole rigid body, disjointness is preserved.
        # We only need to check if it fits.
        all_verts = []
        for i in range(N):
            # Local angle is 0, but the hexagon is rotated by global_rotation?
            # Yes, if the hexagon is rigidly attached to the frame.
            verts = hex_vertices(rotated_positions[i, 0], rotated_positions[i, 1], global_rotation)
            all_verts.append(verts)
        
        all_verts_arr = np.vstack(all_verts)
        S = min_side_for_points(all_verts_arr)
        return S

    # Find best global rotation
    best_S = np.inf
    best_rot = 0
    
    # The bounding box of a hexagon-like shape has 6-fold symmetry, but our shape (4 extra at 90 deg)
    # has lower symmetry.
    # Let's sample rotations.
    # Since the outer hexagon has 6-fold symmetry, we only need to search 0 to 60 degrees?
    # Actually, the "fit" into a hexagon depends on the orientation.
    # Let's do a fine scan.
    
    for rot_deg in np.linspace(0, 60, 120):
        rot_rad = np.deg2rad(rot_deg)
        S = evaluate_config(positions, angles, rot_rad)
        if S < best_S:
            best_S = S
            best_rot = rot_rad

    # Now, can we improve by rotating individual hexagons?
    # If we rotate a hexagon by 30 degrees, its "width" (distance between opposite edges) stays the same (sqrt(3)),
    # but its "height" (distance between opposite vertices) changes from 2 to sqrt(3)?
    # No, for a regular hexagon, the width is constant regardless of rotation?
    # Distance between opposite edges is always sqrt(3) * side.
    # Distance between opposite vertices is always 2 * side.
    # So the "envelope" of a single hexagon is a circle of radius 1? No.
    # The maximum distance from center to any point is 1 (vertex).
    # The minimum distance from center to any edge is sqrt(3)/2.
    # So the "bounding circle" is radius 1.
    # Rotating a hexagon does NOT change its convex hull's extent in any direction?
    # Wait. The support function h(theta) of a regular hexagon is periodic with period 60 degrees.
    # h(theta) = R_in * cos(theta - theta_edge) ...
    # Actually, the support function of a regular polygon is constant if we average, but for a specific direction,
    # it varies.
    # However, if we rotate the hexagon, we change which vertices point in which direction.
    # To minimize the maximum projection onto the 6 outer normals, we want to align the "flat" sides with the outer normals?
    # Or the "points"?
    # If the outer hexagon has edges perpendicular to normals n_k, we want the inner hexagons to have their
    # "flattest" side parallel to the outer edge?
    # The distance from center to edge of inner hex is sqrt(3)/2.
    # The distance from center to vertex is 1.
    # If we align an edge of the inner hexagon parallel to an edge of the outer hexagon,
    # the projection of the inner hexagon onto the normal is exactly sqrt(3)/2.
    # If we align a vertex, the projection is 1.
    # So, aligning edges (angle 0 if outer normals are 30,90... and inner edges are 0,60...?)
    # Let's check.
    # Outer normals: 30, 90, 150...
    # Inner hexagon (angle 0): Vertices at 0, 60... Edges at 30, 90...
    # So if inner angle is 0, the edges are perpendicular to 30, 90... which matches outer normals.
    # This minimizes the projection of the inner hexagon onto the outer normals.
    # So angle 0 is optimal for individual hexagons if they are aligned with the global frame.
    
    # So the current configuration (all angles 0, global rotation optimized) is likely very good.
    
    # Let's verify validity with the best rotation.
    # We need to ensure disjointness.
    # In the "Center + Ring" topology with R1=sqrt(3), the core hexagons touch.
    # The extra hexagons are at R2=2*sqrt(3).
    # Distance from core hex (at R1) to extra hex (at R2) depends on angle.
    # If extra is at 0 deg, core is at 0 deg. Distance = R2 - R1 = sqrt(3). They touch.
    # If extra is at 90 deg, core is at 60 or 120.
    # Distance between (R1, 60) and (R2, 90)?
    # R1=sqrt(3), R2=2sqrt(3).
    # x1 = sqrt(3)*0.5 = 0.866, y1 = 1.5
    # x2 = 0, y2 = 3.464
    # dist^2 = 0.866^2 + (3.464-1.5)^2 = 0.75 + 1.964^2 = 0.75 + 3.85 = 4.6.
    # dist = 2.14.
    # Required distance for disjoint hexagons (side 1) is sqrt(3) = 1.732.
    # 2.14 > 1.732, so they are disjoint.
    
    # So the configuration is valid.
    
    # Final check and return
    positions_final = positions.copy()
    c = np.cos(best_rot)
    s = np.sin(best_rot)
    rot_mat = np.array([[c, -s], [s, c]])
    positions_final = positions_final @ rot_mat.T
    
    # The angles of the inner hexagons in the global frame.
    # Since we rotated the whole system, the local angle 0 becomes best_rot.
    angles_final = np.full(N, best_rot * 180.0 / np.pi)
    
    # One final validity check just in case
    if not is_valid_packing(positions_final, angles_final, best_S):
        # Fallback: expand S slightly
        best_S *= 1.01
        if not is_valid_packing(positions_final, angles_final, best_S):
            # Emergency fallback
            return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    inner = []
    for i in range(N):
        inner.append([positions_final[i,0], positions_final[i,1], angles_final[i]])
    
    return inner, (0.0, 0.0), best_S, 0.0

