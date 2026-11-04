import numpy as np
import time

N = 12
DEADLINE = 145.0
_TIME_START = time.time()

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)

    # The optimal packing for 12 regular hexagons of side 1 is the honeycomb lattice.
    # It consists of:
    # 1 central hexagon at (0,0)
    # 6 hexagons at distance sqrt(3) (inradius*2) around it
    # 5 hexagons at distance 2*sqrt(3) in the gaps of the first ring?
    # Actually, the standard "hexagonal number" H_2 = 7 hexagons (1+6) fit in a hexagon of side 2.
    # We have 12 hexagons.
    # A hexagon of side 3 can hold 1 + 6 + 12 = 19 hexagons.
    # So 12 hexagons can definitely fit in side 3.
    # Can they fit in side < 3?
    # The area of 12 unit hexagons is 12 * (3*sqrt(3)/2) = 18*sqrt(3) ≈ 31.17.
    # Area of enclosing hexagon with side L is (3*sqrt(3)/2) * L^2.
    # L^2 >= 18 => L >= sqrt(18) ≈ 4.24.
    # Wait, area argument gives L >= 4.24.
    # Let's re-evaluate the area.
    # Area of unit hexagon (side 1) = (3*sqrt(3)/2) * 1^2 ≈ 2.598.
    # Total area = 12 * 2.598 = 31.176.
    # Area of outer hexagon (side L) = (3*sqrt(3)/2) * L^2 ≈ 2.598 * L^2.
    # L^2 >= 12 => L >= sqrt(12) ≈ 3.464.
    # So the lower bound is 3.464.
    
    # Let's check the honeycomb packing side length again.
    # Center at (0,0).
    # Ring 1: 6 hexagons. Centers at distance d1 = sqrt(3) from origin.
    # Ring 2: 12 hexagons. Centers at distance d2 = 2*sqrt(3) from origin?
    # In a hexagonal lattice, the distance from center to the k-th ring is k * sqrt(3).
    # Ring 1 (6 hexes): dist = sqrt(3) ≈ 1.732.
    # Ring 2 (12 hexes): dist = 2*sqrt(3) ≈ 3.464.
    # The vertices of the outermost hexagons in Ring 2 will extend further.
    # The maximum distance from the origin to a vertex of a hexagon in Ring 2:
    # The hexagon center is at distance 2*sqrt(3).
    # The vertex furthest from origin is at distance 2*sqrt(3) + 1 (side length).
    # So max radius R = 2*sqrt(3) + 1 ≈ 4.464.
    # The circumradius of a regular hexagon with side L is L.
    # So L must be at least R = 2*sqrt(3) + 1 ≈ 4.464.
    
    # Wait, is there a better packing?
    # The area bound was 3.464.
    # The radius bound for honeycomb is 4.464.
    # Can we pack 12 hexagons in a smaller hexagon?
    # Maybe not in a perfect lattice.
    # However, 9.2 is very large. Even a loose packing would be smaller.
    # The current code is likely failing to pack them into a small space, possibly leaving them scattered or using a very loose initial guess that doesn't converge well.
    
    # Let's try to construct a valid packing with side length close to the theoretical minimum.
    # A known optimal packing for 12 circles/hexagons?
    # For hexagons, the lattice packing is usually optimal.
    # Let's calculate the side length of the enclosing hexagon for the 12-hexagon lattice cluster.
    # If we take 12 hexagons from the honeycomb:
    # 1 center + 6 ring 1 + 5 from ring 2.
    # To minimize the enclosing hexagon, we should place the 5 ring-2 hexagons symmetrically?
    # Or maybe a different shape?
    
    # Let's just use the 19-hexagon cluster (side 3) and remove 7? No, that changes the shape.
    # Let's stick to the lattice points.
    # Coordinates for honeycomb lattice (axial coordinates q, r):
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 6 neighbors
    # Next ring: (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2) -> 12 neighbors
    
    # We need 12 total.
    # 1 + 6 = 7.
    # We need 5 more from the second ring.
    # To minimize the bounding box (hexagon), we should pick 5 that are "closest" or most symmetric?
    # Actually, any 5 will break the symmetry.
    # Perhaps a different configuration is better?
    # What if we use a 3x4 grid?
    
    # Given the complexity, let's provide a robust optimization that starts from the honeycomb lattice and refines it.
    # But since the deadline is tight and the current score is poor, a deterministic high-quality guess is safer.
    
    # Let's generate the 12 hexagons in a honeycomb pattern.
    # We will use the 7 central ones and 5 from the next ring.
    # To minimize the enclosing hexagon, we should align the outer hexagon with the cluster's symmetry if possible, or rotate to minimize the max projection.
    
    sqrt3 = np.sqrt(3)
    
    # Define the 12 centers
    centers = []
    # Center
    centers.append((0.0, 0.0))
    
    # Ring 1 (6 hexagons)
    for i in range(6):
        angle = np.radians(60 * i)
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        centers.append((x, y))
        
    # Ring 2 (we need 5 more)
    # The 12 positions in ring 2 are at distance 2*sqrt3.
    # Angles are 0, 30, 60, ... 330? No, they are at the "corners" and "edges".
    # In a hex lattice, the second ring points are at angles 0, 30, 60... relative to the first ring?
    # Let's list them.
    # First ring angles: 0, 60, 120, 180, 240, 300.
    # Second ring points are at the vertices of the hexagon formed by the first ring centers?
    # No, the second ring centers are at distance 2*sqrt3.
    # Their angles are 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330.
    # Let's verify.
    # Point (2,0) in axial coords -> Cartesian: x = sqrt3 * (q + r/2) = sqrt3 * 2 = 2*sqrt3. y = 1.5 * r = 0. Angle 0.
    # Point (1,1) -> x = sqrt3 * (1 + 0.5) = 1.5*sqrt3. y = 1.5. Angle 30.
    # Point (0,2) -> x = sqrt3 * 1 = sqrt3. y = 3. Angle 60.
    # Yes, the angles are 0, 30, 60, ...
    
    # We need to pick 5 of these 12 to minimize the enclosing hexagon.
    # Due to symmetry, any choice of 5 will result in an asymmetric cluster.
    # However, the enclosing hexagon can be rotated.
    # Let's pick 5 that are contiguous to keep the cluster compact?
    # Or spread out?
    # Compactness usually minimizes the bounding box.
    # Let's pick 5 consecutive ones.
    # Indices 0, 1, 2, 3, 4 (angles 0, 30, 60, 90, 120).
    
    ring2_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    selected_ring2 = ring2_angles[:5]
    
    for angle_deg in selected_ring2:
        angle = np.radians(angle_deg)
        x = 2 * sqrt3 * np.cos(angle)
        y = 2 * sqrt3 * np.sin(angle)
        centers.append((x, y))
        
    # We have 12 centers.
    # Now we need to find the minimum enclosing regular hexagon.
    # We can rotate the outer hexagon to minimize its side length.
    # The side length L of a regular hexagon enclosing a set of points P is:
    # L = max over 6 directions (projected width / sqrt(3)) * 2?
    # Specifically, if the hexagon is centered at origin and rotated by alpha:
    # The distance from center to the side (inradius) is R_in = L * sqrt(3) / 2.
    # The condition is that for all points p, the projection onto the 6 normal vectors is <= R_in.
    # So L >= (2 / sqrt(3)) * max_p max_k |p . n_k(alpha)|.
    
    # We want to minimize L(alpha) = (2 / sqrt(3)) * max_p max_k |p . n_k(alpha)|.
    # We can search for the optimal alpha.
    
    def calc_L(alpha_deg):
        alpha = np.radians(alpha_deg)
        # Normals of the hexagon sides
        # Hexagon sides are at angles alpha, alpha+60, ...
        # The outward normals are at alpha+90, alpha+150, ...
        base_normals = np.radians(np.array([90, 150, 210, 270, 330, 30])) + alpha
        normals = np.column_stack((np.cos(base_normals), np.sin(base_normals)))
        
        # Project all centers onto normals
        # centers is (12, 2)
        proj = np.array(centers) @ normals.T # (12, 6)
        # The constraint is |proj| <= R_in for all points and all normals?
        # Yes, because the hexagon is centered at origin.
        max_proj = np.max(np.abs(proj))
        R_in = max_proj
        L = (2.0 / sqrt3) * R_in
        return L

    # Optimize alpha
    best_L = float('inf')
    best_alpha = 0.0
    
    # Search for best rotation
    # The function is periodic with period 60 degrees.
    for a in np.linspace(0, 60, 121):
        L_val = calc_L(a)
        if L_val < best_L:
            best_L = L_val
            best_alpha = a
            
    # Refine
    if time_left() > 5:
        for a in np.linspace(best_alpha - 1, best_alpha + 1, 101):
            L_val = calc_L(a)
            if L_val < best_L:
                best_L = L_val
                best_alpha = a

    # Construct the return values
    # inner: list of (x, y, angle_degrees)
    # The inner hexagons are aligned (angle 0).
    inner = []
    for c in centers:
        inner.append([float(c[0]), float(c[1]), 0.0])
        
    center = [0.0, 0.0]
    side = float(best_L)
    angle = float(best_alpha)
    
    return inner, center, side, angle
