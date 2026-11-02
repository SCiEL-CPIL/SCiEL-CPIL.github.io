"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Hexagon vertices for unit hexagon (side 1), pointy-top with vertex at angle 0
    hex_angles = [k * 60 * math.pi / 180 for k in range(6)]
    hex_vertices = [(math.cos(a), math.sin(a)) for a in hex_angles]
    
    # Container hexagon: vertex at angle 0, side S
    # Point (x,y) inside iff max_k (x*cos(k*60) + y*sin(k*60)) <= S
    container_angles = [k * 60 * math.pi / 180 for k in range(6)]
    
    def min_container_side(cx_list, cy_list, container_rot=0.0):
        """Compute minimum container side S for given inner hexagon centers."""
        max_val = 0.0
        for cx, cy in zip(cx_list, cy_list):
            for vx, vy in hex_vertices:
                px, py = cx + vx, cy + vy
                # Rotate by -container_rot to align with container axes
                pr = px * math.cos(-container_rot) - py * math.sin(-container_rot)
                qr = px * math.sin(-container_rot) + py * math.cos(-container_rot)
                for ca in container_angles:
                    val = pr * math.cos(ca) + qr * math.sin(ca)
                    if val > max_val:
                        max_val = val
        return max_val
    
    # Try multiple arrangements
    best_S = float('inf')
    best_arrangement = None
    best_rot = 0.0
    
    # Arrangement 1: Central + 6 ring + 4 in second ring (partial)
    # Triangular lattice: basis (sqrt(3), 0) and (sqrt(3)/2, 3/2)
    sqrt3 = math.sqrt(3)
    
    # Generate candidate center positions on triangular lattice
    # For 11 hexagons, try several compact shapes
    
    # Shape A: 1 + 6 + 4 (the 4 in the most symmetric positions)
    # Centers: (0,0), 6 at distance sqrt3, and 4 at distance 2*sqrt3 in alternating directions
    centers_A = [(0.0, 0.0)]
    for k in range(6):
        angle = k * 60 * math.pi / 180
        centers_A.append((sqrt3 * math.cos(angle), sqrt3 * math.sin(angle)))
    # Add 4 more at the "mid" positions of second ring
    for k in range(0, 6, 1):
        if k in [0, 2, 4]:  # 3 positions, need 4 total, add one more
            angle = k * 60 * math.pi / 180
            centers_A.append((2 * sqrt3 * math.cos(angle), 2 * sqrt3 * math.sin(angle)))
    # That's only 10, add one more
    angle = 3 * 60 * math.pi / 180
    centers_A.append((2 * sqrt3 * math.cos(angle), 2 * sqrt3 * math.sin(angle)))
    # Now 11
    
    # Shape B: Row-based 3-4-3-1 or similar
    # Let's try a more systematic approach: brute force small sets of lattice points
    
    # Generate all lattice points within a certain radius
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * sqrt3 + j * sqrt3 / 2
            y = j * 3 / 2
            lattice_points.append((x, y))
    
    # Try all subsets of size 11? That's too many. Instead, try symmetric arrangements.
    
    # Shape C: Hexagonal cluster of 19 minus 8 corners
    # Full hexagon of "radius 2" in hex lattice has 19 points
    # Remove 8 to get 11, keeping it compact
    
    # Let me just try a few hand-picked arrangements and optimize rotation
    
    # Arrangement 1: The "flower" of 7 plus 4 more
    arrangements = []
    
    # A1: 7-hex flower + 4 at second ring positions 0,60,120,180
    arr1 = [(0.0, 0.0)]
    for k in range(6):
        angle = k * 60 * math.pi / 180
        arr1.append((sqrt3 * math.cos(angle), sqrt3 * math.sin(angle)))
    for k in [0, 1, 2, 3]:
        angle = k * 60 * math.pi / 180
        arr1.append((2 * sqrt3 * math.cos(angle), 2 * sqrt3 * math.sin(angle)))
    arrangements.append(arr1)
    
    # A2: 7-hex flower + 4 at second ring positions 0,60,120,240
    arr2 = [(0.0, 0.0)]
    for k in range(6):
        angle = k * 60 * math.pi / 180
        arr2.append((sqrt3 * math.cos(angle), sqrt3 * math.sin(angle)))
    for k in [0, 1, 2, 4]:
        angle = k * 60 * math.pi / 180
        arr2.append((2 * sqrt3 * math.cos(angle), 2 * sqrt3 * math.sin(angle)))
    arrangements.append(arr2)
    
    # A3: Row-based: rows of 3,4,3,1
    # y positions: -1.5, 0, 1.5, 3.0 (spacing 1.5 in y)
    # Actually triangular lattice y-spacing is 3/2 = 1.5
    arr3 = []
    # Row at y=-1.5: 3 hexagons
    for i in [-1, 0, 1]:
        arr3.append((i * sqrt3, -1.5))
    # Row at y=0: 4 hexagons
    for i in [-1.5, -0.5, 0.5, 1.5]:
        arr3.append((i * sqrt3, 0.0))
    # Row at y=1.5: 3 hexagons
    for i in [-1, 0, 1]:
        arr3.append((i * sqrt3, 1.5))
    # Row at y=3.0: 1 hexagon
    arr3.append((0.0, 3.0))
    arrangements.append(arr3)
    
    # A4: Row-based: 2,3,4,2
    arr4 = []
    for i in [-0.5, 0.5]:
        arr4.append((i * sqrt3, -1.5))
    for i in [-1, 0, 1]:
        arr4.append((i * sqrt3, 0.0))
    for i in [-1.5, -0.5, 0.5, 1.5]:
        arr4.append((i * sqrt3, 1.5))
    for i in [-0.5, 0.5]:
        arr4.append((i * sqrt3, 3.0))
    arrangements.append(arr4)
    
    # A5: 1,3,5,2 (center-heavy)
    arr5 = []
    arr5.append((0.0, 0.0))
    for i in [-1, 0, 1]:
        arr5.append((i * sqrt3, 1.5))
    for i in [-1.5, -0.5, 0.5, 1.5, 2.5]:
        arr5.append((i * sqrt3, 3.0))
    for i in [-0.5, 0.5]:
        arr5.append((i * sqrt3, 4.5))
    arrangements.append(arr5)
    
    # For each arrangement, try several rotations
    for arr in arrangements:
        for rot_deg in range(0, 360, 5):
            rot = rot_deg * math.pi / 180
            S = min_container_side([c[0] for c in arr], [c[1] for c in arr], rot)
            if S < best_S:
                best_S = S
                best_arrangement = arr
                best_rot = rot
    
    # Refine: try finer rotation around the best
    if best_arrangement is not None:
        for rot_deg in np.arange(best_rot * 180 / math.pi - 5, best_rot * 180 / math.pi + 5, 0.5):
            rot = rot_deg * math.pi / 180
            S = min_container_side([c[0] for c in best_arrangement], [c[1] for c in best_arrangement], rot)
            if S < best_S:
                best_S = S
                best_rot = rot
    
    # Build the inner hexagons list
    inner = []
    for cx, cy in best_arrangement:
        # The hexagon orientation: we want vertices at angles matching the lattice
        # For the packing to work, all inner hexagons should have the same orientation
        # as the lattice. The lattice was defined with hexagon vertices at 0,60,120,...
        # So angle_degrees = 0 for all
        inner.append([cx, cy, 0.0])
    
    # Container center is at origin, side length is best_S, rotation is best_rot
    container_center = (0.0, 0.0)
    container_side = best_S
    container_angle = best_rot * 180 / math.pi
    
    return inner, container_center, container_side, container_angle
# EVOLVE-BLOCK-END
