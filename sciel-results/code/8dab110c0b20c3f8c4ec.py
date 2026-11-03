import numpy as np
import time

N = 11

def _hex_vertices(cx, cy, angle_deg=0.0):
    """
    Get vertices of a unit regular hexagon centered at (cx, cy) rotated by angle_deg.
    """
    angle_rad = np.deg2rad(angle_deg)
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    angles = base_angles + angle_rad
    x = cx + np.cos(angles)
    y = cy + np.sin(angles)
    return np.column_stack((x, y))

def _min_outer_side(inner_centers, inner_angles, outer_angle_deg=0.0):
    """
    Calculate the minimum side length S of the outer hexagon.
    """
    theta_rad = np.deg2rad(outer_angle_deg)
    normals = []
    for k in range(6):
        alpha = theta_rad + np.deg2rad(30 + k * 60)
        normals.append((np.cos(alpha), np.sin(alpha)))
    
    max_proj = 0.0
    for (cx, cy), ang in zip(inner_centers, inner_angles):
        verts = _hex_vertices(cx, cy, ang)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                proj = vx * nx + vy * ny
                if proj > max_proj:
                    max_proj = proj
    
    S = 2.0 * max_proj / np.sqrt(3.0)
    return S

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # Lattice basis
    A = np.array([np.sqrt(3), 0.0])
    B = np.array([np.sqrt(3)/2, 1.5])
    
    # Generate lattice points up to a certain index range
    # We expect points to be within radius ~4
    # Max index roughly 4
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            c = i * A + j * B
            points.append((tuple(c), (i, j)))
            
    # We need to find 11 disjoint points.
    # Distance between centers must be >= sqrt(3).
    # We will try specific topologies to save time, rather than brute forcing all combinations.
    
    # Topology 1: Center + 6 neighbors + 4 from next shell
    # Center: (0,0)
    # Neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Next shell candidates: (1,1), (1,-2), (2,-1), (-1,-1), (-2,1), (-1,2) [Distance 3]
    # And (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2) [Distance 2*sqrt(3)]
    
    base_centers = [
        (0.0, 0.0),
        (np.sqrt(3), 0.0),
        (np.sqrt(3)/2, 1.5),
        (-np.sqrt(3)/2, 1.5),
        (-np.sqrt(3), 0.0),
        (-np.sqrt(3)/2, -1.5),
        (np.sqrt(3)/2, -1.5)
    ]
    
    # Candidates for the 4th ring
    # Ring A (Dist 3):
    ring_A = [
        (1.5*np.sqrt(3), 1.5),
        (0.0, 3.0),
        (-1.5*np.sqrt(3), 1.5),
        (-1.5*np.sqrt(3), -1.5),
        (0.0, -3.0),
        (1.5*np.sqrt(3), -1.5)
    ]
    
    # Ring B (Dist 2*sqrt(3)):
    ring_B = [
        (2*np.sqrt(3), 0.0),
        (np.sqrt(3), 3.0),
        (-np.sqrt(3), 3.0),
        (-2*np.sqrt(3), 0.0),
        (-np.sqrt(3), -3.0),
        (np.sqrt(3), -3.0)
    ]
    
    best_S = float('inf')
    best_config = None
    
    # Try different subsets of 4 from Ring A and Ring B
    # And different rotations
    
    # Generate all combinations of 4 from Ring A
    from itertools import combinations
    
    candidates = []
    for combo in combinations(range(6), 4):
        # Check if disjoint from each other and base
        # Distance between any two in combo must be >= sqrt(3)
        valid = True
        for i in range(4):
            for j in range(i+1, 4):
                p1 = np.array(ring_A[combo[i]])
                p2 = np.array(ring_A[combo[j]])
                if np.linalg.norm(p1 - p2) < np.sqrt(3) - 1e-9:
                    valid = False
                    break
            if not valid: break
        if valid:
            candidates.append((ring_A, combo))
            
    # Generate all combinations of 4 from Ring B
    for combo in combinations(range(6), 4):
        valid = True
        for i in range(4):
            for j in range(i+1, 4):
                p1 = np.array(ring_B[combo[i]])
                p2 = np.array(ring_B[combo[j]])
                if np.linalg.norm(p1 - p2) < np.sqrt(3) - 1e-9:
                    valid = False
                    break
            if not valid: break
        if valid:
            candidates.append((ring_B, combo))
            
    # Also try mixed rings? Unlikely to be better, but let's stick to pure rings for speed.
    
    for ring, combo in candidates:
        # Construct full center list
        extra_centers = [ring[idx] for idx in combo]
        all_centers = base_centers + extra_centers
        
        # Try rotations of the entire set
        # The set has some symmetry, but we'll try 0, 30, 60...
        for rot_deg in range(0, 360, 15): # Step 15 for safety, or just 0,30,60?
            if time.time() > deadline - 5:
                break
                
            rot_rad = np.deg2rad(rot_deg)
            cos_r = np.cos(rot_rad)
            sin_r = np.sin(rot_rad)
            
            rotated_centers = []
            for (x, y) in all_centers:
                rx = x * cos_r - y * sin_r
                ry = x * sin_r + y * cos_r
                rotated_centers.append((rx, ry))
                
            # Inner hexagons aligned with outer (0 deg)
            S = _min_outer_side(rotated_centers, [0.0]*N, outer_angle_deg=0.0)
            
            if S < best_S:
                best_S = S
                best_config = (rotated_centers, [0.0]*N, 0.0)
                
    if best_config is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    centers, angles, outer_ang = best_config
    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    return inner, (0.0, 0.0), best_S, outer_ang

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
