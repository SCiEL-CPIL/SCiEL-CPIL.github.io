import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # Safer margin
    
    # Initial positions
    # Center
    centers = np.array([[0.0, 0.0]])
    angles = np.array([0.0])
    
    # Ring 1
    for i in range(6):
        theta = np.pi / 3 * i
        x = np.sqrt(3) * np.cos(theta)
        y = np.sqrt(3) * np.sin(theta)
        centers = np.vstack([centers, [[x, y]]])
        angles = np.append(angles, 0.0)
        
    # Ring 2 (4 hexes)
    # Let's try a symmetric arrangement for the remaining 4
    # To fit 11 hexes, we need a compact cluster.
    # A standard hexagonal packing of 19 hexes has rings 1, 6, 12.
    # We have 1 + 6 + 4.
    # Let's place the 4 in the "corners" of the second ring to keep it compact.
    # The 12 positions of ring 2 are at angles 0, 30, 60... 330 degrees?
    # No, ring 2 centers are at distance 3*sqrt(3) if aligned?
    # Distance from center to ring 1 is sqrt(3).
    # Distance from center to ring 2 is 2*sqrt(3) ?
    # Let's just use a grid-based initialization or simple radial placement.
    
    # For now, keep the existing r2_pos but ensure they are distinct and reasonable.
    r2_pos = [
        [0.0, 3.0 * np.sqrt(3) * 0.577], # This is getting messy. 
    ]
    
    # Let's use a simpler, known good initial configuration for 11 hexes.
    # Center (0,0)
    # 6 neighbors at dist sqrt(3)
    # 4 more neighbors. 
    # To minimize bounding hexagon, we want to fill in the gaps.
    # The gaps in ring 1 are at angles 30, 90, 150...
    # Placing hexes at distance 2*sqrt(3) * cos(30) = 3 ?
    # Let's just use the previous r2_pos but verify they don't overlap.
    # Previous r2_pos:
    # [0.0, 3.0] -> dist 3. Ring 1 top is at y=sqrt(3)=1.732. Gap is 3-1.732=1.268. Hex radius is 1. So 1.268 > 1? No, distance between centers must be > sqrt(3)=1.732 for no overlap if aligned.
    # Center (0,0) to (0,3) is 3. OK.
    # Ring 1 top (0, 1.732) to (0,3) is 1.268. 1.268 < 1.732. OVERLAP!
    
    # The initial configuration in the failed code likely had overlaps, causing the search to fail or get stuck.
    # We need a valid initial configuration.
    
    # Valid config for 11:
    # 1 at (0,0)
    # 6 at dist sqrt(3)
    # 4 at dist 3*sqrt(3)/2 ? No.
    
    # Let's construct a valid one programmatically.
    # Place 19 hexes in a hexagonal pattern, then remove 8.
    # Or just place them in a line? No, we want to minimize outer hexagon.
    
    # Let's use a grid of hexagons.
    # Flat top hexagons.
    # dx = 1.5, dy = sqrt(3)
    # Offset rows by sqrt(3)/2.
    
    centers = []
    angles = []
    
    # Center
    centers.append([0.0, 0.0])
    angles.append(0.0)
    
    # 6 neighbors
    for i in range(6):
        theta = np.pi / 3 * i
        x = np.sqrt(3) * np.cos(theta)
        y = np.sqrt(3) * np.sin(theta)
        centers.append([x, y])
        angles.append(0.0)
        
    # 4 more.
    # To keep it compact, place them in the "holes" of the first ring?
    # The holes are at distance 2*sqrt(3) * sin(30) ? No.
    # The vertices of the inner hexagon formed by the 6 neighbors are at distance 1 from center? No.
    # The 6 neighbors touch each other.
    # The space between 3 mutually touching hexagons (center + 2 neighbors) is a rhombus.
    # We can't fit a whole hexagon there.
    
    # We need to go to the second shell.
    # Second shell positions are at distance 3*sqrt(3) / 2 * 2 ?
    # Distance to ring 1 is sqrt(3).
    # Distance to ring 2 is 2*sqrt(3) ?
    # If we place a hex at (0, 2*sqrt(3)), it touches the top neighbor (0, sqrt(3)) at distance sqrt(3). OK.
    # It is at distance 2*sqrt(3) from center.
    # Let's place 4 hexes at the "corners" of the second ring.
    # Angles 90, 210, 330, ...?
    # Let's try placing them at angles 0, 120, 240 and one more?
    # Actually, for 11 hexes, a common minimal bounding box arrangement is a central cluster.
    
    # Let's place 4 hexes at distance 3*sqrt(3) * 0.577?
    # Let's just place them at:
    # (0, 2*sqrt(3))
    # (1.5, -sqrt(3)) ? No.
    
    # Let's use a simple heuristic: place them at distance 3*sqrt(3)/2 * 2?
    # Let's try:
    r2_pos = [
        [0.0, 2.0 * np.sqrt(3)],
        [1.5 * np.sqrt(3), -1.0 * np.sqrt(3)],
        [-1.5 * np.sqrt(3), -1.0 * np.sqrt(3)],
        # Need one more.
        # Maybe [3.0, 0.0]?
        # Let's check symmetry.
    ]
    
    # This is getting complicated. Let's just use a random restart strategy with a valid initial seed.
    # Generate 11 random positions, check validity, if invalid, nudge.
    
    # Better: Use the "19 hex" pattern and pick 11.
    # The 19 hex pattern fits in a hexagon of side 3 (if unit hexes are side 1).
    # Side length of outer hex for 19 hexes (side 1) is 3.
    # We want to minimize side length.
    # If we remove 8 hexes from the corners, we might reduce the side length.
    
    # Let's generate the 19 hex centers.
    all_centers = []
    all_angles = []
    
    # Center
    all_centers.append([0.0, 0.0])
    all_angles.append(0.0)
    
    # Ring 1 (6)
    for i in range(6):
        theta = np.pi / 3 * i
        x = np.sqrt(3) * np.cos(theta)
        y = np.sqrt(3) * np.sin(theta)
        all_centers.append([x, y])
        all_angles.append(0.0)
        
    # Ring 2 (12)
    # Positions:
    # 6 at distance 3*sqrt(3) ? No.
    # The 19 hex cluster has side length 3.
    # The outer vertices are at distance 3.
    # The centers of the outer ring are at distance 2*sqrt(3) * cos(30) = 3? No.
    # Apothem of outer hex (side 3) is 3*sqrt(3)/2.
    # Distance from center to outer hex vertices is 3.
    # The centers of the outer hexes in the 19-cluster are at distance:
    # For the "flat" sides, the centers are at distance 2*sqrt(3) * 0.866?
    # Let's just calculate.
    # In a hex grid, neighbors are at dist sqrt(3).
    # Ring 2 centers are 2 steps away.
    # Directions: 0, 60, 120... and 30, 90...
    # 6 centers at distance 2*sqrt(3) * cos(30) = 3?
    # 6 centers at distance 2*sqrt(3) * sin(30) = sqrt(3)? No.
    
    # Let's use coordinates.
    # u, v coordinates.
    # Center (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) in axial coords?
    # Cartesian:
    # (sqrt(3), 0)
    # (sqrt(3)/2, 1.5)
    # (-sqrt(3)/2, 1.5)
    # (-sqrt(3), 0)
    # (-sqrt(3)/2, -1.5)
    # (sqrt(3)/2, -1.5)
    
    # Ring 2:
    # (2*sqrt(3), 0)
    # (3*sqrt(3)/2, 3)
    # (0, 3)
    # (-3*sqrt(3)/2, 3)
    # (-2*sqrt(3), 0)
    # (-3*sqrt(3)/2, -3)
    # (0, -3)
    # (3*sqrt(3)/2, -3)
    # (2*sqrt(3), 0) ... wait, that's 8?
    # Ring 2 has 12 hexes.
    
    # Let's just take the first 11 from a sorted list of 19?
    # Or just use the 19-hex config and remove the 8 that are furthest from the center?
    # They are all at the same distance or close.
    
    # Let's just use the 19-hex config, and remove 8 random ones?
    # No, we want to minimize the bounding hexagon.
    # Removing corner hexes is best.
    
    # Let's define the 19 centers explicitly.
    # It's easier to just use a valid 11-hex config.
    # How about a 3x3 grid of hexagons?
    # 3 rows.
    # Row 0: 3 hexes
    # Row 1: 3 hexes
    # Row 2: 3 hexes
    # Total 9. Need 2 more.
    
    # Let's stick to the radial approach but fix the overlaps.
    # The failed code's r2_pos had overlaps.
    
    # Correct r2_pos for 4 hexes to complement the 7 (center+ring1):
    # The 7 hexes form a cluster of diameter 2*sqrt(3) (distance between opposite ring 1 hexes is 2*sqrt(3)).
    # We need to add 4 hexes.
    # If we add them at the "corners" of the hexagon formed by ring 1, they will be at distance 3 from center?
    # Let's try placing them at angles 0, 120, 240, and ...?
    # If we place 3 at 0, 120, 240 with dist 3, and 1 at 90?
    
    # Let's just use a random valid initialization.
    # Generate 11 random points in a circle of radius 5.
    # Check for overlaps. If overlap, move apart.
    
    np.random.seed(42)
    centers = np.random.uniform(-3, 3, (N, 2))
    angles = np.zeros(N)
    
    # Force center to be near 0,0
    centers[0] = [0.0, 0.0]
    
    # Relax overlaps
    for _ in range(1000):
        moved = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < np.sqrt(3) - 1e-6:
                    # Push apart
                    diff = centers[i] - centers[j]
                    if d == 0:
                        diff = np.array([1.0, 0.0])
                    else:
                        diff /= d
                    centers[i] += 0.1 * diff
                    centers[j] -= 0.1 * diff
                    moved = True
        if not moved:
            break
            
    # Verify no overlaps
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(centers[i] - centers[j]) < np.sqrt(3) - 1e-6:
                valid = False
    if not valid:
        # Fallback to a simple grid
        centers = []
        angles = []
        for i in range(3):
            for j in range(3):
                x = i * 1.5
                y = j * np.sqrt(3) + (i % 2) * np.sqrt(3)/2
                centers.append([x - 1.5, y - np.sqrt(3)/2])
                angles.append(0.0)
        # Add 2 more
        centers.append([0.0, 2.5 * np.sqrt(3)])
        angles.append(0.0)
        centers.append([0.0, -2.5 * np.sqrt(3)])
        angles.append(0.0)
        centers = np.array(centers)
        angles = np.array(angles)

    best_side = 100.0
    best_centers = centers.copy()
    best_angles = angles.copy()
    
    def get_hex_vertices(center, angle):
        c = np.array(center)
        a = np.deg2rad(angle)
        v_rel = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        v_rot = v_rel @ R.T
        return c + v_rot
        
    def check_overlap(c1, a1, c2, a2):
        d = np.linalg.norm(c1 - c2)
        if d < np.sqrt(3) - 1e-6:
            return True
        return False

    def get_min_side(centers, angles, outer_rotation=0.0):
        normals = []
        for k in range(6):
            theta = np.deg2rad(outer_rotation + 30 + 60 * k)
            normals.append(np.array([np.cos(theta), np.sin(theta)]))
            
        max_proj = 0.0
        for i in range(len(centers)):
            v = get_hex_vertices(centers[i], angles[i])
            for n in normals:
                projs = np.dot(v, n)
                max_proj = max(max_proj, np.max(projs))
                
        S = max_proj / (np.sqrt(3) / 2)
        return S

    current_side = get_min_side(centers, angles)
    best_side = current_side
    best_centers = centers.copy()
    best_angles = angles.copy()
    
    step_size = 0.5
    no_improve_count = 0
    
    while time.time() < deadline:
        new_centers = centers.copy()
        new_angles = angles.copy()
        
        idx = np.random.randint(0, N)
        new_centers[idx] += np.random.normal(0, step_size, 2)
        new_angles[idx] += np.random.normal(0, step_size)
        
        overlap = False
        for i in range(N):
            for j in range(i+1, N):
                if check_overlap(new_centers[i], new_angles[i], new_centers[j], new_angles[j]):
                    overlap = True
                    break
            if overlap: break
            
        if overlap:
            no_improve_count += 1
            continue
            
        new_side = get_min_side(new_centers, new_angles)
        
        if new_side < current_side:
            centers = new_centers
            angles = new_angles
            current_side = new_side
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if new_side < best_side:
            best_side = new_side
            best_centers = new_centers.copy()
            best_angles = new_angles.copy()
            
        if no_improve_count > 100:
            step_size *= 0.9
            no_improve_count = 0
        elif no_improve_count > 500:
            # Restart from best
            centers = best_centers.copy()
            angles = best_angles.copy()
            current_side = best_side
            step_size = 0.5
            no_improve_count = 0

    return [[c[0], c[1], a] for c, a in zip(best_centers, best_angles)], (0.0, 0.0), best_side, 0.0

