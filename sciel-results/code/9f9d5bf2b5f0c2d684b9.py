import numpy as np

N = 11

def generate_hex_lattice(n):
    points = []
    if n <= 0:
        return points
    
    # Ring 0
    points.append((0, 0))
    count = 1
    
    if count >= n:
        return points[:n]
        
    ring = 1
    while count < n:
        # Generate points in this ring
        # Start at (ring, 0) and go around
        # Directions:
        # (1,0) -> (0,1) -> (-1,1) -> (-1,0) -> (0,-1) -> (1,-1)
        # Step 1: (ring, 0) to (ring, -ring) ? No.
        # Standard hex ring generation:
        # Corner 1: (ring, 0)
        # Side 1: (ring, -ring) to (ring, 0)? 
        # Let's use axial coordinates.
        # Corners are:
        # (ring, 0), (ring, -ring), (0, -ring), (-ring, 0), (-ring, ring), (0, ring)
        
        # We can generate all 6*ring points.
        ring_points = []
        
        # Side 1: q from ring to 0, r from 0 to -ring?
        # Let's just iterate the 6 sides.
        
        # Side 1: (ring, 0) to (ring, -ring)
        for i in range(ring + 1):
            q = ring
            r = -i
            ring_points.append((q, r))
            
        # Side 2: (ring, -ring) to (0, -ring)
        for i in range(1, ring + 1):
            q = ring - i
            r = -ring
            ring_points.append((q, r))
            
        # Side 3: (0, -ring) to (-ring, -ring)
        for i in range(1, ring + 1):
            q = -i
            r = -ring
            ring_points.append((q, r))
            
        # Side 4: (-ring, -ring) to (-ring, 0)
        for i in range(1, ring + 1):
            q = -ring
            r = -ring + i
            ring_points.append((q, r))
            
        # Side 5: (-ring, 0) to (0, ring)
        for i in range(1, ring + 1):
            q = -ring + i
            r = i
            ring_points.append((q, r))
            
        # Side 6: (0, ring) to (ring, 0)
        for i in range(1, ring + 1):
            q = i
            r = ring
            ring_points.append((q, r))
            
        # Remove duplicates if any (corners might be added twice? No, ranges are exclusive of start in subsequent sides)
        # Side 1 adds (ring, 0) ... (ring, -ring)
        # Side 2 adds (ring-1, -ring) ... (0, -ring)
        # ...
        # Side 6 adds (1, ring) ... (ring, 0) -> Wait, (ring, 0) is already added in Side 1.
        # So Side 6 should stop before (ring, 0).
        # My Side 6 loop: i from 1 to ring. q=i, r=ring.
        # Last point: q=ring, r=ring. This is NOT (ring, 0).
        # The corner is (ring, 0).
        # The points on the side between (0, ring) and (ring, 0) are (q, ring-q) ?
        # Axial coords: x = q + r/2.
        # (0, ring) is x=ring/2. (ring, 0) is x=ring.
        # The line connecting them: r = ring - q.
        # So q goes from 0 to ring.
        # Side 6 should be q=1..ring, r=ring-q.
        # My previous Side 6: q=i, r=ring. This is wrong.
        
        # Let's restart ring generation logic.
        # It's safer to use a standard algorithm.
        
        ring_points = []
        # Start at (ring, 0)
        q, r = ring, 0
        for side in range(6):
            for i in range(ring):
                ring_points.append((q, r))
                # Move to next point in this side
                # Side 0: (1, 0) -> (1, -1)? No.
                # Let's use cube coords for easier movement?
                # Or just hardcode the 6 directions for the perimeter.
                # Perimeter directions in axial:
                # (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)
                # Wait, these are the 6 neighbor directions.
                # To trace the ring, we move in one direction for 'ring' steps, then turn.
                
                # Let's just generate all points in the ring by checking distance.
                # Distance in axial coords is max(|q|, |r|, |q+r|)? No.
                # Cube coords: x=q, z=r, y=-q-r.
                # dist = max(|x|, |y|, |z|).
                # So ring k consists of points where max(|q|, |r|, |q+r|) == k.
                
                pass # This loop is getting messy. Let's use the distance check method.
                
        # Simpler: Generate all points in a bounding box and filter by distance.
        # For ring k, q, r in [-k, k].
        ring_points = []
        for q in range(-ring, ring + 1):
            for r in range(-ring, ring + 1):
                # Cube coord distance
                x = q
                z = r
                y = -x - z
                dist = max(abs(x), abs(y), abs(z))
                if dist == ring:
                    ring_points.append((q, r))
                    
        # Sort ring_points to be in a contiguous sequence?
        # For "compactness" of the partial ring, we want them to be adjacent.
        # Sorting by angle or just taking the first N might work if they are generated in order.
        # The loop above generates them in row-major order, which is not contiguous around the hex.
        # We should sort them by angle around the center.
        
        def angle_key(pt):
            q, r = pt
            x = np.sqrt(3) * (q + r/2.0)
            y = 1.5 * r
            return np.arctan2(y, x)
            
        ring_points.sort(key=angle_key)
        
        # Add as many as needed
        needed = n - count
        for pt in ring_points:
            if needed <= 0:
                break
            points.append(pt)
            needed -= 1
            count += 1
            
        ring += 1
        
    return points

def construct():
    axial_points = generate_hex_lattice(N)
    
    centers = []
    for q, r in axial_points:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        centers.append([x, y])
        
    if not centers:
        return [], (0.0, 0.0), 1.0, 0.0
        
    centers = np.array(centers)
    
    best_S = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for theta_deg in [0, 15, 30, 45, 60]:
        theta = np.deg2rad(theta_deg)
        
        normals = []
        for k in range(6):
            alpha = theta + k * np.pi / 3.0
            normals.append((np.cos(alpha), np.sin(alpha)))
        
        M = []
        m = []
        for u, v in normals:
            proj = centers[:, 0] * u + centers[:, 1] * v
            M.append(np.max(proj))
            m.append(np.min(proj))
            
        max_width = np.max([M[k] - m[k] for k in range(6)])
        S_candidate = 1.0 + max_width / np.sqrt(3)
        
        cx_guess = np.mean(centers[:, 0])
        cy_guess = np.mean(centers[:, 1])
        
        valid = True
        for k in range(6):
            u, v = normals[k]
            proj_c = cx_guess * u + cy_guess * v
            lower = m[k] + np.sqrt(3)/2 * (S_candidate - 1)
            upper = M[k] - np.sqrt(3)/2 * (S_candidate - 1)
            if proj_c < lower - 1e-9 or proj_c > upper + 1e-9:
                valid = False
                break
        
        if valid:
            if S_candidate < best_S:
                best_S = S_candidate
                best_center = (cx_guess, cy_guess)
                best_angle = theta_deg

    if best_S == float('inf'):
        best_S = 10.0
        best_center = (0.0, 0.0)
        best_angle = 0.0

    inner = [[x, y, 0.0] for x, y in centers]
    
    return inner, best_center, best_S, best_angle

# EVOLVE-BLOCK-START
# The code above is the solution.
# I will place it inside the block.
# EVOLVE-BLOCK-END
