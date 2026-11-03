import math
import time
import numpy as np

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin

    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    def get_gap(v1, v2):
        # SAT for two convex polygons
        max_gap = -float('inf')
        # Normals for v1
        for i in range(3):
            e1 = v1[(i+1)%6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            norm1 = np.linalg.norm(n1)
            if norm1 > 1e-9:
                n1 = n1 / norm1
            
            # Project v1 and v2 onto n1
            proj1 = np.dot(v1, n1)
            proj2 = np.dot(v2, n1)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2:
                gap = min2 - max1
            elif max2 < min1:
                gap = min1 - max2
            else:
                gap = - (min(max1, max2) - max(min1, min2))
                
            if gap > max_gap:
                max_gap = gap
                
            # Normals for v2 (only need 3 unique directions per hex, but v2 might be rotated)
            # Actually, for two convex polygons, we need normals from both.
            # Since we iterate i in 0..2 for v1, we get 3 normals.
            # We also need 3 normals from v2.
            e2 = v2[(i+1)%6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            norm2 = np.linalg.norm(n2)
            if norm2 > 1e-9:
                n2 = n2 / norm2
                
            proj1_b = np.dot(v1, n2)
            proj2_b = np.dot(v2, n2)
            min1_b, max1_b = np.min(proj1_b), np.max(proj1_b)
            min2_b, max2_b = np.min(proj2_b), np.max(proj2_b)
            
            if max1_b < min2_b:
                gap = min2_b - max1_b
            elif max2_b < min1_b:
                gap = min1_b - max2_b
            else:
                gap = - (min(max1_b, max2_b) - max(min1_b, min2_b))
                
            if gap > max_gap:
                max_gap = gap
                
        return max_gap

    def is_valid(centers, angles):
        verts_list = [hex_vertices(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                gap = get_gap(verts_list[i], verts_list[j])
                if gap < -1e-5:
                    return False
        return True

    def compute_side(centers, angles):
        # For a regular hexagon with vertices at angles 0, 60, ..., the outward normals
        # to the sides are at angles 30, 90, 150, 210, 270, 330.
        # The apothem (distance from center to side) is R * cos(30) = R * sqrt(3)/2.
        # Here R is the circumradius (distance to vertex).
        # For the outer hexagon with side length S, the circumradius is also S.
        # So Apothem = S * sqrt(3) / 2.
        # Condition: max_projection <= Apothem.
        # S >= 2 * max_projection / sqrt(3).
        
        max_proj = 0.0
        # Normals for "pointy right" hexagon (vertices at 0, 60...)
        # Sides are perpendicular to these angles: 30, 90, 150...
        normal_angles = [30, 90, 150, 210, 270, 330]
        
        for i in range(N):
            verts = hex_vertices(centers[i][0], centers[i][1], angles[i])
            for deg in normal_angles:
                a = math.radians(deg)
                n = np.array([math.cos(a), math.sin(a)])
                proj = np.dot(verts, n)
                max_proj = max(max_proj, np.max(proj))
        
        return 2.0 * max_proj / math.sqrt(3)

    # 1. Generate Initial Valid Configuration
    # Use a honeycomb-like structure.
    # Center at (0,0).
    # Hex grid coordinates (axial) to Cartesian:
    # x = size * sqrt(3) * (q + r/2)
    # y = size * 3/2 * r
    # We want 12 hexes. A 1+6+5 ring or similar.
    # Let's try a cluster of 12.
    # Center (0,0).
    # Neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 6 neighbors. Total 7.
    # Next ring: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1).
    # We need 5 more. Pick 5 from the second ring to minimize max radius.
    
    size = 1.0
    sq3 = math.sqrt(3)
    
    def axial_to_xy(q, r, s):
        x = s * sq3 * (q + r/2.0)
        y = s * 1.5 * r
        return x, y

    # 7 central
    centers = [(0.0, 0.0)]
    angles = [0.0]
    
    # 6 neighbors
    neighbors = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    for q, r in neighbors:
        x, y = axial_to_xy(q, r, size)
        centers.append((x, y))
        angles.append(0.0)
        
    # 5 from second ring
    # To keep symmetry or compactness, pick every other one or specific ones.
    # Let's pick (2,0), (1,1), (-1,2), (-2,0), (0,-2) -- this might be unbalanced.
    # Better: pick 5 that form a semi-circle or compact shape.
    # Actually, just pick the 5 closest to the center from the second ring.
    # All are at same distance. Let's pick (2,0), (1,1), (-1,1) is already taken? No, (-1,1) is neighbor.
    # Second ring axial coords:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # Let's take (2,0), (1,1), (-1,2), (-2,0), (0,-2).
    extra = [(2,0), (1,1), (-1,2), (-2,0), (0,-2)]
    for q, r in extra:
        x, y = axial_to_xy(q, r, size)
        centers.append((x, y))
        angles.append(0.0)
        
    # Verify validity
    if not is_valid(centers, angles):
        # Fallback to a very large spread to ensure validity, then optimize
        centers = []
        for i in range(N):
            a = i * 2 * math.pi / N
            centers.append((10.0 * math.cos(a), 10.0 * math.sin(a)))
        angles = [0.0] * N

    best_side = compute_side(centers, angles)
    best_centers = centers.copy()
    best_angles = angles.copy()

    # 2. Local Optimization (Simulated Annealing)
    T = 1.0
    T_min = 0.001
    alpha = 0.999
    
    while time.time() < deadline and T > T_min:
        # Perturb one hexagon
        idx = np.random.randint(0, N)
        new_centers = [c.copy() if isinstance(c, list) else (c[0], c[1]) for c in centers]
        new_angles = angles.copy()
        
        # Random move
        if np.random.random() < 0.7:
            # Move
            new_centers[idx] = (centers[idx][0] + np.random.normal(0, 0.1), 
                                centers[idx][1] + np.random.normal(0, 0.1))
        else:
            # Rotate
            new_angles[idx] = angles[idx] + np.random.normal(0, 0.1)
            
        if is_valid(new_centers, new_angles):
            new_side = compute_side(new_centers, new_angles)
            delta = new_side - best_side
            if delta < 0 or np.random.random() < math.exp(-delta / T):
                centers = new_centers
                angles = new_angles
                best_side = new_side
                best_centers = new_centers
                best_angles = new_angles
                
        T *= alpha

    # Final Result
    inner = []
    for i in range(N):
        inner.append([float(best_centers[i][0]), float(best_centers[i][1]), float(best_angles[i])])
        
    # Recalculate side for final positions to ensure accuracy
    final_side = compute_side(best_centers, best_angles)
    
    # Add safety margin to side (1% + small epsilon)
    final_side = final_side * 1.02 + 1e-3
    
    return inner, (0.0, 0.0), float(final_side), 0.0
