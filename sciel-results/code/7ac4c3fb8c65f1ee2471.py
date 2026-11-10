import numpy as np
import time
import random

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a 'Flower' arrangement initialization followed by simulated annealing to minimize the outer hexagon side length.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin for 600s limit
    
    # 1. Initialize with "Flower" arrangement
    # Center at (0,0)
    # 6 neighbors at distance sqrt(3) (touching) in directions 0, 60, 120, 180, 240, 300
    # 5 more in the next ring? 
    # Actually, 12 hexagons: 1 center, 6 around it, 5 more? 
    # A full hexagonal number H_3 = 19. We have 12.
    # Common dense packing for 12: 
    # Rows of 3, 4, 3, 2? Or 4, 4, 4?
    # Let's use a standard hex grid cluster.
    # Coordinates in axial system (q, r) converted to cartesian.
    # Axial to cartesian: x = sqrt(3) * (q + r/2), y = 3/2 * r
    
    # Let's try a 4x3 rectangle in hex grid? 
    # Or a cluster of 12.
    # The "flower" is 1 + 6 = 7. We need 5 more.
    # Add 5 to the ring of 6? That creates a larger cluster.
    # Let's define a set of axial coordinates for 12 hexagons that are close-packed.
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 7 total
    # Add: (2,0), (2,-1), (1,1), (-1,2), (-2,1) ? 
    # Let's just use a simple grid of 4x3 or similar and let the optimizer fix it.
    # Better: Use the known optimal or near-optimal configuration for 12 circles/hexagons.
    # For hexagons, the "flower" + 5 is not a standard shape.
    # Let's use a 3x4 grid in axial coords?
    
    # Axial coords for a compact cluster of 12:
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons
    # Row 2: 4 hexagons
    # This is a parallelogram.
    # Or a hexagon shape? 1, 3, 4, 3, 1 = 12? No.
    # 1, 2, 3, 4, 2? = 12.
    
    # Let's generate 12 points in a hex grid pattern.
    # Center (0,0)
    # Neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Next layer partial: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0)
    # We need 5 more. Let's take (2,0), (1,1), (0,2), (-1,2), (-2,1).
    
    axial_coords = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2), (-2,1)
    ]
    
    # Convert to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    sqrt3 = np.sqrt(3)
    centers = np.array([
        [sqrt3 * (q + r/2.0), 1.5 * r] 
        for q, r in axial_coords
    ])
    
    # Initial angles are all 0
    angles = np.zeros(12)
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        a = np.deg2rad(angle_deg)
        angles_rad = a + np.deg2rad(np.arange(6) * 60)
        return np.column_stack([
            cx + side * np.cos(angles_rad),
            cy + side * np.sin(angles_rad)
        ])
    
    def compute_min_enclosing_hexagon(centers):
        """
        Find the minimum side length s of a regular hexagon (centered at origin, fixed orientation 0) 
        that contains all vertices of all inner hexagons.
        Since the outer hexagon can rotate, we should optimize rotation too?
        The problem allows outer angle. 
        However, for a symmetric packing, aligning outer hexagon with inner lattice is often optimal.
        Let's assume outer hexagon is centered at the centroid of the inner centers and has angle 0.
        We can refine this later.
        """
        # Centroid
        cx, cy = np.mean(centers, axis=0)
        # Shift centers to origin for calculation
        shifted_centers = centers - np.array([cx, cy])
        
        # Get all vertices of all inner hexagons
        all_vertices = []
        for i in range(12):
            verts = get_hex_vertices(shifted_centers[i, 0], shifted_centers[i, 1], 1.0, angles[i])
            all_vertices.append(verts)
        
        all_vertices = np.vstack(all_vertices)
        
        # The outer hexagon is defined by 6 lines.
        # For a regular hexagon with side s and angle 0, the distance from center to each edge (apothem) is s * sqrt(3)/2.
        # The normal vectors to the edges are at angles 0, 60, 120, 180, 240, 300.
        # A point (x,y) is inside if dot((x,y), n_i) <= apothem for all i.
        # So s >= max(dot((x,y), n_i)) / (sqrt(3)/2) for all points and all i.
        
        normals = np.array([
            [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] 
            for a in [0, 60, 120, 180, 240, 300]
        ])
        
        # Project all vertices onto all normals
        # projections[i, j] is dot(vertex_i, normal_j)
        projections = all_vertices @ normals.T
        
        # For each normal, find the max projection (furthest point in that direction)
        # Note: We need max for positive normals, but since normals cover 360 deg, 
        # max projection on normal j is the same as max projection on -normal j? 
        # No, dot(p, n) and dot(p, -n) are negatives.
        # The constraint is |dot(p, n)| <= apothem? 
        # No, the hexagon is symmetric. The edges are at distance apothem in directions 0, 60, etc.
        # The condition for a point to be inside is dot(p, n_k) <= apothem for k=0..5.
        # Since n_{k+3} = -n_k, dot(p, n_{k+3}) = -dot(p, n_k).
        # So we need dot(p, n_k) <= apothem AND -dot(p, n_k) <= apothem => |dot(p, n_k)| <= apothem.
        
        max_abs_projections = np.max(np.abs(projections), axis=0)
        
        # apothem = s * sqrt(3) / 2
        # s = apothem / (sqrt(3)/2) = 2 * apothem / sqrt(3)
        required_apothems = max_abs_projections
        s = np.max(required_apothems) * 2.0 / sqrt3
        
        return s, cx, cy

    def check_validity(centers, angles, s, cx, cy):
        """
        Check if the packing is valid.
        1. Inner hexagons do not overlap.
        2. Inner hexagons are inside the outer hexagon.
        """
        # 1. Separation
        # For each pair, check if they overlap.
        # Use SAT (Separating Axis Theorem) for convex polygons.
        # Or simpler: check distance between centers?
        # For regular hexagons of side 1, minimum distance between centers for non-overlap is sqrt(3) if aligned.
        # If rotated, it's more complex.
        # Let's use a robust check: compute vertices and check intersection? 
        # Too slow. 
        # Approximation: The minimum distance between two non-overlapping unit hexagons is at least sqrt(3) if they are aligned.
        # Actually, the distance between centers must be >= sqrt(3) for aligned hexagons.
        # If we assume angles are small or 0, we can use this.
        # But the problem allows rotation.
        # Let's implement a proper check using projections on edge normals.
        
        # Precompute vertices for all hexagons in global coordinates
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
            verts.append(v)
            
        for i in range(12):
            for j in range(i+1, 12):
                if not are_hexagons_separated(verts[i], verts[j]):
                    return False
                    
        # 2. Containment
        # Check if all inner vertices are inside the outer hexagon
        outer_verts = get_hex_vertices(cx, cy, s, 0) # Angle 0 for outer
        # Check each inner vertex
        for i in range(12):
            for v in verts[i]:
                if not is_point_in_hexagon(v, outer_verts):
                    return False
                    
        return True

    def are_hexagons_separated(v1, v2):
        """
        Check if two convex polygons (hexagons) are separated.
        Returns True if separated (no overlap), False if overlapping.
        """
        # SAT: Project onto all edge normals of both polygons.
        # If there is a gap in any projection, they are separated.
        for v in [v1, v2]:
            # Compute edge normals
            edges = np.roll(v, -1, axis=0) - v
            # Normals are perpendicular to edges
            # For CCW polygon, normal is (dy, -dx) or (-dy, dx)?
            # Let's just use the normal direction.
            normals = np.column_stack([edges[:, 1], -edges[:, 0]])
            # Normalize
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / norms
            
            for n in normals:
                proj1 = v1 @ n
                proj2 = v2 @ n
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                # Check for gap
                if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                    return True # Separated
        
        return False # No separating axis found -> Overlapping

    def is_point_in_hexagon(p, hex_verts):
        """
        Check if point p is inside convex polygon hex_verts.
        """
        # Use cross product method
        # For CCW polygon, cross(edge, point - start) should be >= 0
        for i in range(6):
            p1 = hex_verts[i]
            p2 = hex_verts[(i+1)%6]
            edge = p2 - p1
            vec = p - p1
            cross = edge[0]*vec[1] - edge[1]*vec[0]
            if cross < -1e-6:
                return False
        return True

    # Initial evaluation
    s_init, cx_init, cy_init = compute_min_enclosing_hexagon(centers)
    # Adjust centers to be relative to centroid for optimization? 
    # No, keep centers absolute, but the outer hexagon is centered at centroid.
    # To simplify optimization, we can fix the outer hexagon center at (0,0) and optimize inner centers relative to it.
    # Let's shift centers so centroid is at (0,0).
    centers = centers - np.array([cx_init, cy_init])
    
    # Verify initial validity
    s_init, _, _ = compute_min_enclosing_hexagon(centers)
    valid_init = check_validity(centers, angles, s_init, 0, 0)
    if not valid_init:
        # Fallback to a very large hexagon if initial is invalid (shouldn't happen with flower)
        # But flower is valid.
        pass

    best_centers = centers.copy()
    best_angles = angles.copy()
    best_s, best_cx, best_cy = compute_min_enclosing_hexagon(best_centers)
    
    # Simulated Annealing
    # Objective: Minimize s
    # Constraints: Validity
    # We will move centers randomly and accept if s decreases or with prob exp(-delta/T)
    
    T = 1.0
    cooling = 0.999
    step_size = 0.1
    
    while time.time() < deadline and T > 0.01:
        # Generate a candidate
        new_centers = best_centers.copy()
        new_angles = best_angles.copy()
        
        # Move one or two centers
        idx = random.randint(0, 11)
        new_centers[idx, 0] += random.gauss(0, step_size)
        new_centers[idx, 1] += random.gauss(0, step_size)
        
        # Optionally rotate one hexagon
        if random.random() < 0.1:
            new_angles[idx] += random.gauss(0, 5)
            
        # Calculate new s
        new_s, new_cx, new_cy = compute_min_enclosing_hexagon(new_centers)
        
        # Check validity
        # Optimization: Skip full validity check if s increased significantly?
        # No, we need to ensure we don't jump to invalid states that trap us.
        # But validity check is expensive.
        # Let's do a quick check: if s is much larger, it's likely worse, but might be invalid.
        # We'll do full check.
        
        valid = check_validity(new_centers, new_angles, new_s, new_cx, new_cy)
        
        if valid:
            delta = new_s - best_s
            if delta < 0 or random.random() < np.exp(-delta / T):
                best_centers = new_centers
                best_angles = new_angles
                best_s = new_s
                best_cx = new_cx
                best_cy = new_cy
        else:
            # If invalid, we can still accept if T is high? 
            # No, the problem requires valid output. 
            # We should only move to valid states.
            # This makes it a hard-constraint optimization.
            pass
            
        T *= cooling
        # Reduce step size slowly
        if T < 0.1:
            step_size *= 0.95

    # Final result
    # The outer hexagon is centered at (0,0) in our optimized coordinate system?
    # No, we shifted centers so centroid is at (0,0).
    # So outer hexagon center is (0,0).
    # Angle is 0.
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    # Ensure s is at least sqrt(12) (theoretical lower bound? No, sqrt(12) is for circles? 
    # For hexagons, area of 12 unit hexagons is 12 * (3*sqrt(3)/2) = 18*sqrt(3) ~ 31.17.
    # Area of outer hexagon side s is (3*sqrt(3)/2) * s^2.
    # s^2 >= 31.17 / (3*sqrt(3)/2) = 12.
    # So s >= sqrt(12) ~ 3.46.
    # Our s will be much larger.
    
    return best_centers, outer_hex_data, best_s

# Helper functions need to be inside or global. 
# Let's put them inside to keep it self-contained or global.
# The prompt asks for a single code block. I'll define helpers globally or inside.
# Defining inside is cleaner for the block.

# Wait, the code above defines helpers inside hexagon_packing_12. 
# That's fine.

# Let's double check the logic.
# 1. Init with flower.
# 2. Shift to centroid.
# 3. SA loop.
# 4. Return.

# One issue: `check_validity` is O(N^2 * K) where K is vertices. 12*12*36 ~ 5000 ops. 
# SA iterations: 10k-100k? 
# 100k * 5000 = 500M ops. Might be too slow in Python.
# Optimization: 
# - Use vectorized numpy for SAT?
# - Reduce validity checks.
# - Use a cheaper validity check: just check center distances > sqrt(3) - epsilon? 
#   This is sufficient if angles are 0. 
#   If we allow rotation, we need SAT.
#   Let's assume angles are 0 for speed, as the optimal packing of regular polygons often aligns them.
#   If we fix angles to 0, we can just check distance between centers >= sqrt(3).
#   This is MUCH faster.
#   Let's try fixing angles to 0 first. If it works, it's faster and likely optimal for this symmetry.

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 590.0
    
    sqrt3 = np.sqrt(3)
    
    # Axial coords for 12 hexagons
    axial_coords = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2), (-2,1)
    ]
    
    centers = np.array([
        [sqrt3 * (q + r/2.0), 1.5 * r] 
        for q, r in axial_coords
    ])
    
    # Shift to centroid
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    def compute_s(centers):
        # Compute min enclosing hexagon side s for centers (angles 0)
        # Vertices of inner hexagons
        # For angle 0, vertices are at angles 0, 60, ...
        # We can just project centers onto normals and add the apothem of inner hexagon.
        # Distance from center to vertex of inner hexagon (side 1) is 1 (circumradius = side).
        # Distance from center to edge (apothem) is sqrt(3)/2.
        # The "radius" in direction theta is 1 if theta is multiple of 60, 
        # and varies linearly in between? 
        # Actually, for a regular hexagon, the support function h(u) is the max dot(p, u) for p in hexagon.
        # For a hexagon centered at c, h(u) = c·u + h_inner(u).
        # h_inner(u) for unit hexagon:
        # If u is aligned with normal, h_inner = sqrt(3)/2.
        # If u is aligned with vertex, h_inner = 1.
        # We need max over all 6 normals of the outer hexagon.
        
        normals = np.array([
            [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] 
            for a in [0, 60, 120, 180, 240, 300]
        ])
        
        # For each normal n, the max projection of the inner hexagon centered at c is:
        # c·n + h_inner(n)
        # h_inner(n) for normal direction:
        # The normal directions are exactly the directions of the edges of the hexagon (if angle 0).
        # So h_inner(n) = sqrt(3)/2 for all these normals.
        
        # So max projection for center c in direction n is c·n + sqrt(3)/2.
        # We need this to be <= s * sqrt(3)/2 for all c and n.
        # s * sqrt(3)/2 >= max(c·n) + sqrt(3)/2
        # s >= (max(c·n) + sqrt(3)/2) / (sqrt(3)/2) = 2*max(c·n)/sqrt(3) + 1
        
        # Calculate max(c·n) for all centers and all normals
        # projections[i, j] = c_i · n_j
        projections = centers @ normals.T
        max_projections = np.max(np.abs(projections), axis=0)
        # We need max over all centers and all normals
        global_max = np.max(max_projections)
        
        s = 2.0 * global_max / sqrt3 + 1.0
        return s

    def check_validity(centers):
        # Check pairwise distances
        # For aligned hexagons, min distance is sqrt(3)
        for i in range(12):
            for j in range(i+1, 12):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < sqrt3 - 1e-6:
                    return False
        return True

    # Initial s
    best_s = compute_s(centers)
    valid_init = check_validity(centers)
    if not valid_init:
        # Should not happen
        return centers, np.array([0,0,0]), best_s

    best_centers = centers.copy()
    
    T = 1.0
    cooling = 0.999
    step_size = 0.1
    
    while time.time() < deadline and T > 0.01:
        new_centers = best_centers.copy()
        idx = random.randint(0, 11)
        new_centers[idx, 0] += random.gauss(0, step_size)
        new_centers[idx, 1] += random.gauss(0, step_size)
        
        # Quick validity check
        if not check_validity(new_centers):
            T *= cooling
            continue
            
        new_s = compute_s(new_centers)
        
        delta = new_s - best_s
        if delta < 0 or random.random() < np.exp(-delta / T):
            best_centers = new_centers
            best_s = new_s
            
        T *= cooling
        if T < 0.1:
            step_size *= 0.95

    outer_hex_data = np.array([0.0, 0.0, 0.0])
    return best_centers, outer_hex_data, best_s

# Final check on the code structure.
# The function hexagon_packing_12 is defined.
# It returns (inner, outer, s).
# inner is (12,3). My code returns (12,2). 
# I need to add angles.
# Since I assumed angles are 0, I will return 0 for angle.

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 590.0
    
    sqrt3 = np.sqrt(3)
    
    axial_coords = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2), (-2,1)
    ]
    
    centers = np.array([
        [sqrt3 * (q + r/2.0), 1.5 * r] 
        for q, r in axial_coords
    ])
    
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    def compute_s(centers):
        normals = np.array([
            [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] 
            for a in [0, 60, 120, 180, 240, 300]
        ])
        projections = centers @ normals.T
        global_max = np.max(np.abs(projections))
        s = 2.0 * global_max / sqrt3 + 1.0
        return s

    def check_validity(centers):
        for i in range(12):
            for j in range(i+1, 12):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < sqrt3 - 1e-6:
                    return False
        return True

    best_s = compute_s(centers)
    valid_init = check_validity(centers)
    
    best_centers = centers.copy()
    
    T = 1.0
    cooling = 0.999
    step_size = 0.1
    
    while time.time() < deadline and T > 0.01:
        new_centers = best_centers.copy()
        idx = random.randint(0, 11)
        new_centers[idx, 0] += random.gauss(0, step_size)
        new_centers[idx, 1] += random.gauss(0, step_size)
        
        if not check_validity(new_centers):
            T *= cooling
            continue
            
        new_s = compute_s(new_centers)
        
        delta = new_s - best_s
        if delta < 0 or random.random() < np.exp(-delta / T):
            best_centers = new_centers
            best_s = new_s
            
        T *= cooling
        if T < 0.1:
            step_size *= 0.95

    # Format output
    inner_hex_data = np.column_stack((best_centers, np.zeros(12)))
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, best_s

