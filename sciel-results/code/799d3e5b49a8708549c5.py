import math
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Helper: Get vertices of a regular hexagon with side 1, center (cx, cy), angle theta
    def get_hex_vertices(cx, cy, theta):
        verts = []
        for i in range(6):
            angle = theta + i * math.pi / 3
            verts.append((cx + math.cos(angle), cy + math.sin(angle)))
        return np.array(verts)

    # Helper: Check if two convex polygons (hexagons) overlap using SAT
    def polygons_overlap(poly1, poly2):
        # Get all edges (normals) from both polygons
        # For a regular hexagon, we can just use the 6 normal directions
        # But since they might be rotated, we compute normals from edges
        def get_normals(poly):
            normals = []
            for i in range(len(poly)):
                p1 = poly[i]
                p2 = poly[(i+1)%len(poly)]
                edge = p2 - p1
                # Normal is perpendicular to edge
                # (dy, -dx)
                n = np.array([edge[1], -edge[0]])
                n = n / np.linalg.norm(n)
                normals.append(n)
            return normals
        
        normals1 = get_normals(poly1)
        normals2 = get_normals(poly2)
        
        all_normals = normals1 + normals2
        
        for n in all_normals:
            # Project both polygons onto n
            proj1 = np.dot(poly1, n)
            proj2 = np.dot(poly2, n)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for separation
            if max1 < min2 or max2 < min1:
                return False
                
        return True

    # Helper: Calculate the side length L of the minimal enclosing regular hexagon
    # with orientation outer_theta that contains all given points (vertices)
    def calc_enclosing_size(points, outer_theta):
        # The 3 normal directions for the outer hexagon
        # A regular hexagon with "flat top" or "pointy top"?
        # The task says "regular hexagon". The orientation is defined by angle.
        # Let's define the normal directions relative to the outer angle.
        # If outer angle is 0, let's say normals are at 0, 60, 120 deg?
        # Actually, the distance from center to edge (apothem) is L * sqrt(3) / 2.
        # The width in direction n is 2 * apothem = L * sqrt(3).
        # So L = max_width / sqrt(3).
        
        max_width = 0
        for k in range(3):
            # Normals are separated by 60 degrees.
            # Let's align them such that one is at outer_theta + 30 deg? 
            # It doesn't matter for the *minimal* size over all rotations, 
            # but we need a consistent definition.
            # Let's use angles: outer_theta, outer_theta + 60, outer_theta + 120.
            # Wait, the edge normals of a hexagon are separated by 60 degrees.
            # The distance from center to edge is $d = L \frac{\sqrt{3}}{2}$.
            # The projection of a point P onto a normal n is P.n.
            # For the hexagon to contain P, |P - C| . n <= d for all 6 normals.
            # This implies the width in direction n (max P.n - min P.n) <= 2d = L * sqrt(3).
            # So L >= width / sqrt(3).
            
            angle = outer_theta + k * math.pi / 3
            nx = math.cos(angle)
            ny = math.sin(angle)
            
            projections = points[:, 0] * nx + points[:, 1] * ny
            width = np.max(projections) - np.min(projections)
            
            if width > max_width:
                max_width = width
                
        return max_width / math.sqrt(3)

    # Initialize 12 hexagons
    # Start with a 2x6 grid roughly
    # Hex width (point to point) is 2. Height (flat to flat) is sqrt(3).
    # Let's place them in a honeycomb pattern.
    
    positions = []
    angles = []
    
    # Row 0
    for i in range(6):
        x = i * 1.5
        y = 0
        positions.append([x, y])
        angles.append(0.0)
        
    # Row 1 (offset)
    for i in range(6):
        x = i * 1.5 + 0.75
        y = math.sqrt(3) / 2
        positions.append([x, y])
        angles.append(0.0)
        
    positions = np.array(positions)
    angles = np.array(angles)
    
    # Calculate initial size
    # We need to optimize the outer angle too.
    # Let's fix outer angle to 0 for now, or sample a few.
    # To make it robust, we will sample outer angle in the evaluation step.
    
    def get_total_vertices(pos, ang):
        verts_list = []
        for i in range(12):
            v = get_hex_vertices(pos[i][0], pos[i][1], ang[i])
            verts_list.extend(v)
        return np.array(verts_list)
    
    def evaluate_config(pos, ang):
        # Check overlaps
        verts_list = [get_hex_vertices(pos[i][0], pos[i][1], ang[i]) for i in range(12)]
        for i in range(12):
            for j in range(i+1, 12):
                if polygons_overlap(verts_list[i], verts_list[j]):
                    return float('inf') # Overlap penalty
        
        # Calculate minimal enclosing hex size
        # We can optimize the outer angle. 
        # Since the configuration is symmetric or near symmetric, 
        # trying a few outer angles is good.
        all_verts = get_total_vertices(pos, ang)
        
        # Sample outer angles to find the best fit orientation
        best_size = float('inf')
        for outer_deg in range(0, 60, 5):
            outer_rad = math.radians(outer_deg)
            size = calc_enclosing_size(all_verts, outer_rad)
            if size < best_size:
                best_size = size
                
        return best_size

    current_size = evaluate_config(positions, angles)
    best_size = current_size
    best_positions = positions.copy()
    best_angles = angles.copy()
    
    # Simulated Annealing / Hill Climbing
    # We want to minimize size.
    # Temperature starts high, decays.
    
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    while time.time() < deadline:
        # Pick a random hexagon to move
        i = random.randint(0, 11)
        
        # Random perturbation
        dx = random.gauss(0, 0.2)
        dy = random.gauss(0, 0.2)
        dtheta = random.gauss(0, 0.1)
        
        new_pos = positions.copy()
        new_ang = angles.copy()
        
        new_pos[i][0] += dx
        new_pos[i][1] += dy
        new_ang[i] += dtheta
        
        new_size = evaluate_config(new_pos, new_ang)
        
        # Acceptance criteria
        if new_size < current_size:
            positions = new_pos
            angles = new_ang
            current_size = new_size
            
            if current_size < best_size:
                best_size = current_size
                best_positions = positions.copy()
                best_angles = angles.copy()
        else:
            # Metropolis criterion: accept worse with probability exp(-delta/T)
            delta = new_size - current_size
            if delta < 0: # Should not happen if not <, but just in case
                positions = new_pos
                angles = new_ang
                current_size = new_size
            else:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    positions = new_pos
                    angles = new_ang
                    current_size = new_size
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final reconstruction
    # We need to return the inner hexagons, outer center, outer side, outer angle.
    # The `calc_enclosing_size` function assumes the hexagon is centered at the origin?
    # No, `width` is translation invariant. 
    # The center of the enclosing hexagon is the midpoint of the min/max projections.
    
    all_verts = get_total_vertices(best_positions, best_angles)
    
    # Find the best outer angle and center
    best_outer_angle = 0
    best_center = np.array([0, 0])
    best_final_size = float('inf')
    
    for outer_deg in range(0, 60, 1): # Fine grid for final result
        outer_rad = math.radians(outer_deg)
        size = calc_enclosing_size(all_verts, outer_rad)
        if size < best_final_size:
            best_final_size = size
            best_outer_angle = outer_rad
            
            # Calculate center
            # C . n = (min + max) / 2 for the 3 axes
            # We use 2 axes to solve for C
            centers = []
            for k in range(2):
                angle = outer_rad + k * math.pi / 3
                nx = math.cos(angle)
                ny = math.sin(angle)
                projections = all_verts[:, 0] * nx + all_verts[:, 1] * ny
                min_p = np.min(projections)
                max_p = np.max(projections)
                mid_p = (min_p + max_p) / 2.0
                centers.append((nx, ny, mid_p))
                
            A = np.array([centers[0][:2], centers[1][:2]])
            b = np.array([centers[0][2], centers[1][2]])
            try:
                C = np.linalg.lstsq(A, b, rcond=None)[0]
                best_center = C
            except:
                best_center = np.mean(all_verts, axis=0)
                
    # Add small margin
    final_side = best_final_size * 1.001
    
    inner = []
    for i in range(12):
        inner.append((best_positions[i][0], best_positions[i][1], math.degrees(best_angles[i])))
        
    return inner, (best_center[0], best_center[1]), final_side, math.degrees(best_outer_angle)
