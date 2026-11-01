"""
Caging formulation: Optimize inner hexagon positions via Simulated Annealing 
to minimize the side length of the minimal containing regular hexagon.
The outer hexagon is optimized analytically for each inner configuration 
by scanning its rotation angle to minimize the maximum extent of the 
inner convex hull along the outer hexagon's face normals.
"""
import math
import time
import random
import numpy as np

N = 12
DEADLINE = 135.0  # Safety margin before 150s wall-clock kill

def construct():
    t0 = time.time()
    
    # Helper: Get vertices of a regular hexagon centered at (cx, cy) with side 1 and angle deg
    def get_hex_vertices(cx, cy, angle_deg):
        # Precompute cos/sin for angles
        # Angle is rotation of the hexagon. 
        # Standard hexagon vertices at angles 0, 60, 120... relative to center.
        # If angle_deg is the rotation, we add it to these base angles.
        base_angles = np.radians(np.arange(0, 360, 60) + angle_deg)
        x = cx + np.cos(base_angles)
        y = cy + np.sin(base_angles)
        return np.column_stack((x, y))

    # Helper: Check if two hexagons are disjoint
    # Fast rejection: Bounding box check
    # Hexagon side 1: Width = 2 (if flat top) or sqrt(3) (if pointy top). 
    # Actually, width depends on rotation. 
    # Max extent in any direction is 1 (radius). 
    # So if distance between centers > 2, they are disjoint.
    # If distance < sqrt(3) ~ 1.732, they might overlap? 
    # Min distance between centers for disjoint hexagons (touching edges) is sqrt(3).
    # So if dist < sqrt(3) - epsilon, they definitely overlap.
    # We will use a SAT check if dist < 2.0.
    
    def hexagons_disjoint(h1, h2):
        # h1, h2 are (x, y, angle)
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # If far apart, disjoint
        if dist_sq > 4.0: # dist > 2.0
            return True
            
        # If very close, likely overlap, but let's check precisely
        # Get vertices
        v1 = get_hex_vertices(h1[0], h1[1], h1[2])
        v2 = get_hex_vertices(h2[0], h2[1], h2[2])
        
        # SAT: Check 6 normals from each hexagon
        # Normals for hexagon 1
        for i in range(6):
            # Edge vector
            ex = v1[(i+1)%6, 0] - v1[i, 0]
            ey = v1[(i+1)%6, 1] - v1[i, 1]
            # Normal
            nx = -ey
            ny = ex
            norm = math.sqrt(nx*nx + ny*ny)
            if norm == 0: continue
            nx /= norm
            ny /= norm
            
            # Project v1 and v2 onto normal
            proj1_min = np.min(v1[:, 0]*nx + v1[:, 1]*ny)
            proj1_max = np.max(v1[:, 0]*nx + v1[:, 1]*ny)
            proj2_min = np.min(v2[:, 0]*nx + v2[:, 1]*ny)
            proj2_max = np.max(v2[:, 0]*nx + v2[:, 1]*ny)
            
            if proj1_max < proj2_min or proj2_max < proj1_min:
                return True
                
        for i in range(6):
            ex = v2[(i+1)%6, 0] - v2[i, 0]
            ey = v2[(i+1)%6, 1] - v2[i, 1]
            nx = -ey
            ny = ex
            norm = math.sqrt(nx*nx + ny*ny)
            if norm == 0: continue
            nx /= norm
            ny /= norm
            
            proj1_min = np.min(v1[:, 0]*nx + v1[:, 1]*ny)
            proj1_max = np.max(v1[:, 0]*nx + v1[:, 1]*ny)
            proj2_min = np.min(v2[:, 0]*nx + v2[:, 1]*ny)
            proj2_max = np.max(v2[:, 0]*nx + v2[:, 1]*ny)
            
            if proj1_max < proj2_min or proj2_max < proj1_min:
                return True
                
        return False

    def is_valid_configuration(inner):
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint(inner[i], inner[j]):
                    return False
        return True

    def compute_min_outer_side(inner):
        """
        Compute the minimum side length of a regular hexagon containing all inner hexagons.
        We optimize the rotation angle of the outer hexagon.
        The outer hexagon is centered at the centroid of the inner configuration.
        """
        # 1. Get all vertices of inner hexagons
        all_verts = []
        for h in inner:
            all_verts.append(get_hex_vertices(h[0], h[1], h[2]))
        all_verts = np.vstack(all_verts)
        
        # 2. Compute centroid (approximate center for outer hex)
        center = np.mean(all_verts, axis=0)
        offsets = all_verts - center
        
        # 3. Find optimal rotation angle phi for outer hexagon
        # Outer hexagon normals are at angles phi + 30 + k*60
        # Required side S = max_k ( 2/sqrt(3) * max_v | n_k . offset_v | )
        # We scan phi in [0, 60)
        
        sqrt3 = math.sqrt(3)
        best_side = float('inf')
        best_phi = 0.0
        
        # Coarse scan
        for phi_deg in range(0, 60, 2):
            phi = np.radians(phi_deg)
            max_extent = 0.0
            for k in range(6):
                normal_angle = phi + np.pi/6 + k * np.pi/3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                
                # Project all offsets
                proj = offsets[:, 0] * nx + offsets[:, 1] * ny
                max_proj = np.max(np.abs(proj))
                
                # S = 2/sqrt(3) * max_proj
                side_k = (2.0 / sqrt3) * max_proj
                if side_k > max_extent:
                    max_extent = side_k
            
            if max_extent < best_side:
                best_side = max_extent
                best_phi = phi_deg
                
        # Fine scan around best_phi
        for phi_deg in np.arange(best_phi - 2, best_phi + 2, 0.1):
            if phi_deg < 0 or phi_deg >= 60: continue
            phi = np.radians(phi_deg)
            max_extent = 0.0
            for k in range(6):
                normal_angle = phi + np.pi/6 + k * np.pi/3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                
                proj = offsets[:, 0] * nx + offsets[:, 1] * ny
                max_proj = np.max(np.abs(proj))
                
                side_k = (2.0 / sqrt3) * max_proj
                if side_k > max_extent:
                    max_extent = side_k
            
            if max_extent < best_side:
                best_side = max_extent
                best_phi = phi_deg

        return best_side, best_phi, center

    # Initial Configuration: 6x2 Grid
    # Flat-top hexagons. 
    # Horizontal pitch 3, Vertical pitch sqrt(3).
    cols = 6
    rows = 2
    x0 = -(cols - 1) / 2.0 * 3.0
    y0 = -(rows - 1) / 2.0 * math.sqrt(3)
    
    initial_inner = []
    for r in range(rows):
        for c in range(cols):
            x = x0 + c * 3.0
            y = y0 + r * math.sqrt(3)
            initial_inner.append([x, y, 0.0])
            
    best_inner = [list(h) for h in initial_inner]
    best_side, best_phi, best_center = compute_min_outer_side(best_inner)
    
    current_inner = [list(h) for h in initial_inner]
    current_side, _, _ = compute_min_outer_side(current_inner)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    step_size = 0.5
    
    step_count = 0
    max_steps = 20000 # Hard limit
    
    while time.time() - t0 < DEADLINE and step_count < max_steps:
        step_count += 1
        
        # Perturb one random hexagon
        idx = random.randint(0, N - 1)
        new_inner = [list(h) for h in current_inner]
        
        # Perturb position and angle
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 10.0) # degrees
        
        new_inner[idx][0] += dx
        new_inner[idx][1] += dy
        new_inner[idx][2] += da
        
        # Check validity (disjointness)
        valid = True
        # Only need to check the moved hexagon against others
        for j in range(N):
            if j == idx: continue
            if not hexagons_disjoint(new_inner[idx], new_inner[j]):
                valid = False
                break
                
        if not valid:
            # Reject
            T *= cooling
            if T < T_min: T = T_min
            continue
            
        # Compute new side length
        new_side, new_phi, new_center = compute_min_outer_side(new_inner)
        
        # Accept if better or with probability exp(-delta/T)
        if new_side < current_side:
            current_inner = new_inner
            current_side = new_side
            if new_side < best_side:
                best_side = new_side
                best_inner = [list(h) for h in current_inner]
                best_phi = new_phi
                best_center = new_center
        else:
            delta = new_side - current_side
            if delta < 0: delta = 0
            if random.random() < math.exp(-delta / T):
                current_inner = new_inner
                current_side = new_side
                if new_side < best_side:
                    best_side = new_side
                    best_inner = [list(h) for h in current_inner]
                    best_phi = new_phi
                    best_center = new_center
                    
        T *= cooling
        if T < T_min: T = T_min

    # Final Result
    # Ensure best_inner is a list of lists [x, y, angle]
    # Center is tuple (x, y)
    # Side is float
    # Angle is float (outer rotation)
    
    return best_inner, (float(best_center[0]), float(best_center[1])), float(best_side), float(best_phi)

# EVOLVE-BLOCK-END
