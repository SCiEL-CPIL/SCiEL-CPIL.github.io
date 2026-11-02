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
        base_angles = np.radians(np.arange(0, 360, 60) + angle_deg)
        x = cx + np.cos(base_angles)
        y = cy + np.sin(base_angles)
        return np.column_stack((x, y))

    # Precomputed unit vectors for hexagon edges/normals
    # For a regular hexagon with vertices at angles 0, 60, ..., 300:
    # The edge from vertex i to i+1 has a normal direction.
    # Let's precompute the 6 outward normals for a hexagon with angle 0.
    # Vertices: (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # Edge 0: v0->v1. Vector: (-0.5, sqrt(3)/2). Normal: (-sqrt(3)/2, -0.5) -> inward?
    # Let's just compute them properly.
    
    # Helper: Check if two hexagons are disjoint using SAT
    # Fast rejection: Bounding box check
    def hexagons_disjoint(h1, h2):
        # h1, h2 are (x, y, angle)
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # If far apart, disjoint
        if dist_sq > 4.0: # dist > 2.0 (max distance between centers for touching is 2*radius=2)
            return True
            
        # If very close, likely overlap, but let's check precisely
        # Get vertices
        v1 = get_hex_vertices(h1[0], h1[1], h1[2])
        v2 = get_hex_vertices(h2[0], h2[1], h2[2])
        
        # SAT: Check 6 normals from each hexagon
        # We can precompute the normals for a unit hexagon at angle 0, then rotate them.
        # Or just compute from vertices. Computing from vertices is safer but slower.
        # Let's compute normals from v1 and v2.
        
        # Optimization: Only compute v1, v2 if necessary.
        # We already did fast rejection.
        
        # Normals for hexagon 1
        # For i in range(6):
        #   edge = v1[i+1] - v1[i]
        #   normal = (-edge_y, edge_x) normalized
        #   project both hexagons onto normal
        
        # To speed up, we can check bounding boxes first?
        # Bounding box of hexagon with side 1:
        # If angle 0 (pointy top): width=sqrt(3), height=2
        # If angle 30 (flat top): width=2, height=sqrt(3)
        # General: width/height depends on angle.
        # Let's stick to SAT for correctness, but optimize the inner loop.
        
        # Precompute edge normals for v1 and v2
        # Actually, we can just iterate.
        
        for i in range(6):
            ex = v1[(i+1)%6, 0] - v1[i, 0]
            ey = v1[(i+1)%6, 1] - v1[i, 1]
            nx = -ey
            ny = ex
            norm = math.sqrt(nx*nx + ny*ny)
            if norm == 0: continue
            nx /= norm
            ny /= norm
            
            # Project v1 and v2 onto normal
            proj1 = v1[:, 0]*nx + v1[:, 1]*ny
            proj2 = v2[:, 0]*nx + v2[:, 1]*ny
            
            if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
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
            
            proj1 = v1[:, 0]*nx + v1[:, 1]*ny
            proj2 = v2[:, 0]*nx + v2[:, 1]*ny
            
            if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                return True
                
        return False

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
                
                proj = offsets[:, 0] * nx + offsets[:, 1] * ny
                max_proj = np.max(np.abs(proj))
                
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

    # Initial Configuration: Honeycomb-like
    # Let's try a hexagonal packing.
    # Center at 0,0.
    # 1 hex at center.
    # 6 hexes around it.
    # 5 more to make 12?
    # Or a 3x4 grid?
    # Let's use a grid as a robust starting point.
    cols = 4
    rows = 3
    # Pitch: 3.0 horizontal, sqrt(3) vertical
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
    T_min = 0.001
    cooling = 0.9995
    step_size = 0.5
    
    step_count = 0
    max_steps = 50000 # Hard limit
    
    # Add a local polish step using Nelder-Mead on the best configuration found?
    # Or just run SA longer.
    
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
