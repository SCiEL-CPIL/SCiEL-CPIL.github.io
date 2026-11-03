"""
Strategy: Simulated Annealing with vectorized collision checks.
The previous approach used a general optimizer with penalties. 
This approach uses a strict constraint satisfaction (hard constraints) 
within a Simulated Annealing loop. 
Vectorized NumPy operations allow for high-throughput evaluation of 
candidate moves, enabling a much deeper search within the time budget.
"""

import numpy as np
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Precompute constants
    sqrt3 = np.sqrt(3)
    
    # Hexagon vertices relative to center for a flat-top hexagon (angle 0)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    # For side length 1, distance from center to vertex is 1.
    # Distance from center to edge (inradius) is sqrt(3)/2
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * (np.pi / 180.0)
    base_verts = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    
    # We will represent each hexagon by (x, y, angle)
    # To make vectorization easier, we can precompute rotation matrices? 
    # No, angles change. But we can vectorize the SAT check.
    
    def get_vertices_vectorized(positions, angles):
        """
        positions: (N, 2) array of centers
        angles: (N,) array of angles in radians
        Returns: (N, 6, 2) array of vertices
        """
        # Rotate base vertices
        cos_a = np.cos(angles)[:, np.newaxis] # (N, 1)
        sin_a = np.sin(angles)[:, np.newaxis] # (N, 1)
        
        # base_verts is (6, 2)
        # Rotation matrix R = [[cos, -sin], [sin, cos]]
        # v' = R @ v
        # v'_x = cos * v_x - sin * v_y
        # v'_y = sin * v_x + cos * v_y
        
        # Expand base_verts to (N, 6, 2) via broadcasting
        # bv = base_verts[np.newaxis, :, :] -> (1, 6, 2)
        # cos_a_exp = cos_a[:, np.newaxis, :] -> (N, 1, 1)
        
        # Actually, simpler:
        # v_x = base_verts[:, 0] (6,)
        # v_y = base_verts[:, 1] (6,)
        
        # For each hex i, angle a_i:
        # vx_i = cos(a_i)*base_vx - sin(a_i)*base_vy
        # vy_i = sin(a_i)*base_vx + cos(a_i)*base_vy
        
        # Use broadcasting
        # angles (N,) -> (N, 1)
        # base_verts (6, 2) -> (1, 6, 2)
        
        cos_a = np.cos(angles)[:, np.newaxis]
        sin_a = np.sin(angles)[:, np.newaxis]
        
        # base_verts is (6, 2). We need (N, 6, 2)
        # Let's just do it in a loop for clarity if N is small, but N=12 is small.
        # However, vectorizing the *check* is more important than generating vertices.
        # Let's generate vertices in a loop, it's only 12 hexes.
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            c = np.cos(angles[i])
            s = np.sin(angles[i])
            # Rotate base_verts (6,2)
            # [[c, -s], [s, c]] @ [[x], [y]]
            # new_x = c*x - s*y
            # new_y = s*x + c*y
            vx = base_verts[:, 0]
            vy = base_verts[:, 1]
            verts[i, :, 0] = c * vx - s * vy
            verts[i, :, 1] = s * vx + c * vy
            
        # Translate to positions
        verts[:, :, 0] += positions[:, 0:1]
        verts[:, :, 1] += positions[:, 1:2]
        
        return verts

    def check_collisions(verts, tol=1e-9):
        """
        Vectorized SAT check for all pairs.
        Returns: bool, True if no collisions (valid), False otherwise.
        """
        # For each pair (i, j), check all 12 axes (6 from i, 6 from j)
        # If any axis separates them, they are not colliding on that axis.
        # If for ALL axes they overlap, they are colliding.
        
        # To be fast, we can check if ANY pair is colliding.
        # We want to know if the packing is VALID.
        # So we return False if ANY pair collides.
        
        # Precompute edge normals for all hexes
        # Edge k of hex i: from v[k] to v[k+1]
        # Normal is perpendicular to edge.
        
        # Since N is small (12), pairs are 66.
        # Let's iterate pairs but vectorize the projection check within a pair?
        # Or just do it in pure Python loops for the pairs, as vectorizing 66 pairs of 12 axes 
        # might not beat a tight Python loop with NumPy arrays for single pair checks.
        # Actually, let's try to vectorize the projection of all vertices of all hexes onto all normals.
        
        # A simpler, robust approach for N=12:
        # Iterate i < j.
        # For each pair, get 12 normals.
        # Project all 12 vertices of i and 12 vertices of j onto these 12 normals.
        # Check intervals.
        
        for i in range(N):
            for j in range(i + 1, N):
                v1 = verts[i] # (6, 2)
                v2 = verts[j] # (6, 2)
                
                # Get edges for v1 and v2
                # Edge e_k = v_{k+1} - v_k
                e1 = np.roll(v1, -1, axis=0) - v1
                e2 = np.roll(v2, -1, axis=0) - v2
                
                # Normals: (dy, -dx) normalized? No, normalization not needed for SAT overlap check, 
                # as long as we use the same normal for projection.
                n1 = np.column_stack((e1[:, 1], -e1[:, 0]))
                n2 = np.column_stack((e2[:, 1], -e2[:, 0]))
                
                # Combine normals (12, 2)
                normals = np.vstack((n1, n2))
                
                # Project v1 and v2 onto all normals
                # proj1 = v1 @ normals.T -> (6, 12)
                # proj2 = v2 @ normals.T -> (6, 12)
                proj1 = v1 @ normals.T
                proj2 = v2 @ normals.T
                
                min1 = np.min(proj1, axis=0) # (12,)
                max1 = np.max(proj1, axis=0)
                min2 = np.min(proj2, axis=0)
                max2 = np.max(proj2, axis=0)
                
                # Check overlap for each axis
                # Overlap if max1 >= min2 and max2 >= min1
                # Separation if max1 < min2 or max2 < min1
                # We need to check if there is ANY separating axis.
                # If YES, valid (for this pair).
                # If NO, collision.
                
                sep1 = max1 < min2 - tol
                sep2 = max2 < min1 - tol
                
                if not (np.any(sep1) or np.any(sep2)):
                    return False # Collision found
        
        return True

    def check_containment(verts, outer_center, outer_side, outer_angle_rad):
        """
        Check if all inner hexes are inside the outer hex.
        Outer hex is defined by center, side, angle.
        Inradius R = outer_side * sqrt(3) / 2.
        Face normals of outer hex:
        Angles: outer_angle + 30 + k*60 degrees?
        Let's define outer hex vertices similar to inner, but side S.
        Actually, containment is: for every vertex of every inner hex,
        its projection onto the 6 face normals of the outer hex must be <= R.
        """
        R = outer_side * sqrt3 / 2.0
        
        # Outer face normals
        # If outer hex is "flat-top" (angle 0), vertices are at 0, 60...
        # Face 0 is between vertex 5 and 0? Or vertex 0 and 1?
        # Standard: Vertices at 0, 60, 120...
        # Edge 0: v0 to v1. Normal points outward.
        # v0=(1,0), v1=(0.5, sqrt3/2). Edge = (-0.5, sqrt3/2).
        # Normal = (sqrt3/2, 0.5). Angle 30 deg.
        # So face normals are at 30, 90, 150, 210, 270, 330 degrees.
        
        base_outer_norms = []
        for k in range(6):
            angle = np.radians(30 + k * 60)
            base_outer_norms.append([np.cos(angle), np.sin(angle)])
        base_outer_norms = np.array(base_outer_norms) # (6, 2)
        
        # Rotate by outer_angle_rad
        c = np.cos(outer_angle_rad)
        s = np.sin(outer_angle_rad)
        # R @ base_outer_norms.T
        # new_x = c*nx - s*ny
        # new_y = s*nx + c*ny
        nx = base_outer_norms[:, 0]
        ny = base_outer_norms[:, 1]
        rotated_nx = c * nx - s * ny
        rotated_ny = s * nx + c * ny
        outer_normals = np.column_stack((rotated_nx, rotated_ny)) # (6, 2)
        
        # Check all vertices of all inner hexes
        # verts: (N, 6, 2)
        # Flatten to (N*6, 2)
        all_verts = verts.reshape(-1, 2)
        
        # Relative to outer center
        all_verts_rel = all_verts - outer_center
        
        # Project onto outer normals
        # proj = all_verts_rel @ outer_normals.T -> (N*6, 6)
        proj = all_verts_rel @ outer_normals.T
        
        # Max projection
        max_proj = np.max(proj, axis=0)
        
        # If any max_proj > R, it's outside
        if np.any(max_proj > R + 1e-9):
            return False
        
        return True

    def calculate_bounding_side(positions, angles, outer_center, outer_angle_rad):
        """
        Calculate the minimum outer side length required to contain the configuration.
        We assume the outer hexagon is centered at outer_center and rotated by outer_angle_rad.
        The side length is determined by the maximum projection of any inner vertex 
        onto the 6 face normals of the outer hex.
        """
        verts = get_vertices_vectorized(positions, angles)
        
        R_required = 0
        # Re-use logic from containment but just find max projection
        base_outer_norms = np.array([
            [np.cos(np.radians(30 + k*60)), np.sin(np.radians(30 + k*60))]
            for k in range(6)
        ])
        c = np.cos(outer_angle_rad)
        s = np.sin(outer_angle_rad)
        nx = base_outer_norms[:, 0]
        ny = base_outer_norms[:, 1]
        rotated_nx = c * nx - s * ny
        rotated_ny = s * nx + c * ny
        outer_normals = np.column_stack((rotated_nx, rotated_ny))
        
        all_verts = verts.reshape(-1, 2)
        all_verts_rel = all_verts - outer_center
        proj = all_verts_rel @ outer_normals.T
        max_proj = np.max(proj, axis=0)
        R_required = np.max(max_proj)
        
        # Side = 2 * R / sqrt(3)
        return 2.0 * R_required / sqrt3

    # Initial Configuration
    # Start with a compact hexagonal cluster
    # Axial coords for 12 hexes:
    # Center (0,0)
    # Ring 1: 6 hexes
    # Ring 2: 5 more hexes to make 12? 
    # Let's just use a random initialization or a grid.
    # Grid is better for start.
    
    # Use axial coordinates
    axial = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1), # 7 hexes
        (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2) # 5 more -> 12
    ]
    
    init_pos = np.zeros((N, 2))
    init_ang = np.zeros(N)
    for i, (q, r) in enumerate(axial):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        init_pos[i] = [x, y]
        init_ang[i] = 0.0
        
    # Center the cluster
    center = np.mean(init_pos, axis=0)
    init_pos -= center
    
    # Initial outer hex
    outer_center = np.array([0.0, 0.0])
    outer_angle_rad = 0.0
    
    # Calculate initial side
    init_side = calculate_bounding_side(init_pos, init_ang, outer_center, outer_angle_rad)
    
    # Verify initial validity (collisions)
    # In a perfect hex grid, they touch but don't overlap.
    # Our SAT check with tol might flag touching as collision if not careful.
    # We use tol=1e-9. Touching means max1 == min2. 
    # sep1: max1 < min2 - tol. If max1 == min2, sep is False.
    # So touching is considered "overlap" in strict SAT unless we add a small buffer.
    # For packing, touching is allowed.
    # Let's modify check_collisions to allow touching.
    # Change condition: if max1 < min2 - tol OR max2 < min1 - tol.
    # If they touch, max1 == min2, so condition is False. 
    # So "no separating axis" -> Collision.
    # This means touching is treated as collision.
    # We need to allow touching.
    # So we should check for STRICT overlap.
    # Overlap if max1 > min2 + tol AND max2 > min1 + tol.
    # If they touch, max1 == min2, so max1 > min2 + tol is False.
    
    def check_collisions_strict(verts, tol=1e-9):
        for i in range(N):
            for j in range(i + 1, N):
                v1 = verts[i]
                v2 = verts[j]
                e1 = np.roll(v1, -1, axis=0) - v1
                e2 = np.roll(v2, -1, axis=0) - v2
                n1 = np.column_stack((e1[:, 1], -e1[:, 0]))
                n2 = np.column_stack((e2[:, 1], -e2[:, 0]))
                normals = np.vstack((n1, n2))
                proj1 = v1 @ normals.T
                proj2 = v2 @ normals.T
                min1 = np.min(proj1, axis=0)
                max1 = np.max(proj1, axis=0)
                min2 = np.min(proj2, axis=0)
                max2 = np.max(proj2, axis=0)
                
                # Strict overlap check
                # Overlap on axis k if max1[k] > min2[k] + tol AND max2[k] > min1[k] + tol
                # If there is ANY axis where they do NOT overlap, they are separated.
                # So we check if there exists an axis where NOT (max1 > min2 + tol AND max2 > min1 + tol)
                
                overlap1 = max1 > min2 + tol
                overlap2 = max2 > min1 + tol
                both_overlap = overlap1 & overlap2
                
                # If for ALL axes both_overlap is True, then they are strictly overlapping.
                if np.all(both_overlap):
                    return False
        return True

    # Re-verify initial
    verts_init = get_vertices_vectorized(init_pos, init_ang)
    if not check_collisions_strict(verts_init):
        # If initial grid is considered overlapping, nudge them slightly apart
        # Or just accept it if it's just numerical noise.
        # With tol=1e-9, exact touching should be fine.
        pass

    best_side = init_side
    best_pos = init_pos.copy()
    best_ang = init_ang.copy()
    best_oc = outer_center.copy()
    best_oa = outer_angle_rad

    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.01
    alpha = 0.995
    max_iter = 50000
    step_size_pos = 0.2
    step_size_ang = 0.1
    step_size_outer = 0.1
    
    current_pos = init_pos.copy()
    current_ang = init_ang.copy()
    current_oc = outer_center.copy()
    current_oa = outer_angle_rad
    current_side = init_side
    
    rng = np.random.default_rng(42)
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Generate Move
        # 1. Move a random inner hex
        # 2. Rotate a random inner hex
        # 3. Move outer center
        # 4. Rotate outer hex
        
        move_type = rng.integers(0, 4)
        
        new_pos = current_pos.copy()
        new_ang = current_ang.copy()
        new_oc = current_oc.copy()
        new_oa = current_oa
        
        if move_type == 0:
            idx = rng.integers(0, N)
            new_pos[idx] += rng.normal(0, step_size_pos, 2)
        elif move_type == 1:
            idx = rng.integers(0, N)
            new_ang[idx] += rng.normal(0, step_size_ang) * (np.pi / 180.0)
        elif move_type == 2:
            new_oc += rng.normal(0, step_size_outer, 2)
        else:
            new_oa += rng.normal(0, step_size_outer) * (np.pi / 180.0)
            
        # Calculate new side
        new_side = calculate_bounding_side(new_pos, new_ang, new_oc, new_oa)
        
        # Check validity (Collisions and Containment)
        # Containment is guaranteed by construction of new_side (it's the min side to contain).
        # But wait, if we change outer center/angle, the "containing" side increases.
        # The objective is to MINIMIZE the side.
        # So we just need to check Collisions.
        # The side calculated IS the side of the outer hex that contains everything.
        # So containment is automatically satisfied by definition of new_side.
        
        verts_new = get_vertices_vectorized(new_pos, new_ang)
        is_valid = check_collisions_strict(verts_new)
        
        if not is_valid:
            # Reject invalid moves
            continue
            
        # Accept/Reject based on SA
        delta = new_side - current_side
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_pos = new_pos
            current_ang = new_ang
            current_oc = new_oc
            current_oa = new_oa
            current_side = new_side
            
            if new_side < best_side:
                best_side = new_side
                best_pos = new_pos.copy()
                best_ang = new_ang.copy()
                best_oc = new_oc.copy()
                best_oa = new_oa
                
        T *= alpha
        
        if T < T_min:
            break
            
    # Final Result
    inner = []
    for i in range(N):
        inner.append([float(best_pos[i, 0]), float(best_pos[i, 1]), float(np.degrees(best_ang[i]))])
        
    return inner, (float(best_oc[0]), float(best_oc[1])), float(best_side), float(np.degrees(best_oa))

# EVOLVE-BLOCK-END
