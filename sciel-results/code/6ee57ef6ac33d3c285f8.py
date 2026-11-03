import numpy as np
import time
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    # Hexagon vertices for unit hexagon at origin, angle 0
    # Angles: 0, pi/3, ...
    angles = np.arange(0, 2 * np.pi, np.pi / 3)
    base_vertices = np.column_stack((np.cos(angles), np.sin(angles)))
    
    def get_hex_vertices(cx, cy, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        rot = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        verts = base_vertices @ rot.T + np.array([cx, cy])
        return verts

    def sat_overlap_depth(v1, v2):
        """
        v1, v2: arrays of shape (6, 2)
        Returns minimum penetration depth if overlapping, 0 otherwise.
        """
        min_depth = float('inf')
        
        # Get edges of v1
        edges1 = np.roll(v1, -1, axis=0) - v1
        # Get edges of v2
        edges2 = np.roll(v2, -1, axis=0) - v2
        
        # Combine all edges to get normals
        all_edges = np.vstack([edges1, edges2])
        
        # Normals (perpendicular to edges)
        # Edge (dx, dy) -> Normal (-dy, dx)
        normals = np.column_stack((-all_edges[:, 1], all_edges[:, 0]))
        
        # Normalize normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normals = normals / norms
        
        # Project both polygons onto each normal
        for n in normals:
            proj1 = v1 @ n
            proj2 = v2 @ n
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for separation
            if max1 < min2 or max2 < min1:
                return 0.0
            
            # Calculate overlap depth on this axis
            depth = min(max1, max2) - max(min1, min2)
            min_depth = min(min_depth, depth)
            
        return min_depth if min_depth < float('inf') else 0.0

    def calculate_cost(config):
        """
        config: list of (x, y, angle_deg)
        Returns: (S, penalty)
        """
        verts_list = []
        for x, y, ang in config:
            verts_list.append(get_hex_vertices(x, y, ang))
        
        # Check overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                depth = sat_overlap_depth(verts_list[i], verts_list[j])
                if depth > 0:
                    penalty += depth ** 2
        
        # Calculate minimal S
        # Collect all vertices
        all_verts = np.vstack(verts_list)
        
        min_S = float('inf')
        # Search for optimal outer angle
        # Due to symmetry, we only need to check 0 to 60 degrees
        # But the inner config might break symmetry, so we check the full range?
        # No, the outer hexagon has 6-fold symmetry. Rotating it by 60 deg is equivalent.
        # So we check 0 to 60 deg.
        # To be safe and fast, we check 100 steps.
        for phi_deg in np.linspace(0, 60, 100, endpoint=False):
            phi_rad = np.deg2rad(phi_deg)
            # Normals of outer hexagon sides
            # Sides are perpendicular to normals.
            # Normals at angles: phi, phi+60, ...
            normal_angles = phi_rad + np.arange(0, 360, 60) * np.pi / 180.0
            normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
            
            # Project all vertices onto normals
            # We need max |v . n|
            # v . n for all v and all n
            # all_verts: (M, 2), normals: (6, 2)
            # We want max over M and 6 of |v_i . n_j|
            # This is equivalent to max over i of max over j |v_i . n_j|
            
            # Efficiently:
            # For each normal n, find max |v . n|
            # v . n is (M, 6) matrix
            dots = np.abs(all_verts @ normals.T)
            max_proj = np.max(dots)
            
            S_candidate = max_proj * 2.0 / np.sqrt(3.0)
            if S_candidate < min_S:
                min_S = S_candidate
                
        return min_S, penalty

    # Initialize configuration: Hexagonal lattice
    # Spacing sqrt(3)
    spacing = np.sqrt(3.0)
    config = []
    # Generate 11 points in a hexagonal pattern
    # Center (0,0)
    # 6 neighbors, 4 more?
    # Let's just generate a grid and pick 11
    # Or specific pattern:
    # 1 in center
    # 6 around it
    # 4 more around that?
    
    # Simpler: Start with a line or a cluster
    # Let's try a compact cluster
    # (0,0)
    # (sqrt(3), 0), (-sqrt(3), 0)
    # (0, sqrt(3)*1.5), (0, -sqrt(3)*1.5) -> Wait, y spacing for hex lattice
    # If x spacing is sqrt(3), y spacing for adjacent rows is 1.5?
    # Distance between (0,0) and (sqrt(3)/2, 1.5) is sqrt(3/4 + 2.25) = sqrt(3)
    # Yes.
    
    # Let's just use a simple heuristic initialization
    # Place them in a circle or grid
    for i in range(N):
        # Spiral or random?
        # Let's use a hex grid
        pass
    
    # Better initialization:
    # 1 at (0,0)
    # 6 at distance sqrt(3)
    # 4 at distance 2*sqrt(3) or similar
    
    init_pos = [
        (0, 0),
        (np.sqrt(3), 0),
        (-np.sqrt(3), 0),
        (0, 1.5), # Wait, (0, 1.5) is not distance sqrt(3) from (0,0)
        # Distance (0,0) to (0, 1.5) is 1.5 < 1.732. Overlap!
        # Correct hex lattice neighbors for (0,0) with spacing sqrt(3):
        # (sqrt(3), 0)
        # (sqrt(3)/2, 1.5) -> dist sqrt(3/4 + 2.25) = sqrt(3)
        # (-sqrt(3)/2, 1.5)
        # (-sqrt(3), 0)
        # (-sqrt(3)/2, -1.5)
        # (sqrt(3)/2, -1.5)
    ]
    
    # Let's define the 7 central hexagons
    pos7 = [
        (0, 0),
        (np.sqrt(3), 0),
        (np.sqrt(3)/2, 1.5),
        (-np.sqrt(3)/2, 1.5),
        (-np.sqrt(3), 0),
        (-np.sqrt(3)/2, -1.5),
        (np.sqrt(3)/2, -1.5)
    ]
    
    # Add 4 more around
    # (2*sqrt(3), 0)
    # (3*sqrt(3)/2, 3)
    # (-3*sqrt(3)/2, 3)
    # (-2*sqrt(3), 0)
    
    pos11 = pos7 + [
        (2*np.sqrt(3), 0),
        (3*np.sqrt(3)/2, 3.0),
        (-3*np.sqrt(3)/2, 3.0),
        (-2*np.sqrt(3), 0)
    ]
    
    # Check if these 11 overlap?
    # The new ones are at distance 2*sqrt(3) from center?
    # (2*sqrt(3), 0) is dist 2*sqrt(3) from (0,0).
    # It is dist sqrt(3) from (sqrt(3), 0).
    # It is dist sqrt( (2s - s/2)^2 + 3^2 )? No.
    # (2s, 0) and (s/2, 1.5). dx = 1.5s, dy = -1.5.
    # dist^2 = 2.25*3 + 2.25 = 6.75 + 2.25 = 9. dist = 3.
    # 3 > sqrt(3). So no overlap.
    
    config = [(x, y, 0.0) for x, y in pos11]
    
    best_cost, best_penalty = calculate_cost(config)
    best_S = best_cost
    best_config = [list(c) for c in config]
    
    T = 10.0
    cooling = 0.995
    
    while time.time() < deadline:
        # Pick random hexagon
        i = np.random.randint(N)
        # Perturb
        dx = np.random.normal(0, 0.5)
        dy = np.random.normal(0, 0.5)
        dangle = np.random.normal(0, 5.0) # degrees
        
        new_config = [list(c) for c in config]
        new_config[i][0] += dx
        new_config[i][1] += dy
        new_config[i][2] += dangle
        
        new_S, new_penalty = calculate_cost(new_config)
        new_cost = new_S + 100.0 * new_penalty
        
        old_S, old_penalty = calculate_cost(config)
        old_cost = old_S + 100.0 * old_penalty
        
        delta = new_cost - old_cost
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            config = new_config
            if new_cost < best_cost:
                best_cost = new_cost
                best_S = new_S
                best_config = [list(c) for c in config]
                
        T *= cooling
        
    # Final polish?
    # Return best config
    inner = []
    for x, y, ang in best_config:
        inner.append([x, y, ang])
        
    # Center of outer hexagon should be (0,0) for our calculation
    # Side length is best_S
    # Angle is 0 (we optimized over angle, but the container can be rotated to 0 if we report the config relative to it?
    # The interface returns: inner rows (x, y, angle_degrees), outer center (x, y), side length, rotation.
    # Our calculation assumed outer center (0,0) and optimized outer angle.
    # We should return the configuration that fits in a hexagon of side best_S.
    # The "rotation" of the outer hexagon in the return value can be 0, because we can just define the coordinate system such that the optimal outer hexagon is at 0.
    # Wait, if we found optimal Phi*, we can just rotate the whole system by -Phi* to make the outer hexagon angle 0.
    # Or simply return the best_config as is, and set outer rotation to the optimal Phi*?
    # The evaluator checks if the inner hexagons (with their given angles) fit inside the outer hexagon (with its given angle).
    # So we must return the optimal outer angle.
    
    # Let's re-find the optimal angle for best_config to be precise
    # And return that angle.
    
    # Recalculate optimal angle for best_config
    verts_list = []
    for x, y, ang in best_config:
        verts_list.append(get_hex_vertices(x, y, ang))
    all_verts = np.vstack(verts_list)
    
    min_S_final = float('inf')
    best_phi = 0.0
    for phi_deg in np.linspace(0, 60, 360, endpoint=False):
        phi_rad = np.deg2rad(phi_deg)
        normal_angles = phi_rad + np.arange(0, 360, 60) * np.pi / 180.0
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        dots = np.abs(all_verts @ normals.T)
        max_proj = np.max(dots)
        S_candidate = max_proj * 2.0 / np.sqrt(3.0)
        if S_candidate < min_S_final:
            min_S_final = S_candidate
            best_phi = phi_deg
            
    return inner, (0.0, 0.0), min_S_final, best_phi
# EVOLVE-BLOCK-END
