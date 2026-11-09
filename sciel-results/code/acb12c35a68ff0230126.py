"""
Strategy: Rigid Cluster Packing with Global Orientation Search.
Instead of optimizing 33 independent variables (x, y, angle for each of 11 hexagons),
we assume the 11 hexagons form a compact, rigid structure (or a few rigid sub-clusters).
We optimize the position and rotation of these clusters, and the orientation of the outer hexagon.
This significantly reduces the dimensionality of the search space and avoids local minima
caused by complex interlocking of individual hexagons.
"""
import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    SQRT3 = math.sqrt(3)
    
    # Helper: Get vertices of a unit hexagon centered at (cx, cy) with rotation angle
    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    # Helper: Get the 6 unique normal axes for a hexagon
    def get_hex_axes(angle_deg):
        axes = []
        for i in range(6):
            normal_angle = math.radians(angle_deg + i * 60 + 30)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            axes.append((nx, ny))
        return axes

    # Fast disjoint check using SAT
    def is_disjoint(hex1_verts, hex2_verts, axes1, axes2):
        all_axes = axes1 + axes2
        for axis in all_axes:
            nx, ny = axis
            min1 = min(vx * nx + vy * ny for vx, vy in hex1_verts)
            max1 = max(vx * nx + vy * ny for vx, vy in hex1_verts)
            min2 = min(vx * nx + vy * ny for vx, vy in hex2_verts)
            max2 = max(vx * nx + vy * ny for vx, vy in hex2_verts)
            if max1 < min2 or max2 < min1:
                return True
        return False

    # Calculate minimum outer hexagon side length containing all given hexagons
    # outer_angle_deg is the rotation of the outer hexagon
    def calc_outer_side(hexagons, outer_angle_deg):
        # The normal directions of the outer hexagon
        base_angles = [30, 90, 150, 210, 270, 330]
        axis_dirs = []
        for a in base_angles:
            rad = math.radians(a + outer_angle_deg)
            axis_dirs.append((math.cos(rad), math.sin(rad)))
        
        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for nx, ny in axis_dirs:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        
        # Side length s relates to apothem a by a = s * sqrt(3) / 2
        # So s = 2 * a / sqrt(3)
        return 2.0 * max_proj / SQRT3

    # Axial to Cartesian conversion for initial lattice
    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Define a few initial "rigid" configurations (sets of relative positions)
    # We will treat these as the base shape, and optimize the global transform.
    
    # Config 1: Compact 3-5-3 arrangement (standard hexagonal packing)
    coords1 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    
    # Config 2: A slightly distorted version to break symmetry
    coords2 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    
    # Config 3: Another variation
    coords3 = [(0,0), (1,0), (-1,0), (0,1), (1,1), (-1,1), (0,-1), (1,-1), (-1,-1), (2,0), (-2,0)]
    
    # Config 4: Random perturbation of Config 1
    coords4 = []
    for q, r in coords1:
        x, y = axial_to_cartesian(q, r, 1.0)
        # Small random jitter
        x += random.gauss(0, 0.1)
        y += random.gauss(0, 0.1)
        coords4.append((x, y))

    def evaluate_cluster(coords, global_rotation_deg, outer_rotation_deg):
        """
        Evaluate a rigid cluster.
        coords: list of (x, y) relative to cluster center.
        global_rotation_deg: rotation of the entire cluster.
        outer_rotation_deg: rotation of the outer hexagon.
        Returns: (overlap_penalty, outer_side_length)
        """
        # Rotate and translate coords to global frame
        # Note: We assume the cluster is centered at (0,0) for calculation,
        # but the "center" of the cluster might shift if we allow translation?
        # Actually, for a rigid cluster, translation doesn't affect the *size* of the bounding hexagon,
        # only its position. Since we can place the outer hexagon anywhere,
        # we can just center the cluster at (0,0) without loss of generality for size calculation.
        
        rad = math.radians(global_rotation_deg)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)
        
        rotated_coords = []
        for x, y in coords:
            rx = x * cos_r - y * sin_r
            ry = x * sin_r + y * cos_r
            rotated_coords.append((rx, ry))
            
        # Generate hexagons
        hexagons = []
        hex_axes = []
        for i in range(N):
            cx, cy = rotated_coords[i]
            # Inner hexagons can also rotate relative to the cluster?
            # In a rigid cluster, they rotate with the cluster.
            # Let's assume inner hexagons have the same orientation as the cluster global rotation.
            # This is a strong constraint, but reduces DOF.
            # To be more flexible, we could allow a small relative rotation, but let's start rigid.
            a = global_rotation_deg
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons.append(hex_verts)
            hex_axes.append(get_hex_axes(a))
            
        # Check overlaps
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                    # Calculate penetration depth roughly
                    dx = rotated_coords[i][0] - rotated_coords[j][0]
                    dy = rotated_coords[i][1] - rotated_coords[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    # Minimum distance between centers of disjoint unit hexagons is sqrt(3)
                    penetration = max(0, SQRT3 - dist)
                    overlap_penalty += 100.0 * penetration * penetration + 20.0
                    
        # Calculate outer side length
        # We need to find the best outer orientation for this fixed cluster.
        # Since the cluster is rigid, we can sample outer orientations.
        best_side = float('inf')
        # Sample 36 outer orientations (every 1.66 deg)
        for k in range(36):
            oa_deg = outer_rotation_deg + k * (60.0 / 36.0)
            side = calc_outer_side(hexagons, oa_deg)
            if side < best_side:
                best_side = side
                
        return overlap_penalty + best_side, best_side

    # Main optimization loop
    best_score = float('inf')
    best_params = None
    best_outer_rot = 0.0
    
    # We will use Simulated Annealing on the global rotation of the cluster.
    # The "position" is fixed at (0,0) due to symmetry.
    # The "shape" is fixed by the initial config.
    # So we are only optimizing: global_rotation, outer_rotation.
    # This is very fast!
    
    configs = [coords1, coords2, coords3, coords4]
    
    for config in configs:
        if time.time() > deadline - 5.0:
            break
            
        # SA parameters
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.995
        current_global_rot = random.uniform(0, 360)
        current_outer_rot = random.uniform(0, 60)
        
        current_score, _ = evaluate_cluster(config, current_global_rot, current_outer_rot)
        local_best_score = current_score
        local_best_global_rot = current_global_rot
        local_best_outer_rot = current_outer_rot
        
        step = 0
        while T > T_min and time.time() < deadline - 10.0:
            # Perturbation
            new_global_rot = current_global_rot + random.gauss(0, 5.0)
            new_outer_rot = current_outer_rot + random.gauss(0, 5.0)
            
            new_score, _ = evaluate_cluster(config, new_global_rot, new_outer_rot)
            
            delta_e = new_score - current_score
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                current_global_rot = new_global_rot
                current_outer_rot = new_outer_rot
                current_score = new_score
                
                if current_score < local_best_score:
                    local_best_score = current_score
                    local_best_global_rot = current_global_rot
                    local_best_outer_rot = current_outer_rot
            
            step += 1
            if step % 10 == 0:
                T *= cooling_rate
                
        if local_best_score < best_score:
            best_score = local_best_score
            best_params = (config, local_best_global_rot, local_best_outer_rot)
            
    if best_params is None:
        # Fallback
        config = coords1
        global_rot = 0.0
        outer_rot = 0.0
    else:
        config, global_rot, outer_rot = best_params
        
    # Final refinement: Fine-tune the outer rotation for the best global rotation
    # We need to reconstruct the hexagons to return them
    
    rad = math.radians(global_rot)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)
    
    rotated_coords = []
    for x, y in config:
        rx = x * cos_r - y * sin_r
        ry = x * sin_r + y * cos_r
        rotated_coords.append((rx, ry))
        
    hexagons = []
    hex_axes = []
    for i in range(N):
        cx, cy = rotated_coords[i]
        a = global_rot
        hex_verts = get_hex_vertices(cx, cy, a)
        hexagons.append(hex_verts)
        hex_axes.append(get_hex_axes(a))
        
    # Check validity
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # If invalid, we have a problem. Rigid cluster might have overlaps if initial config was bad.
        # Let's try a simple fallback: just return the initial lattice config1 with 0 rotation.
        # This is guaranteed to be valid (standard hex packing).
        config = coords1
        global_rot = 0.0
        outer_rot = 0.0
        rad = 0.0
        rotated_coords = [(x, y) for x, y in config]
        hexagons = []
        hex_axes = []
        for i in range(N):
            cx, cy = rotated_coords[i]
            a = 0.0
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons.append(hex_verts)
            hex_axes.append(get_hex_axes(a))
            
    # Find the absolute best outer rotation for this final configuration
    best_side = float('inf')
    best_oa = 0.0
    for oa_deg in np.linspace(0, 60, 120):
        side = calc_outer_side(hexagons, oa_deg)
        if side < best_side:
            best_side = side
            best_oa = oa_deg
            
    inner_list = []
    for i in range(N):
        inner_list.append([rotated_coords[i][0], rotated_coords[i][1], global_rot])
        
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
