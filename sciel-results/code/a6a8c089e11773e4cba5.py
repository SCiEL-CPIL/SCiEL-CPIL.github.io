# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _get_all_vertices(positions, angles, side=1.0):
    """positions: (N,2), angles: (N,) in degrees. Returns (N*6, 2)."""
    verts = []
    for i in range(len(positions)):
        verts.append(_hex_vertices(positions[i,0], positions[i,1], side, angles[i]))
    return np.vstack(verts)

def _min_outer_side(vertices):
    """
    Compute minimum side length s of a regular hexagon centered at origin (angle 0)
    that contains all given vertices.
    """
    outward_normals = np.array([
        [np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))]
        for k in range(6)
    ])
    dots = vertices @ outward_normals.T
    max_dot = np.max(dots, axis=1)
    apothem_min = np.max(max_dot)
    s_min = 2.0 * apothem_min / np.sqrt(3.0)
    return s_min

def _check_separation(pos1, ang1, pos2, ang2, side=1.0):
    """
    Check if two unit hexagons are separated using SAT.
    Returns True if separated (no overlap), False if overlapping.
    """
    v1 = _hex_vertices(pos1[0], pos1[1], side, ang1)
    v2 = _hex_vertices(pos2[0], pos2[1], side, ang2)
    
    normals = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i+1)%6] - v[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                normals.append(n)
    
    for n in normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 - 1e-8 or max2 < min1 - 1e-8:
            return True
    
    return False

def _is_valid_config(positions, angles, side=1.0):
    """Check if configuration is valid: all pairs separated and inside outer hex."""
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            if not _check_separation(positions[i], angles[i], positions[j], angles[j], side):
                return False
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 590.0  # 600s limit, leave margin
    
    n = 12
    side = 1.0
    
    # Define the topology: 
    # 0: Center
    # 1-6: Inner ring (hexagonal neighbors)
    # 7-11: Outer ring (attached to inner ring, skipping 1 to make 5)
    # This corresponds to the standard "flower" + 5 petals arrangement which is dense.
    
    # Initial guess parameters for the topology
    # r1: distance from center to inner ring
    # r2: distance from inner ring to outer ring
    # theta: angular offset for outer ring relative to inner
    
    # Ideal distance for touching hexagons is 2*side (center to center)
    # But they can be closer if rotated, but let's start with touching.
    
    def generate_config(params):
        """
        params: [r1, r2, theta_outer, angle_center, angle_inner, angle_outer]
        Returns: positions (12, 2), angles (12,)
        """
        r1, r2, theta_off, a_c, a_i, a_o = params
        
        positions = np.zeros((12, 2))
        angles = np.zeros(12)
        
        # Center
        positions[0] = [0, 0]
        angles[0] = a_c
        
        # Inner ring (6 hexagons)
        for k in range(6):
            theta = np.radians(60 * k)
            # Distance r1
            positions[k+1] = [r1 * np.cos(theta), r1 * np.sin(theta)]
            angles[k+1] = a_i
            
        # Outer ring (5 hexagons attached to inner ring indices 1,2,3,4,5 - skipping 6 which is 'top' maybe? 
        # Let's attach to 0,1,2,3,4 of the inner ring? No, standard is attached to the "shoulders".
        # Let's attach to inner indices 1, 2, 3, 4, 5.
        # The direction from center to inner k is 60k. The outer one is further out.
        for k in range(5):
            inner_idx = k + 1 # inner indices 1..5
            theta_inner = np.radians(60 * inner_idx)
            
            # The outer hex is attached to the side of the inner hex facing away from center.
            # The center of the outer hex is at distance r1 + r2 along the same radial line?
            # Actually, if they touch, the distance between centers is 2*side.
            # So outer pos = inner pos + (2*side) * unit_vector(radial)?
            # Wait, if we just move radially, the outer hexagon might overlap with adjacent inner ones if r2 is too small or angle is wrong.
            # Let's define outer position as:
            # pos_outer = pos_inner + r2 * (cos(theta_inner + theta_off), sin(theta_inner + theta_off))
            
            positions[k+7] = positions[inner_idx] + r2 * np.array([
                np.cos(theta_inner + np.radians(theta_off)),
                np.sin(theta_inner + np.radians(theta_off))
            ])
            angles[k+7] = a_o
            
        return positions, angles

    def objective(params):
        """
        Objective: Minimize outer hexagon side length + penalty for overlaps.
        """
        positions, angles = generate_config(params)
        
        # Check separation
        overlap_penalty = 0.0
        valid = True
        for i in range(12):
            for j in range(i+1, 12):
                # Fast check: distance
                d = np.linalg.norm(positions[i] - positions[j])
                if d < 2.0 * side - 0.1:
                    if not _check_separation(positions[i], angles[i], positions[j], angles[j], side):
                        # Calculate penetration depth roughly for penalty
                        # Just use a large penalty if overlapping
                        overlap_penalty += 100.0 * (2.0 * side - d) ** 2
                        valid = False
        
        vertices = _get_all_vertices(positions, angles, side)
        s = _min_outer_side(vertices)
        
        return s + overlap_penalty

    # Initial parameters
    # r1 ~ 2.0 (touching)
    # r2 ~ 2.0 (touching)
    # theta_off ~ 0 (radial)
    # Angles 0
    
    best_s = float('inf')
    best_positions = None
    best_angles = None
    
    # Run multiple restarts with different initial angles and radii
    # The topology is fixed, but the "twist" can vary.
    
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        # Randomize initial angles slightly to escape local minima
        # The symmetry suggests angles should be 0 or 30, but let's allow optimization.
        
        # Random initial r1, r2 in [1.8, 2.2]
        r1_init = random.uniform(1.8, 2.2)
        r2_init = random.uniform(1.8, 2.2)
        theta_init = random.uniform(-10, 10)
        a_c_init = random.uniform(0, 60)
        a_i_init = random.uniform(0, 60)
        a_o_init = random.uniform(0, 60)
        
        x0 = np.array([r1_init, r2_init, theta_init, a_c_init, a_i_init, a_o_init])
        
        # Bounds:
        # r1, r2 > 1.0 (must not overlap center too much)
        # angles 0-60
        bounds = [
            (1.0, 5.0),
            (1.0, 5.0),
            (-30, 30),
            (0, 60),
            (0, 60),
            (0, 60)
        ]
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
            
            if res.success or res.fun < best_s:
                # Validate the best found
                pos, ang = generate_config(res.x)
                
                # Final strict check
                if _is_valid_config(pos, ang, side):
                    verts = _get_all_vertices(pos, ang, side)
                    s_val = _min_outer_side(verts)
                    
                    if s_val < best_s:
                        best_s = s_val
                        best_positions = pos
                        best_angles = ang
                        
        except Exception:
            pass
            
    # If optimization failed or didn't find valid, fallback to a known good static config
    if best_positions is None:
        # Static config: Honeycomb
        # Center (0,0)
        # 6 around at distance 2
        # 5 more around those
        # This is a valid packing.
        positions = np.zeros((12, 2))
        angles = np.zeros(12)
        
        positions[0] = [0,0]
        
        for k in range(6):
            theta = np.radians(60 * k)
            positions[k+1] = [2.0 * np.cos(theta), 2.0 * np.sin(theta)]
            
        for k in range(5):
            inner_idx = k + 1
            theta = np.radians(60 * inner_idx)
            # Attach radially
            positions[k+7] = positions[inner_idx] + 2.0 * np.array([np.cos(theta), np.sin(theta)])
            
        best_positions = positions
        best_angles = angles
        verts = _get_all_vertices(positions, angles, side)
        best_s = _min_outer_side(verts)
        
    # Ensure centering: shift so centroid is at 0? 
    # The outer hex is centered at 0. The inner ones should be arranged around 0.
    # Our generator puts center at 0, so it's already centered.
    
    inner_hex_data = np.column_stack([best_positions, best_angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
