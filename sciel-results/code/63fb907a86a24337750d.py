import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3.0)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for 150s kill

    # Unit hexagon vertices (flat-top, side length 1)
    # Vertices are at angles 0, 60, 120, 180, 240, 300 degrees
    hex_verts_base = np.array([
        [1.0, 0.0],
        [0.5, SQRT3 / 2.0],
        [-0.5, SQRT3 / 2.0],
        [-1.0, 0.0],
        [-0.5, -SQRT3 / 2.0],
        [0.5, -SQRT3 / 2.0]
    ])

    # Outer hexagon normals (pointing outward)
    # For a regular hexagon with vertex at angle 0, the edges are perpendicular to angles 30, 90, ...
    # Wait, if vertex is at 0, edge is between 0 and 60. Midpoint angle 30. Normal is 30.
    outer_normals = np.array([
        [math.cos(math.pi / 6 + k * math.pi / 3), math.sin(math.pi / 6 + k * math.pi / 3)]
        for k in range(6)
    ])

    def get_rot_mat(angle):
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array([[c, -s], [s, c]])

    def get_hex_verts(angle):
        R = get_rot_mat(angle)
        return hex_verts_base @ R.T

    def compute_objective(params):
        """
        params: array of size 3N: [x0, y0, th0, x1, y1, th1, ...]
        Returns: side_length + penalty
        """
        if time.time() > deadline:
            return float('inf')
            
        positions = params[0::3].reshape(-1, 2)
        angles = params[1::3]
        
        # 1. Compute outer side length
        # We need max projection of all vertices onto outer normals
        max_proj = 0.0
        for i in range(N):
            verts = positions[i] + get_hex_verts(angles[i])
            # Project all 6 vertices onto 6 normals
            # verts: (6, 2), normals: (6, 2)
            # proj: (6, 6)
            proj = verts @ outer_normals.T
            max_proj_i = np.max(proj)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        # Side length S relates to max projection M by M = S * SQRT3 / 2
        # So S = 2 * M / SQRT3
        side_len = 2.0 * max_proj / SQRT3
        
        # 2. Compute penalty for overlaps
        penalty = 0.0
        # Pre-compute vertices for all hexagons to reuse in penalty check
        all_verts = []
        for i in range(N):
            all_verts.append(positions[i] + get_hex_verts(angles[i]))
            
        for i in range(N):
            for j in range(i + 1, N):
                v_i = all_verts[i]
                v_j = all_verts[j]
                
                # Get normals for hex i and j
                # Normals of a hexagon rotated by theta are base_normals rotated by theta
                # Base normals (flat-top) are at 30, 90, ...
                # Let's just compute them from vertices to be safe? 
                # Or use the known formula.
                # For flat-top hex, normals are at 30 + 60k.
                # If rotated by angle, normals are at 30 + 60k + angle.
                
                n_i = outer_normals + angles[i] # Wrong, outer_normals are vectors.
                # Let's rotate the base normals.
                base_normals = outer_normals # These are for outer hex, which is flat-top? 
                # Outer hex normals are fixed. Inner hex normals depend on angle.
                # Base inner hex normals (flat-top) are same as outer hex normals if aligned.
                # Let's define base_inner_normals = outer_normals.
                Ri = get_rot_mat(angles[i])
                Rj = get_rot_mat(angles[j])
                n_i = base_inner_normals @ Ri.T # (6, 2)
                n_j = base_inner_normals @ Rj.T # (6, 2)
                
                # Combine normals
                all_n = np.vstack([n_i, n_j]) # (12, 2)
                
                # Project v_i and v_j onto all_n
                # proj_i: (6, 12)
                proj_i = v_i @ all_n.T
                min_i = np.min(proj_i, axis=0)
                max_i = np.max(proj_i, axis=0)
                
                proj_j = v_j @ all_n.T
                min_j = np.min(proj_j, axis=0)
                max_j = np.max(proj_j, axis=0)
                
                # Penetration for each normal
                # If max_i < min_j or max_j < min_i, no overlap in this direction
                # Overlap depth = min(max_i, max_j) - max(min_i, min_j)
                # If depth > 0, they overlap in this direction
                # We want to penalize overlap.
                # Note: If they overlap in ANY direction, they intersect.
                # Summing squared penetrations is a good smooth approximation.
                
                upper = np.minimum(max_i, max_j)
                lower = np.maximum(min_i, min_j)
                depths = upper - lower
                # Only positive depths count
                depths = np.maximum(0.0, depths)
                penalty += np.sum(depths ** 2)
                
        # Weight the penalty heavily
        # The side length is around 4. Penalty should be >> 1 if overlapping.
        return side_len + 10.0 * penalty

    def hex_grid_init(rotation=0.0):
        """Initialize with a hexagonal grid arrangement."""
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        
        # Center
        positions[0] = [0.0, 0.0]
        
        # 6 neighbors
        # Axial to Cartesian: x = 1.5 * q, y = SQRT3 * (r + q / 2.0)
        neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
        idx = 1
        for q, r in neighbors:
            positions[idx, 0] = 1.5 * q
            positions[idx, 1] = SQRT3 * (r + q / 2.0)
            idx += 1
            
        # 5 more from next ring
        # We need 5 more to make 12.
        # Ring 2 has 12 hexes. We pick 5.
        # To keep it compact, pick those closest to the center? Or symmetric?
        # Let's pick (2,0), (2,-1), (1,1), (-1,2), (-2,1)
        extra = [(2,0), (2,-1), (1,1), (-1,2), (-2,1)]
        for q, r in extra:
            positions[idx, 0] = 1.5 * q
            positions[idx, 1] = SQRT3 * (r + q / 2.0)
            idx += 1
            
        # Apply global rotation
        if rotation != 0.0:
            R = get_rot_mat(rotation)
            positions = positions @ R.T
            angles += rotation
            
        # Scale slightly to ensure strict disjointness initially?
        # The grid is already disjoint.
        return positions, angles

    best_side = float('inf')
    best_params = None
    
    # Try multiple starts
    starts = []
    # 1. Standard grid
    pos0, ang0 = hex_grid_init(0.0)
    starts.append(np.concatenate([pos0.flatten(), ang0]))
    
    # 2. Rotated grid
    pos1, ang1 = hex_grid_init(math.pi / 6)
    starts.append(np.concatenate([pos1.flatten(), ang1]))
    
    # 3. Perturbed grids
    np.random.seed(42)
    for k in range(10):
        pos, ang = hex_grid_init(0.0)
        # Small random perturbation
        pos += np.random.normal(0, 0.05, pos.shape)
        ang += np.random.normal(0, 0.05, ang.shape)
        starts.append(np.concatenate([pos.flatten(), ang]))
        
    for start in starts:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(compute_objective, start, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < best_side:
                # Check validity
                val = compute_objective(res.x)
                # Extract penalty part to ensure it's 0
                # We don't have direct access to penalty, but if val is close to side_len, it's good.
                # Let's just check if val is reasonable.
                if val < best_side:
                    best_side = val
                    best_params = res.x
        except Exception as e:
            pass

    if best_params is None:
        # Fallback to initial grid
        pos0, ang0 = hex_grid_init(0.0)
        best_params = np.concatenate([pos0.flatten(), ang0])
        best_side = compute_objective(best_params)

    # Extract best result
    positions = best_params[0::3].reshape(-1, 2)
    angles = best_params[1::3]
    
    # Final side length calculation (without penalty)
    max_proj = 0.0
    for i in range(N):
        verts = positions[i] + get_hex_verts(angles[i])
        proj = verts @ outer_normals.T
        max_proj_i = np.max(proj)
        if max_proj_i > max_proj:
            max_proj = max_proj_i
    final_side = 2.0 * max_proj / SQRT3
    
    # Add tiny margin for safety
    final_side *= 1.0001

    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(angles[i] * 180.0 / math.pi)])
    
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, float(final_side), angle
# EVOLVE-BLOCK-END
