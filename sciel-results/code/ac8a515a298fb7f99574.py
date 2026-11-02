import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = 145.0

# Unit hexagon geometry
# Vertices at angles 0, 60, 120, 180, 240, 300
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])

# Outward normals for the 6 edges (pointing to edge midpoints)
# Midpoints at 30, 90, 150, 210, 270, 330
OUTER_NORMALS = np.array([
    [np.cos(np.radians(60 * i + 30)), np.sin(np.radians(60 * i + 30))] for i in range(6)
])

def get_hex_vertices(centers):
    """
    centers: (N, 3) array of [x, y, angle_deg]
    Returns: (N, 6, 2) array of vertices
    """
    n = len(centers)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        x, y, a_deg = centers[i]
        a_rad = np.radians(a_deg)
        c = np.cos(a_rad)
        s = np.sin(a_rad)
        R = np.array([[c, -s], [s, c]])
        verts[i] = UNIT_VERTS @ R.T + np.array([x, y])
    return verts

def compute_required_side(centers, outer_center=(0.0, 0.0), outer_angle=0.0):
    """
    Compute the minimum outer side length needed to contain all inner hexagons.
    """
    verts = get_hex_vertices(np.array(centers, dtype=float))
    
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    max_extent = 0.0
    for i in range(len(centers)):
        v = verts[i]
        v_rel = v - np.array(outer_center)
        v_rot = v_rel @ R_o.T
        # Project onto outward normals
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        max_extent = max(max_extent, np.max(max_proj))
    
    # Apothem = side * sqrt(3)/2  =>  side = apothem / (sqrt(3)/2)
    return max_extent / (np.sqrt(3) / 2)

def check_validity(centers, outer_center=(0.0, 0.0), outer_side=None, outer_angle=0.0, tol=1e-7):
    """
    Check if packing is valid: no overlaps, all inside outer hexagon.
    """
    if outer_side is None:
        outer_side = compute_required_side(centers, outer_center, outer_angle)
        
    verts = get_hex_vertices(np.array(centers, dtype=float))
    n = verts.shape[0]
    
    # 1. Check pairwise separation (SAT)
    for i in range(n):
        v1 = verts[i]
        for j in range(i + 1, n):
            v2 = verts[j]
            edges1 = np.roll(v1, -1, axis=0) - v1
            edges2 = np.roll(v2, -1, axis=0) - v2
            axes = np.vstack([edges1, edges2])
            axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-15)
            
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                            proj2.min(axis=0) - proj1.max(axis=0))
            
            if np.max(sep) < -tol:
                return False
    
    # 2. Check containment
    apothem = outer_side * np.sqrt(3) / 2
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    for i in range(n):
        v = verts[i]
        v_rel = v - np.array(outer_center)
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        
        if np.max(max_proj) > apothem + tol:
            return False
            
    return True

def generate_symmetric_configs():
    """
    Generate configurations with 6-fold rotational symmetry.
    The 12 hexagons are arranged in a ring of 6 pairs, or a ring of 12.
    We model one hexagon (or pair) and rotate it 6 times.
    """
    configs = []
    
    # Symmetry 1: Ring of 6 pairs (12 hexagons)
    # Each "unit" consists of 2 hexagons.
    # We optimize the position/angle of the first hexagon in the unit.
    # The second hexagon is placed relative to the first to minimize local extent.
    
    # Let's define a function to expand a symmetric seed to full 12
    def expand_symmetric_6pairs(x, y, angle_deg, offset_x, offset_y, offset_angle):
        centers = []
        base_angle_rad = np.radians(angle_deg)
        for k in range(6):
            rot_angle = 60 * k
            R = np.array([
                [np.cos(np.radians(rot_angle)), -np.sin(np.radians(rot_angle))],
                [np.sin(np.radians(rot_angle)), np.cos(np.radians(rot_angle))]
            ])
            
            # Hex 1
            p1 = np.array([x, y])
            p1_rot = R @ p1
            a1 = angle_deg + rot_angle
            centers.append([p1_rot[0], p1_rot[1], a1])
            
            # Hex 2 (offset relative to Hex 1 in local frame)
            p2_local = np.array([x + offset_x, y + offset_y])
            p2_rot = R @ p2_local
            a2 = angle_deg + offset_angle + rot_angle
            centers.append([p2_rot[0], p2_rot[1], a2])
            
        return np.array(centers)
    
    # Symmetry 2: Ring of 12 (each hexagon rotated by 30 degrees)
    def expand_symmetric_12(x, y, angle_deg):
        centers = []
        for k in range(12):
            rot_angle = 30 * k
            R = np.array([
                [np.cos(np.radians(rot_angle)), -np.sin(np.radians(rot_angle))],
                [np.sin(np.radians(rot_angle)), np.cos(np.radians(rot_angle))]
            ])
            p = np.array([x, y])
            p_rot = R @ p
            a = angle_deg + rot_angle
            centers.append([p_rot[0], p_rot[1], a])
        return np.array(centers)
    
    # Initial guesses for 6-pair symmetry
    # Hexagons are side 1. Width is sqrt(3).
    # Place them around a circle.
    # Try a few radii and angles
    
    radii = [1.5, 2.0, 2.5, 3.0]
    angles = [0, 30, 60, 90, 120]
    
    for r in radii:
        for a in angles:
            # Hexagon center at (r, 0)
            # Offset for second hexagon: try placing it "outward" or "inward" or "tangential"
            # Let's try a few relative offsets
            offsets = [
                (0.5, 0.5, 0), # Diagonal out
                (-0.5, 0.5, 0), # Diagonal in
                (0.8, 0.0, 0), # Radial out
                (0.0, 0.8, 0), # Tangential
                (0.5, 0.0, 30), # Radial out, rotated
                (-0.5, 0.0, -30), # Radial in, rotated
            ]
            for ox, oy, oa in offsets:
                # The base hexagon is at (r, 0) with angle a
                # We want the pair to be compact.
                # Let's just generate the config and let the optimizer fix it.
                cfg = expand_symmetric_6pairs(r, 0.0, a, ox, oy, oa)
                # Center of mass correction? The symmetry ensures COM is at 0.
                configs.append(cfg)
                
    # Initial guesses for 12-ring symmetry
    for r in radii:
        for a in angles:
            cfg = expand_symmetric_12(r, 0.0, a)
            configs.append(cfg)
            
    return configs

def optimize_symmetric_ring(params, mode='6pairs'):
    """
    params: 
      if mode='6pairs': [x, y, angle, ox, oy, oangle]
      if mode='12': [x, y, angle]
    Returns: (centers, cost)
    """
    if mode == '6pairs':
        x, y, angle, ox, oy, oangle = params
        centers = []
        for k in range(6):
            rot_angle = 60 * k
            R = np.array([
                [np.cos(np.radians(rot_angle)), -np.sin(np.radians(rot_angle))],
                [np.sin(np.radians(rot_angle)), np.cos(np.radians(rot_angle))]
            ])
            
            p1 = np.array([x, y])
            p1_rot = R @ p1
            a1 = angle + rot_angle
            centers.append([p1_rot[0], p1_rot[1], a1])
            
            p2_local = np.array([x + ox, y + oy])
            p2_rot = R @ p2_local
            a2 = angle + oangle + rot_angle
            centers.append([p2_rot[0], p2_rot[1], a2])
            
        centers = np.array(centers)
    else:
        x, y, angle = params
        centers = []
        for k in range(12):
            rot_angle = 30 * k
            R = np.array([
                [np.cos(np.radians(rot_angle)), -np.sin(np.radians(rot_angle))],
                [np.sin(np.radians(rot_angle)), np.cos(np.radians(rot_angle))]
            ])
            p = np.array([x, y])
            p_rot = R @ p
            a = angle + rot_angle
            centers.append([p_rot[0], p_rot[1], a])
        centers = np.array(centers)
        
    # Cost function:
    # 1. Overlap penalty (SAT)
    # 2. Extent (side length)
    
    verts = get_hex_vertices(centers)
    
    # Overlap penalty
    overlap_cost = 0.0
    for i in range(N):
        v1 = verts[i]
        for j in range(i + 1, N):
            v2 = verts[j]
            edges1 = np.roll(v1, -1, axis=0) - v1
            edges2 = np.roll(v2, -1, axis=0) - v2
            axes = np.vstack([edges1, edges2])
            axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-15)
            
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                            proj2.min(axis=0) - proj1.max(axis=0))
            
            max_sep = np.max(sep)
            if max_sep < 0:
                # Quadratic penalty for overlap
                overlap_cost += 1000.0 * max_sep ** 2
    
    # Extent cost
    side = compute_required_side(centers)
    extent_cost = side ** 2
    
    total_cost = overlap_cost + extent_cost
    return centers, total_cost

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    configs = generate_symmetric_configs()
    
    # Shuffle to ensure variety
    np.random.seed(42)
    np.random.shuffle(configs)
    
    # We will try to optimize each config using L-BFGS-B
    # Since we have symmetry, the dimensionality is low (6 or 3).
    # We can afford to run many optimizations.
    
    # Split configs into two groups: 6-pair and 12-ring
    # The generator returns a list of (12, 3) arrays.
    # We need to know which symmetry generated them to know the parameter space.
    # Let's re-generate them in a structured way for optimization.
    
    # Optimization 1: 6-pair symmetry
    # Params: [x, y, angle, ox, oy, oangle]
    # Bounds:
    # x, y: [-5, 5]
    # angle: [0, 60]
    # ox, oy: [-2, 2]
    # oangle: [-60, 60]
    
    bounds_6 = [
        (-5.0, 5.0), # x
        (-5.0, 5.0), # y
        (0.0, 60.0), # angle
        (-2.0, 2.0), # ox
        (-2.0, 2.0), # oy
        (-60.0, 60.0) # oangle
    ]
    
    # Optimization 2: 12-ring symmetry
    # Params: [x, y, angle]
    bounds_12 = [
        (-5.0, 5.0),
        (-5.0, 5.0),
        (0.0, 30.0)
    ]
    
    # Initial guesses for 6-pair
    # Based on the generated configs, we can extract the first hexagon's params
    # But it's easier to just define a few good starting points for the optimizer.
    
    start_points_6 = [
        [2.0, 0.0, 0.0, 0.5, 0.5, 0.0],
        [2.0, 0.0, 30.0, -0.5, 0.5, 0.0],
        [2.5, 0.0, 0.0, 0.8, 0.0, 0.0],
        [1.5, 0.0, 0.0, 0.0, 0.8, 0.0],
        [2.0, 0.0, 0.0, 0.5, 0.0, 30.0],
        [2.0, 0.0, 0.0, -0.5, 0.0, -30.0],
        [2.2, 0.5, 10.0, 0.3, 0.3, 10.0],
        [1.8, 0.5, 20.0, 0.2, 0.4, -10.0]
    ]
    
    start_points_12 = [
        [2.0, 0.0, 0.0],
        [2.5, 0.0, 30.0],
        [1.8, 0.0, 15.0],
        [2.2, 0.0, 45.0]
    ]
    
    # Run optimizations
    # Time budget: 145s
    # Each optimization takes ~0.1-0.5s
    # We can run ~100 optimizations
    
    def run_optimization(start_point, bounds, mode, time_limit=2.0):
        def objective(params):
            centers, cost = optimize_symmetric_ring(params, mode)
            return cost
        
        start = time.time()
        try:
            res = minimize(objective, start_point, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5})
            if time.time() - start > time_limit:
                return None
            centers, cost = optimize_symmetric_ring(res.x, mode)
            return centers, res.x
        except Exception:
            return None
            
    # Process 6-pair starts
    for sp in start_points_6:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
            
        result = run_optimization(sp, bounds_6, '6pairs')
        if result:
            centers, _ = result
            side = compute_required_side(centers)
            if check_validity(centers, outer_side=side):
                if side < best_score:
                    best_score = side
                    best_result = (centers.tolist(), (0.0, 0.0), side, 0.0)
                    
    # Process 12-ring starts
    for sp in start_points_12:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
            
        result = run_optimization(sp, bounds_12, '12')
        if result:
            centers, _ = result
            side = compute_required_side(centers)
            if check_validity(centers, outer_side=side):
                if side < best_score:
                    best_score = side
                    best_result = (centers.tolist(), (0.0, 0.0), side, 0.0)
                    
    # If we have time, try a global search on the symmetric parameters
    if time.time() - start_time < DEADLINE - 20:
        # Use Differential Evolution for the 6-pair case (6 dims)
        def obj_de(params):
            centers, cost = optimize_symmetric_ring(params, '6pairs')
            return cost
            
        try:
            res_de = differential_evolution(obj_de, bounds_6, seed=42, maxiter=50, tol=1e-4, polish=True)
            centers, _ = optimize_symmetric_ring(res_de.x, '6pairs')
            side = compute_required_side(centers)
            if check_validity(centers, outer_side=side):
                if side < best_score:
                    best_score = side
                    best_result = (centers.tolist(), (0.0, 0.0), side, 0.0)
        except Exception:
            pass
            
    # Fallback if no valid result found
    if best_result is None:
        # Use a known valid configuration from the lattice approach
        # Simple 4x3 grid
        a1 = np.array([np.sqrt(3), 0.0])
        a2 = np.array([np.sqrt(3) / 2.0, 1.5])
        centers = []
        for i in range(4):
            for j in range(3):
                pos = i * a1 + j * a2
                # Center the grid
                pos -= np.array([1.5 * np.sqrt(3), 1.5])
                centers.append([pos[0], pos[1], 0.0])
        centers = np.array(centers)
        side = compute_required_side(centers)
        if check_validity(centers, outer_side=side):
            best_result = (centers.tolist(), (0.0, 0.0), side, 0.0)
        else:
            # Absolute fallback
            inner_default = []
            for i in range(N):
                theta = 2 * np.pi * i / N
                x = 3.5 * np.cos(theta)
                y = 3.5 * np.sin(theta)
                inner_default.append([x, y, 0.0])
            required_side = compute_required_side(inner_default)
            best_result = (inner_default, (0.0, 0.0), required_side, 0.0)
            
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Best Side: {result[2]}")
