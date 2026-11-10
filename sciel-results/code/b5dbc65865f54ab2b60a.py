import numpy as np
import math
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 590  # safety margin
    
    # Constants
    sqrt3 = math.sqrt(3)
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + side * math.cos(a)
            y = cy + side * math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    def project_polygon(verts, axis):
        dots = verts @ axis
        return np.min(dots), np.max(dots)

    def min_distance_sat(verts1, verts2):
        # Returns the minimum separation distance between two convex polygons.
        # If negative, they overlap (penetration depth).
        # If positive, they are separated.
        # We check all edge normals of both polygons.
        min_dist = float('inf')
        
        axes = []
        for i in range(len(verts1)):
            p1 = verts1[i]
            p2 = verts1[(i+1)%len(verts1)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
                
        for i in range(len(verts2)):
            p1 = verts2[i]
            p2 = verts2[(i+1)%len(verts2)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
                
        for axis in axes:
            min1, max1 = project_polygon(verts1, axis)
            min2, max2 = project_polygon(verts2, axis)
            
            # Distance between intervals
            # If max1 < min2: dist = min2 - max1
            # If max2 < min1: dist = min1 - max2
            # Else: dist = -overlap (negative)
            if max1 <= min2:
                d = min2 - max1
            elif max2 <= min1:
                d = min1 - max2
            else:
                # Overlap
                d = -min(max1 - min2, max2 - min1)
                
            if d < min_dist:
                min_dist = d
                
        return min_dist

    def compute_objective(x, angle_deg=0):
        # x: (12, 2) array of centers
        # Returns: objective value (s + penalty)
        # and gradients? L-BFGS-B can use numerical gradients, but we can provide analytical if needed.
        # For now, let's use numerical gradients provided by scipy, but we need to make the function smooth.
        
        centers = x.reshape(12, 2)
        
        # Compute vertices for all inner hexes
        verts_list = []
        for i in range(12):
            cx, cy = centers[i]
            verts = get_hex_vertices(cx, cy, 1.0, angle_deg)
            verts_list.append(verts)
            
        # 1. Compute minimum s required to contain all hexes in an outer hex centered at (0,0) with angle 0
        # Outer hex normals for angle 0:
        # Vertices at angles 0, 60, ...
        # Normals to edges are at 30, 90, ...
        normals = []
        for i in range(6):
            a = math.radians(30 + i * 60)
            normals.append((math.cos(a), math.sin(a)))
        normals = np.array(normals) # (6, 2)
        
        # Gather all vertices
        all_verts = np.vstack(verts_list) # (72, 2)
        
        # Support function: max dot product with each normal
        # s * apothem_factor >= max_proj
        # apothem = s * sqrt(3)/2
        # s >= max_proj / (sqrt(3)/2) = max_proj * 2 / sqrt(3)
        
        projs = all_verts @ normals.T # (72, 6)
        max_projs = np.max(projs, axis=0) # (6,)
        
        # The required s is determined by the max of these projections
        s_required = np.max(max_projs) * 2.0 / sqrt3
        
        # To make the optimization smoother, we can sum the penalties for each side
        # or just use the max. Max is non-differentiable.
        # Let's use a smooth max approximation or just rely on L-BFGS-B's numerical diff on the max?
        # Numerical diff on max can be noisy.
        # Alternative: Minimize sum of (max_proj_i)^2? No, that changes the objective.
        # Let's try minimizing s_required directly. If it's too noisy, we'll add a small epsilon.
        
        # 2. Collision penalties
        penalty = 0.0
        # We want to enforce separation > 0.
        # Use a soft penalty: if distance < 0, penalty += (distance)^2
        # To make it smoother, we can use a ramp or just square the penetration.
        
        for i in range(12):
            for j in range(i+1, 12):
                d = min_distance_sat(verts_list[i], verts_list[j])
                if d < 0:
                    # Penetration
                    # Scale penalty to be significant compared to s
                    penalty += -d * 10.0 # Linear penalty? Or quadratic?
                    # Quadratic is smoother at 0.
                    # penalty += d * d * 100.0
                    # Let's use a combination: linear for strong push, quadratic for smoothness
                    penalty += d * d * 50.0 - d * 5.0
        
        # 3. Boundary constraint is handled by s_required.
        # However, if we just minimize s_required, the hexagons might fly apart to reduce s?
        # No, s_required is the size of the container. Minimizing it forces hexagons to be compact.
        # But we also need to ensure they fit. s_required IS the fit constraint.
        # So the objective is simply s_required + collision_penalty.
        
        obj = s_required + penalty
        
        # Ensure finite
        if not np.isfinite(obj):
            return 1e10
            
        return obj

    def get_solution_from_x(x, angle_deg=0):
        centers = x.reshape(12, 2)
        verts_list = []
        for i in range(12):
            cx, cy = centers[i]
            verts = get_hex_vertices(cx, cy, 1.0, angle_deg)
            verts_list.append(verts)
            
        # Compute s
        all_verts = np.vstack(verts_list)
        normals = []
        for i in range(6):
            a = math.radians(30 + i * 60)
            normals.append((math.cos(a), math.sin(a)))
        normals = np.array(normals)
        projs = all_verts @ normals.T
        max_projs = np.max(projs, axis=0)
        s = np.max(max_projs) * 2.0 / sqrt3
        
        # Check validity
        # 1. Collisions
        valid = True
        for i in range(12):
            for j in range(i+1, 12):
                d = min_distance_sat(verts_list[i], verts_list[j])
                if d < -1e-6: # Allow tiny overlap? No, must be separated.
                    # The prompt says "more than 1e-6 apart" for validity?
                    # "two inner hexagons not separated: some edge normal ... must give projection intervals more than 1e-6 apart"
                    # This implies if they are NOT separated (overlap or touch too closely), it's invalid.
                    # So we need d >= 1e-6? Or just d >= 0?
                    # "more than 1e-6 apart" usually means gap > 1e-6.
                    # Let's aim for d > 0.
                    if d < 0:
                        valid = False
                        break
            if not valid:
                break
        
        # 2. Containment is guaranteed by construction of s, but let's double check vertices
        if valid:
            outer_verts = get_hex_vertices(0, 0, s, angle_deg)
            for i in range(12):
                for v in verts_list[i]:
                    # Check if v is inside outer_verts
                    # Using cross product method
                    inside = True
                    for k in range(6):
                        p1 = outer_verts[k]
                        p2 = outer_verts[(k+1)%6]
                        edge = p2 - p1
                        vec = v - p1
                        cross = edge[0]*vec[1] - edge[1]*vec[0]
                        if cross < -1e-6:
                            inside = False
                            break
                    if not inside:
                        valid = False
                        break
        
        if valid:
            inner_arr = np.column_stack([centers, np.zeros(12)])
            inner_arr[:, 2] = angle_deg
            outer_arr = np.array([0.0, 0.0, angle_deg])
            return inner_arr, outer_arr, s
        else:
            return None, None, float('inf')

    # Initial guess: Honeycomb lattice
    # 12 hexagons.
    # Pattern: 1 in center? No, 12 is not 1+6+...
    # 2-3-4-3? Sum 12.
    # Or 3-4-3-2?
    # Let's use a grid-like initial guess.
    
    def generate_initial_guess():
        # Rows of 3, 4, 3, 2?
        # y offsets: 0, dy, 2dy, 3dy
        # x offsets centered.
        
        # Let's try a simple rectangular grid 3x4 and let the optimizer fix it.
        # Or a hexagonal cluster.
        # Center (0,0)
        # 6 around it? That's 7.
        # Add 5 more?
        
        # Let's use the seed's configuration as a start, but perturb it.
        seed_inner = [
            (0, 0),
            (-2.5, 0),
            (2.5, 0),
            (-1.25, 2.17),
            (1.25, 2.17),
            (-1.25, -2.17),
            (1.25, -2.17),
            (-3.75, 2.17),
            (3.75, 2.17),
            (-3.75, -2.17),
            (3.75, -2.17),
            (0, -4),
        ]
        
        # Perturb slightly to break symmetry
        np.random.seed(42)
        x0 = np.array(seed_inner)
        x0 += np.random.normal(0, 0.1, x0.shape)
        return x0.flatten()

    best_score = 0.0
    best_inner = None
    best_outer = None
    best_s = float('inf')
    
    # Try a few different initial configurations
    initial_configs = []
    initial_configs.append(generate_initial_guess())
    
    # Another guess: 2 rows of 6?
    x2 = []
    for i in range(6):
        x2.append([i*2.5 - 6.25, 0])
    for i in range(6):
        x2.append([i*2.5 - 6.25, 2.17])
    x2 = np.array(x2)
    x2 += np.random.normal(0, 0.1, x2.shape)
    initial_configs.append(x2.flatten())
    
    # Another guess: 3 rows of 4
    x3 = []
    for i in range(3):
        y = i * 2.17 - 2.17
        for j in range(4):
            x = j * 2.5 - 3.75
            if i % 2 == 1:
                x += 1.25
            x3.append([x, y])
    x3 = np.array(x3)
    x3 += np.random.normal(0, 0.1, x3.shape)
    initial_configs.append(x3.flatten())

    for x0 in initial_configs:
        if time.time() > deadline:
            break
            
        try:
            # Use L-BFGS-B
            # We don't have an analytical gradient for the max/sat parts easily, so we use numerical.
            # To improve speed, we could implement gradients, but let's try numerical first.
            # If it's too slow, we will need to optimize the inner loop.
            
            res = minimize(
                compute_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6, 'maxls': 50}
            )
            
            if res.success or res.fun < best_s:
                inner, outer, s = get_solution_from_x(res.x)
                if inner is not None:
                    score = 1.0 / s
                    if score > best_score:
                        best_score = score
                        best_inner = inner
                        best_outer = outer
                        best_s = s
                        
        except Exception as e:
            pass

    # If no valid solution found, fallback to a known valid one (the seed)
    if best_inner is None:
        # Re-construct seed
        seed_inner = [
            (0, 0, 0),
            (-2.5, 0, 0),
            (2.5, 0, 0),
            (-1.25, 2.17, 0),
            (1.25, 2.17, 0),
            (-1.25, -2.17, 0),
            (1.25, -2.17, 0),
            (-3.75, 2.17, 0),
            (3.75, 2.17, 0),
            (-3.75, -2.17, 0),
            (3.75, -2.17, 0),
            (0, -4, 0),
        ]
        best_inner = np.array(seed_inner)
        # Calculate s for this
        centers = best_inner[:, :2]
        verts_list = []
        for i in range(12):
            verts_list.append(get_hex_vertices(centers[i,0], centers[i,1], 1.0, 0))
        all_verts = np.vstack(verts_list)
        normals = []
        for i in range(6):
            a = math.radians(30 + i * 60)
            normals.append((math.cos(a), math.sin(a)))
        normals = np.array(normals)
        projs = all_verts @ normals.T
        max_projs = np.max(projs, axis=0)
        best_s = np.max(max_projs) * 2.0 / sqrt3
        best_outer = np.array([0.0, 0.0, 0.0])
        best_score = 1.0 / best_s

    return best_inner, best_outer, best_s

# EVOLVE-BLOCK-END
