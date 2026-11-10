import numpy as np
import time
import itertools

def hexagon_packing_11():
    start_time = time.time()
    deadline = 550.0  # Safety margin before 600s limit

    def get_hex_vertices(cx, cy, s, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        angles = angle_rad + np.deg2rad(np.arange(6) * 60)
        return np.column_stack((cx + s * np.cos(angles), cy + s * np.sin(angles)))

    def check_separation(v1, v2):
        # v1, v2 are (6, 2) arrays of vertices
        # Check separation on all 12 edge normals
        # Edges of hexagon 1
        e1 = np.roll(v1, -1, axis=0) - v1
        n1 = np.column_stack((-e1[:, 1], e1[:, 0])) / np.linalg.norm(e1, axis=1, keepdims=True)
        
        # Edges of hexagon 2
        e2 = np.roll(v2, -1, axis=0) - v2
        n2 = np.column_stack((-e2[:, 1], e2[:, 0])) / np.linalg.norm(e2, axis=1, keepdims=True)
        
        normals = np.vstack([n1, n2])
        
        max_overlap = 0.0
        for n in normals:
            # Project vertices onto normal
            p1 = np.dot(v1, n)
            p2 = np.dot(v2, n)
            
            min1, max1 = np.min(p1), np.max(p1)
            min2, max2 = np.min(p2), np.max(p2)
            
            # Check if intervals [min1, max1] and [min2, max2] are separated
            # They are separated if max1 < min2 or max2 < min1
            # Overlap depth is max(0, min(max1, max2) - max(min1, min2))
            # But we need to check if they are apart by more than 1e-6.
            # So valid if max1 + 1e-6 <= min2 OR max2 + 1e-6 <= min1
            # Penalty is the amount of violation
            sep1 = min2 - max1
            sep2 = min1 - max2
            
            if sep1 > 1e-6 or sep2 > 1e-6:
                continue # Separated on this normal
                
            # Not separated on this normal. Calculate overlap depth.
            # The "distance" between the intervals is negative if they overlap.
            # dist = min(max1, max2) - max(min1, min2)
            # If dist > 0, they overlap.
            # We want to penalize this overlap.
            dist = min(max1, max2) - max(min1, min2)
            if dist > max_overlap:
                max_overlap = dist
                
        return max_overlap

    def objective(params):
        # params: x1, y1, a1, ..., x11, y11, a11, a_out, S
        # 33 + 2 = 35 params
        inner_params = params[:33].reshape(11, 3)
        a_out = params[33]
        S = params[34]
        
        if S < np.sqrt(11):
            S = np.sqrt(11)
            
        penalty = 0.0
        
        # Get outer hexagon vertices
        outer_v = get_hex_vertices(0, 0, S, a_out)
        
        # Get inner hexagon vertices
        inner_vs = []
        for i in range(11):
            v = get_hex_vertices(inner_params[i, 0], inner_params[i, 1], 1.0, inner_params[i, 2])
            inner_vs.append(v)
            
        # Containment check
        # For each inner vertex, check if it's inside outer hexagon
        # Outer hexagon edges
        outer_e = np.roll(outer_v, -1, axis=0) - outer_v
        outer_n = np.column_stack((-outer_e[:, 1], outer_e[:, 0])) / np.linalg.norm(outer_e, axis=1, keepdims=True)
        # Inward normals
        # Center is (0,0). Vector from edge to center should have positive dot product with inward normal.
        # Edge midpoint is (outer_v[i] + outer_v[i+1])/2. Vector from midpoint to center is -midpoint.
        midpoints = (outer_v + np.roll(outer_v, -1, axis=0)) / 2.0
        vec_to_center = -midpoints
        # Check dot product
        dots = np.sum(vec_to_center * outer_n, axis=1)
        # Inward normals should have positive dot product with vector to center
        # If negative, flip
        for i in range(6):
            if dots[i] < 0:
                outer_n[i] *= -1
                
        # Check containment
        for i in range(11):
            v = inner_vs[i]
            # For each edge, check dot product of (vertex - edge_point) with inward normal
            # If dot > 0, it's outside
            for j in range(6):
                edge_point = outer_v[j]
                vec = v - edge_point
                dists = np.sum(vec * outer_n[j], axis=1)
                # dists > 0 means outside
                if np.any(dists > 0):
                    penalty += np.sum(np.maximum(dists, 0)) ** 2 * 100 # Strong penalty

        # Separation check
        for i in range(11):
            for j in range(i+1, 11):
                overlap = check_separation(inner_vs[i], inner_vs[j])
                if overlap > 0:
                    penalty += overlap ** 2 * 1000 # Very strong penalty

        return S + penalty

    best_score = np.inf
    best_params = None
    
    # Initial guesses
    # A simple grid layout
    base_positions = [
        [0, 0],
        [-2.5, 0], [2.5, 0],
        [-1.25, 2.17], [1.25, 2.17],
        [-1.25, -2.17], [1.25, -2.17],
        [-3.75, 2.17], [3.75, 2.17],
        [-3.75, -2.17], [3.75, -2.17]
    ]
    
    initial_params = []
    for pos in base_positions:
        initial_params.extend([pos[0], pos[1], 0])
    initial_params.extend([0, 8.0]) # a_out, S
    
    initial_params = np.array(initial_params)
    
    # Add some noise to initial params for restarts
    np.random.seed(42)
    restarts = 5
    
    for r in range(restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            x0 = initial_params
        else:
            x0 = initial_params + np.random.normal(0, 0.5, size=initial_params.shape)
            x0[34] = 6.0 + np.random.rand() # Perturb S
            
        try:
            res = scipy_optimize_minimize(objective, x0, method='Nelder-Mead', 
                                          options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception as e:
            pass

    if best_params is None:
        # Fallback to seed solution
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    # Extract results
    inner_params = best_params[:33].reshape(11, 3)
    a_out = best_params[33]
    S = best_params[34]
    
    inner_hex_data = inner_params
    outer_hex_data = np.array([0, 0, a_out])
    outer_hex_side_length = S

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# Import scipy.optimize as scipy_optimize_minimize to avoid name conflicts or long names
from scipy.optimize import minimize as scipy_optimize_minimize

if __name__ == "__main__":
    # Quick test
    inner, outer, side = hexagon_packing_11()
    print(f"Side length: {side}")
    print(f"Inner: {inner.shape}")
    print(f"Outer: {outer}")
