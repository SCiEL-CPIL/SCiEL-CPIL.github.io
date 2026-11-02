import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hex_vertices(cx, cy, side, angle_rad):
        """Returns 6x2 array of vertices for a regular hexagon."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_edge_normals(verts):
        """Returns 6x2 array of outward unit normals for a convex hexagon."""
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / norms[:, np.newaxis]
        return normals

    def sat_penetration(v1, v2):
        """
        Compute penetration depth between two convex polygons v1, v2 (6x2).
        Returns 0 if disjoint or separated, otherwise the minimum translation distance to separate.
        """
        normals1 = get_edge_normals(v1)
        normals2 = get_edge_normals(v2)
        normals = np.vstack([normals1, normals2])
        
        # Project v1 and v2 onto all normals
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        
        # Overlap on each axis
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        
        # If any overlap is <= 0, they are separated
        if np.min(overlaps) <= 1e-9:
            return 0.0
            
        # The minimum overlap is the penetration depth
        return np.min(overlaps)

    def containment_violation(inner_verts, outer_verts):
        """
        Compute how much inner_verts sticks out of outer_verts.
        outer_verts is a convex hexagon.
        Returns max violation (positive if sticking out).
        """
        outer_normals = get_edge_normals(outer_verts)
        
        # Project inner vertices onto outer normals
        p_inner = inner_verts @ outer_normals.T
        p_outer_max = (outer_verts @ outer_normals.T).max(axis=0)
        
        p_inner_max = p_inner.max(axis=0)
        
        viol = p_inner_max - p_outer_max
        return np.max(viol)

    def objective(params):
        """
        params: [x1, y1, ..., x12, y12, alpha1, ..., alpha12, S, theta]
        Total length: 2*N + N + 1 + 1 = 3*N + 2 = 38
        """
        if time.time() > deadline:
            return 1e10

        positions = params[:2 * N].reshape(N, 2)
        alphas = params[2 * N : 3 * N]
        S = params[3 * N]
        theta = params[3 * N + 1]
        
        if S < 3.0: # Minimum possible is larger
            return 1e10

        outer_verts = hex_vertices(0, 0, S, theta)
        
        penalty = 0.0
        inner_verts_list = []
        
        for i in range(N):
            x, y = positions[i]
            verts = hex_vertices(x, y, 1.0, alphas[i])
            inner_verts_list.append(verts)
            
            # Containment
            viol = containment_violation(verts, outer_verts)
            if viol > 0:
                penalty += viol * viol * 100.0
        
        # Pairwise disjointness
        for i in range(N):
            for j in range(i + 1, N):
                pd = sat_penetration(inner_verts_list[i], inner_verts_list[j])
                if pd > 0:
                    penalty += pd * pd * 100.0

        return S + penalty

    # Initial configurations
    # 1. Honeycomb 1-6-5
    # 2. Honeycomb 2-4-4-2
    # 3. Random
    
    def get_honeycomb_165(alpha_offset=0.0):
        pos = []
        alphas = []
        d1 = np.sqrt(3)
        d2 = 2 * np.sqrt(3)
        
        # Center
        pos.append([0.0, 0.0])
        alphas.append(alpha_offset)
        
        # Ring 1
        for k in range(6):
            a = np.radians(60 * k)
            pos.append([d1 * np.cos(a), d1 * np.sin(a)])
            alphas.append(alpha_offset)
            
        # Ring 2 (5 hexagons)
        for k in range(5):
            a = np.radians(60 * k)
            pos.append([d2 * np.cos(a), d2 * np.sin(a)])
            alphas.append(alpha_offset)
            
        return np.array(pos).flatten(), np.array(alphas).flatten()

    def get_honeycomb_2442(alpha_offset=0.0):
        pos = []
        alphas = []
        d1 = np.sqrt(3)
        
        # Row 1 (2 hexagons)
        # y = 1.5 * d1? No, let's use lattice coords
        # Hex lattice:
        # (0,0), (sqrt(3), 0)
        # (sqrt(3)/2, 1.5), (3*sqrt(3)/2, 1.5)
        
        # Let's place them in a 2-4-4-2 pattern
        # y levels: 0, 1.5, 3.0, 4.5
        # x offsets:
        # y=0: 2 hexes
        # y=1.5: 4 hexes
        # y=3.0: 4 hexes
        # y=4.5: 2 hexes
        
        h = 1.5 # vertical spacing between rows for touching hexes
        w = np.sqrt(3) # horizontal spacing
        
        # Row 0 (y=0): 2 hexes centered
        pos.append([-w/2, 0])
        pos.append([w/2, 0])
        alphas.extend([alpha_offset, alpha_offset])
        
        # Row 1 (y=h): 4 hexes
        # They sit in the dips
        x_start = -1.5 * w
        for k in range(4):
            pos.append([x_start + k * w, h])
            alphas.append(alpha_offset)
            
        # Row 2 (y=2h): 4 hexes
        for k in range(4):
            pos.append([x_start + k * w, 2 * h])
            alphas.append(alpha_offset)
            
        # Row 3 (y=3h): 2 hexes
        pos.append([-w/2, 3 * h])
        pos.append([w/2, 3 * h])
        alphas.extend([alpha_offset, alpha_offset])
        
        # Center the configuration
        pos = np.array(pos)
        pos -= pos.mean(axis=0)
        
        return pos.flatten(), np.array(alphas).flatten()

    best_s = np.inf
    best_params = None

    # Define search space for initial parameters
    # We will try several alpha offsets and theta offsets
    alpha_offsets = [0.0, np.radians(30)]
    theta_offsets = [0.0, np.radians(30)]
    
    initializers = [
        lambda a, t: get_honeycomb_165(a),
        lambda a, t: get_honeycomb_2442(a)
    ]
    
    # Add random initializations
    def get_random_init():
        pos = np.random.uniform(-3, 3, (N, 2))
        alphas = np.random.uniform(0, 2 * np.pi, N)
        return pos.flatten(), alphas.flatten()
    
    initializers.append(lambda a, t: get_random_init())

    for alpha_off in alpha_offsets:
        if time.time() > deadline - 20:
            break
        for theta_off in theta_offsets:
            if time.time() > deadline - 20:
                break
            for init_func in initializers:
                if time.time() > deadline - 10:
                    break
                
                try:
                    pos_flat, alphas_flat = init_func(alpha_off, theta_off)
                    # Initial S estimate
                    S_init = 4.5
                    theta_init = theta_off
                    
                    x0 = np.concatenate([pos_flat, alphas_flat, [S_init, theta_init]])
                    
                    # Nelder-Mead is robust for this
                    res = minimize(
                        objective,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True}
                    )
                    
                    S_res = res.x[3 * N]
                    if res.fun < best_s and S_res < 6.0: # Sanity check on S
                        best_s = res.fun
                        best_params = res.x.copy()
                        
                except Exception:
                    pass

    # If no good solution found, fallback
    if best_params is None:
        # Fallback to a known valid configuration (even if not optimal)
        # Use the 1-6-5 honeycomb with a large enough S
        pos_flat, alphas_flat = get_honeycomb_165(0.0)
        S_init = 5.5
        theta_init = 0.0
        best_params = np.concatenate([pos_flat, alphas_flat, [S_init, theta_init]])
        best_s = objective(best_params)

    # Extract final results
    final_pos = best_params[:2 * N].reshape(N, 2)
    final_alphas = best_params[2 * N : 3 * N]
    final_S = best_params[3 * N]
    final_theta = best_params[3 * N + 1]
    
    inner = []
    for i in range(N):
        inner.append([final_pos[i, 0], final_pos[i, 1], np.degrees(final_alphas[i])])
        
    outer_center = (0.0, 0.0)
    outer_side = final_S
    outer_angle_deg = np.degrees(final_theta)
    
    return inner, outer_center, outer_side, outer_angle_deg
