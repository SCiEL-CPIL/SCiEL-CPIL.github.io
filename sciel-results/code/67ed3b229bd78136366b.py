import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    sqrt3 = math.sqrt(3)
    
    # Precompute unit hexagon vertices (flat-top orientation relative to local x-axis)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    unit_verts = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])

    def get_minkowski_support(theta_rel):
        """
        Compute the support function of the Minkowski sum of two unit hexagons
        with relative rotation theta_rel.
        The Minkowski sum is a convex polygon (up to 12 vertices).
        We approximate it by checking the support of the sum of vertices.
        For exactness, we can use the fact that the Minkowski sum of two convex polygons
        can be computed, but for speed in the inner loop, we use the property:
        h_A+B(n) = h_A(n) + h_B(n).
        Since both are regular hexagons, h(n) depends only on the angle of n relative to the hexagon's symmetry axes.
        """
        # The support function of a regular hexagon with side 1 (circumradius 1)
        # in direction n (angle phi) is:
        # h(phi) = cos(phi mod 60 - 30) * (sqrt(3)/2) ? No.
        # Let's derive h(phi) for unit hexagon (R=1).
        # Vertices at 0, 60...
        # Edge normal angles: 30, 90, 150, 210, 270, 330.
        # Distance from center to edge (apothem) = sqrt(3)/2.
        # h(phi) = apothem / cos((phi - edge_normal) mod 60 - 30) ??
        # Simpler: h(phi) = max(v_i . n).
        # Due to symmetry, we can map phi to [0, 30) and use the edge normal at 30.
        # Actually, let's just precompute a lookup table or use the formula.
        # Formula: h(phi) = (sqrt(3)/2) / cos(delta) where delta is angle to nearest edge normal.
        # Nearest edge normal to phi is within 30 degrees.
        
        # For the sum of two hexagons with relative rotation theta:
        # h_sum(n) = h_hex(n) + h_hex(n - theta)
        
        pass # We will implement a direct geometric check instead for robustness in the penalty

    def min_dist_for_rotation(theta_rel):
        """
        Calculate the minimum distance between centers of two unit hexagons
        with relative rotation theta_rel such that they do not overlap.
        This is the radius of the Minkowski difference H1 - H2.
        """
        # The Minkowski sum H1 + H2 (where H2 is rotated and reflected) determines the exclusion zone.
        # Actually, we want the support function of H1 + (-H2_rotated).
        # Let's use a numerical approach for the specific theta_rel if needed, 
        # but we can approximate it.
        # For theta_rel = 0, dist = sqrt(3).
        # For theta_rel = 30, dist = 2.0 (vertex to vertex).
        # It varies periodically.
        
        # We will compute the max projection of the Minkowski sum vertices.
        # Vertices of H1: unit_verts
        # Vertices of -H2_rotated: -rotate(unit_verts, theta_rel)
        
        # To be fast, we precompute the Minkowski sum polygon for a grid of theta values
        # or just use a conservative bound if speed is an issue, but we need accuracy.
        
        # Let's compute the support function of the Minkowski sum S = H1 + H2_rot.
        # The minimum distance is the max over all directions n of (h1(n) + h2_rot(n)).
        
        # h1(n) for unit hexagon:
        # Let phi = angle of n.
        # Map phi to [0, 60).
        # The edge normal is at 30.
        # h1(phi) = (sqrt(3)/2) / cos(phi - 30) for phi in [0, 60]?
        # Check phi=0: cos(-30)=sqrt(3)/2. h1 = (sqrt(3)/2)/(sqrt(3)/2) = 1. Correct (vertex).
        # Check phi=30: cos(0)=1. h1 = sqrt(3)/2. Correct (edge midpoint).
        
        def h_hex(phi):
            # phi in radians
            # Reduce to [0, 60)
            phi = phi % (math.pi / 3)
            # Edge normal is at 30 deg = pi/6
            delta = abs(phi - math.pi/6)
            # If delta > pi/6, it's closer to the next edge?
            # The range [0, 60) covers one sector. The normal is at 30.
            # Max delta is 30.
            # h = apothem / cos(delta)
            return (sqrt3 / 2) / math.cos(delta)

        # We need max over n of h_hex(angle(n)) + h_hex(angle(n) - theta_rel)
        # The maximum will occur when the "corners" align or edges align.
        # We can sample the angle n.
        max_d = 0.0
        # Sample 60 points
        for i in range(60):
            ang = i * math.pi / 30
            d = h_hex(ang) + h_hex(ang - theta_rel)
            if d > max_d:
                max_d = d
        return max_d

    # Precompute min_dist for a range of relative rotations
    # Relative rotation ranges from 0 to 60 degrees (symmetry)
    # We will create a lookup table
    theta_grid = np.linspace(0, math.pi/3, 61)
    min_dist_table = np.array([min_dist_for_rotation(t) for t in theta_grid])

    def get_min_dist(theta_rel):
        # Normalize theta_rel to [0, 60)
        theta_rel = theta_rel % (math.pi / 3)
        # Find index
        idx = int((theta_rel / (math.pi / 3)) * 60)
        if idx >= 60: idx = 59
        return min_dist_table[idx]

    def calc_S(positions, alpha_deg, beta_deg):
        # positions: (12, 2)
        # alpha_deg: rotation of inner hexagons (all same)
        # beta_deg: rotation of outer hexagon
        alpha_rad = math.radians(alpha_deg)
        beta_rad = math.radians(beta_deg)
        
        max_proj = 0.0
        for k in range(6):
            normal_angle = math.radians(30) + beta_rad + k * math.radians(60)
            n = np.array([math.cos(normal_angle), math.sin(normal_angle)])
            
            # Support of unit hexagon in direction n_rot = n rotated by -alpha
            # h_unit(n_rot)
            # Using the same logic:
            # n_rot angle relative to hexagon frame
            # The hexagon is rotated by alpha. So the direction n in global frame
            # is n_rot in local frame.
            # h_unit(n_rot) = h_hex(angle(n_rot))
            
            # Angle of n is normal_angle.
            # Angle of n_rot is normal_angle - alpha_rad.
            h_val = (sqrt3 / 2) / math.cos(abs((normal_angle - alpha_rad) % (math.pi/3) - math.pi/6))
            
            # Project all positions
            projs = positions @ n
            max_p_proj = np.max(projs)
            
            S_k = max_p_proj + h_val
            if S_k > max_proj:
                max_proj = S_k
                
        # The side length S of the outer hexagon is related to the support distance.
        # For a regular hexagon, the distance from center to edge (apothem) is S * sqrt(3)/2.
        # So S = 2 * max_proj / sqrt3.
        return 2.0 * max_proj / sqrt3

    def objective(params):
        pos = params[:24].reshape(12, 2)
        alpha = params[24]
        beta = params[25]
        
        S = calc_S(pos, alpha, beta)
        
        # Overlap penalty
        penalty = 0.0
        # We use the precomputed min_dist table
        for i in range(12):
            for j in range(i+1, 12):
                dist = np.linalg.norm(pos[i] - pos[j])
                # Relative rotation is 0 since all inner hexagons have same angle alpha
                # Wait, if all inner hexagons have the SAME angle, the relative rotation is 0.
                # So min_dist is always min_dist_table[0] = sqrt(3).
                # This simplifies things greatly!
                # If we allowed different angles per hexagon, we'd need per-pair checks.
                # The task says "each inner hexagon may translate and rotate".
                # The previous code assumed a single alpha for all.
                # Let's stick to a single alpha for all to reduce dimensionality and ensure stability.
                # If single alpha, relative rotation is 0.
                d_min = sqrt3
                if dist < d_min:
                    penalty += (d_min - dist) ** 2
                    
        return S + 50.0 * penalty

    # Initial configuration: Hexagonal cluster
    # 1 center, 6 ring 1, 5 ring 2 (missing one to make 12? 1+6+5=12)
    # Or 3x4 grid?
    # Let's try a few initial configs.
    
    configs = []
    
    # Config 1: Center + 6 + 5
    hex_cluster = []
    hex_cluster.append((0, 0))
    for k in range(6):
        angle = math.radians(60 * k)
        hex_cluster.append((sqrt3 * math.cos(angle), sqrt3 * math.sin(angle)))
    # Ring 2 positions (distance 2*sqrt3)
    # We need 5 more. Let's place them at angles 30, 90, 150, 210, 270
    for k in range(5):
        angle = math.radians(60 * k + 30)
        hex_cluster.append((2*sqrt3 * math.cos(angle), 2*sqrt3 * math.sin(angle)))
    # Center
    xs = [p[0] for p in hex_cluster]
    ys = [p[1] for p in hex_cluster]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    hex_cluster = [(x-cx, y-cy) for x, y in hex_cluster]
    configs.append(hex_cluster)

    # Config 2: 3x4 Parallelogram (Hex grid)
    para = []
    for q in range(3):
        for r in range(4):
            x = sqrt3 * (q + r/2.0)
            y = 1.5 * r
            para.append((x, y))
    xs = [p[0] for p in para]
    ys = [p[1] for p in para]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    para = [(x-cx, y-cy) for x, y in para]
    configs.append(para)

    best_S = float('inf')
    best_params = None

    for pos_init in configs:
        if time.time() > deadline:
            break
            
        pos_flat = np.array(pos_init).flatten()
        
        # Try a few initial angles
        for a0 in [0, 30, 15]:
            for b0 in [0, 30, 15]:
                if time.time() > deadline:
                    break
                x0 = np.append(pos_flat, [a0, b0])
                try:
                    # Use Nelder-Mead
                    res = minimize(objective, x0, method='Nelder-Mead', 
                                   options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
                    
                    # Check validity
                    pos_res = res.x[:24].reshape(12, 2)
                    valid = True
                    for i in range(12):
                        for j in range(i+1, 12):
                            if np.linalg.norm(pos_res[i] - pos_res[j]) < sqrt3 * 0.999:
                                valid = False
                                break
                        if not valid: break
                    
                    if valid:
                        S_val = calc_S(pos_res, res.x[24], res.x[25])
                        if S_val < best_S:
                            best_S = S_val
                            best_params = res.x
                except Exception as e:
                    pass

    if best_params is None:
        # Fallback
        pos_init = configs[0]
        pos_flat = np.array(pos_init).flatten()
        best_params = np.append(pos_flat, [0, 0])
        best_S = calc_S(np.array(pos_init), 0, 0)

    pos_final = best_params[:24].reshape(12, 2)
    alpha = best_params[24]
    beta = best_params[25]
    
    inner = []
    for i in range(12):
        inner.append([pos_final[i][0], pos_final[i][1], alpha])
        
    return inner, (0.0, 0.0), best_S, beta
