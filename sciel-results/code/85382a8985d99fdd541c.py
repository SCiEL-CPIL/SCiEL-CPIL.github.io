"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # Flat-top hexagon vertices for side length 1, centered at origin
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    base_vertices = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Pre-compute rotation matrices for common angles? 
    # No, angles are continuous. We will compute on the fly.
    
    def get_hex_verts(center, angle_deg, side=1.0):
        theta = np.deg2rad(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return center + (side * base_vertices @ R.T)

    def point_in_hex_dist(pt, center, angle_deg, side=1.0):
        """
        Returns the signed distance from point to the hexagon boundary.
        Positive if outside, negative if inside.
        """
        theta = np.deg2rad(angle_deg)
        c, s = np.cos(-theta), np.sin(-theta)
        # Inverse rotation
        R_inv = np.array([[c, -s], [s, c]])
        local_pt = R_inv @ (pt - center)
        
        x, y = local_pt
        # Hexagon inequalities for side length 'side':
        # 1. |x| <= side
        # 2. |0.5x + sqrt(3)/2 y| <= sqrt(3)/2 * side
        # 3. |-0.5x + sqrt(3)/2 y| <= sqrt(3)/2 * side
        
        # Distance to each side (outward normal direction)
        d1 = x - side
        d2 = 0.5*x + (np.sqrt(3)/2)*y - (np.sqrt(3)/2)*side
        d3 = -0.5*x + (np.sqrt(3)/2)*y - (np.sqrt(3)/2)*side
        d4 = -x - side
        d5 = -0.5*x - (np.sqrt(3)/2)*y - (np.sqrt(3)/2)*side
        d6 = 0.5*x - (np.sqrt(3)/2)*y - (np.sqrt(3)/2)*side
        
        # The signed distance is the maximum of these values (if positive, it's the violation)
        # To be smooth, we use a soft-max (log-sum-exp) approximation
        # However, for the objective function in SA, we can just use the max or a smooth max.
        # Let's use a smooth max for the gradient in L-BFGS later.
        # For SA, the exact value matters less than the relative ordering, but smoothness helps.
        
        # Standard max is non-differentiable. Let's implement a soft-max.
        # But wait, the distance to a convex polygon is the max of the linear functions defining the sides.
        # We can just return the max for SA (it's fine, SA doesn't need gradients), 
        # and use a soft-max for the final L-BFGS polish.
        
        # Actually, let's return the raw max for simplicity in SA, and handle smoothing in the objective wrapper.
        return max(d1, d2, d3, d4, d5, d6)

    def hex_overlap_distance(v1, c1, a1, v2, c2, a2):
        """
        Returns the minimum distance required to separate the two hexagons.
        If they are disjoint, returns 0 (or negative if we want signed, but let's stick to positive violation).
        We check vertices of v1 against v2 and vertices of v2 against v1.
        If a vertex is inside the other hex, the distance is the penetration depth.
        """
        max_penetration = 0.0
        
        # Check vertices of hex1 against hex2
        for pt in v1:
            # Distance from pt to hex2 (positive if outside, negative if inside)
            # We want the magnitude of violation.
            # If pt is inside hex2, point_in_hex_dist returns a negative value.
            # The violation is -dist.
            d = point_in_hex_dist(pt, c2, a2)
            if d < 0:
                max_penetration = max(max_penetration, -d)
                
        # Check vertices of hex2 against hex1
        for pt in v2:
            d = point_in_hex_dist(pt, c1, a1)
            if d < 0:
                max_penetration = max(max_penetration, -d)
                
        return max_penetration

    def objective(params, smooth_beta=100.0):
        """
        Objective function for optimization.
        params: [x1, y1, a1, ..., xN, yN, aN, S]
        """
        if time.time() > deadline:
            return 1e9
            
        S = params[-1]
        if S < 1.0: S = 1.0
        
        inner_params = params[:-1]
        
        penalty = 0.0
        
        # Soft bound on S to keep it reasonable
        if S > 10.0:
            penalty += 100.0 * (S - 10.0)**2

        hex_data = []
        for i in range(N):
            x_i = inner_params[3*i]
            y_i = inner_params[3*i+1]
            a_i = inner_params[3*i+2]
            c_i = np.array([x_i, y_i])
            v_i = get_hex_verts(c_i, a_i)
            hex_data.append((c_i, a_i, v_i))
            
            # Containment in outer hexagon (side S, center 0,0, angle 0)
            # We need all vertices of inner hex i to be inside outer hex.
            # Violation is max(0, dist_to_outer)
            for pt in v_i:
                # Distance to outer hex (side S)
                # Scale the local coordinates by 1/S? No, just use side=S in the function.
                # But point_in_hex_dist assumes side=1.0 by default. 
                # We can scale the point and center? 
                # Easier: just call a version that takes side.
                # Let's inline the check for speed.
                px, py = pt
                d1 = px - S
                d2 = 0.5*px + (np.sqrt(3)/2)*py - S*np.sqrt(3)/2
                d3 = -0.5*px + (np.sqrt(3)/2)*py - S*np.sqrt(3)/2
                d4 = -px - S
                d5 = -0.5*px - (np.sqrt(3)/2)*py - S*np.sqrt(3)/2
                d6 = 0.5*px - (np.sqrt(3)/2)*py - S*np.sqrt(3)/2
                
                violation = max(d1, d2, d3, d4, d5, d6)
                if violation > 0:
                    # Smooth penalty
                    penalty += violation**2
                
        # Overlap between inner hexagons
        for i in range(N):
            c1, a1, v1 = hex_data[i]
            for j in range(i+1, N):
                c2, a2, v2 = hex_data[j]
                
                # Quick reject
                dist_sq = (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2
                if dist_sq > 4.0: # (2*1)^2, if centers are more than 2 apart, they might still touch if rotated, but 2 is max radius.
                    # Actually, max distance from center to vertex is 1. So if dist > 2, no overlap.
                    continue
                    
                overlap = hex_overlap_distance(v1, c1, a1, v2, c2, a2)
                if overlap > 0:
                    penalty += 10.0 * overlap**2
                    
        return S + penalty

    def objective_smooth(params):
        """
        Smooth version of objective for L-BFGS.
        Uses log-sum-exp for max operations.
        """
        if time.time() > deadline:
            return 1e9
            
        S = params[-1]
        if S < 1.0: S = 1.0
        
        inner_params = params[:-1]
        
        penalty = 0.0
        beta = 50.0 # Smoothing parameter
        
        def soft_max(vals):
            # vals is a list or array
            v = np.array(vals)
            v_max = np.max(v)
            if v_max == -np.inf:
                return 0.0
            return v_max + np.log(np.sum(np.exp(v - v_max))) / beta
        
        hex_data = []
        for i in range(N):
            x_i = inner_params[3*i]
            y_i = inner_params[3*i+1]
            a_i = inner_params[3*i+2]
            c_i = np.array([x_i, y_i])
            v_i = get_hex_verts(c_i, a_i)
            hex_data.append((c_i, a_i, v_i))
            
            # Containment
            for pt in v_i:
                px, py = pt
                d1 = px - S
                d2 = 0.5*px + (np.sqrt(3)/2)*py - S*np.sqrt(3)/2
                d3 = -0.5*px + (np.sqrt(3)/2)*py - S*np.sqrt(3)/2
                d4 = -px - S
                d5 = -0.5*px - (np.sqrt(3)/2)*py - S*np.sqrt(3)/2
                d6 = 0.5*px - (np.sqrt(3)/2)*py - S*np.sqrt(3)/2
                
                # We only care about positive violations
                # soft_max of [d1, d2, d3, d4, d5, d6, 0]
                # If all are negative, soft_max will be close to 0 (log(1)/beta = 0)
                # Actually, we want max(0, max(d_i)).
                # So we add 0 to the list.
                violation = soft_max([d1, d2, d3, d4, d5, d6, 0.0])
                penalty += violation**2
                
        for i in range(N):
            c1, a1, v1 = hex_data[i]
            for j in range(i+1, N):
                c2, a2, v2 = hex_data[j]
                
                dist_sq = (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2
                if dist_sq > 4.0:
                    continue
                    
                # Calculate overlap
                # We need the max penetration depth.
                # Penetration is -min_distance.
                # Let's collect all -dist values (where dist is signed distance, neg if inside)
                # Actually, point_in_hex_dist returns positive if outside, negative if inside.
                # So violation is -dist if dist < 0.
                # We want max(0, -min_dist).
                
                # To avoid calculating all vertices if we can reject?
                # We already rejected if dist_sq > 4.
                
                dists = []
                for pt in v1:
                    # Calculate signed distance to hex2
                    theta = np.deg2rad(a2)
                    c, s = np.cos(-theta), np.sin(-theta)
                    R_inv = np.array([[c, -s], [s, c]])
                    local_pt = R_inv @ (pt - c2)
                    lx, ly = local_pt
                    d1 = lx - 1.0
                    d2 = 0.5*lx + (np.sqrt(3)/2)*ly - np.sqrt(3)/2
                    d3 = -0.5*lx + (np.sqrt(3)/2)*ly - np.sqrt(3)/2
                    d4 = -lx - 1.0
                    d5 = -0.5*lx - (np.sqrt(3)/2)*ly - np.sqrt(3)/2
                    d6 = 0.5*lx - (np.sqrt(3)/2)*ly - np.sqrt(3)/2
                    # Signed distance is max(d1...d6). If negative, it's inside.
                    # We want the magnitude of violation: max(0, -signed_dist)
                    # signed_dist = soft_max([d1...d6])
                    # violation = max(0, -signed_dist) = -min(0, signed_dist)
                    # This is getting complex for soft-max.
                    # Let's just use the hard max for the overlap check in the smooth objective?
                    # No, L-BFGS needs gradients.
                    # Let's approximate the overlap penalty with a smooth function of the center distance and relative angle?
                    # That's hard.
                    
                    # Alternative: Just use the hard max in the objective, but use a different optimizer?
                    # Or use a penalty method where we only penalize if overlap > 0, and use a smooth approximation of the indicator?
                    
                    # Let's stick to the hard max for overlap in the smooth objective as well, 
                    # but rely on the fact that L-BFGS can handle some non-smoothness if we use a small step size?
                    # Actually, L-BFGS might fail or stall.
                    
                    # Let's try a different approach for the smooth objective:
                    # Use the sum of squared violations, where violation is calculated via soft-max.
                    
                    # Signed distance d_s = soft_max([d1, d2, d3, d4, d5, d6])
                    # If d_s < 0, the point is inside. Violation = -d_s.
                    # If d_s > 0, the point is outside. Violation = 0.
                    # So Violation = max(0, -d_s) = -min(0, d_s).
                    # This is still non-smooth at d_s=0.
                    
                    # Let's just use the hard max for now in the "smooth" objective too, 
                    # and hope the SA phase gets us close enough that the non-smoothness doesn't derail L-BFGS.
                    # Or, better yet, use COBYLA? No, L-BFGS is faster.
                    
                    # Let's just use the hard max.
                    d_max = max(d1, d2, d3, d4, d5, d6)
                    if d_max < 0:
                        dists.append(-d_max)
                    else:
                        dists.append(0.0)

                for pt in v2:
                    theta = np.deg2rad(a1)
                    c, s = np.cos(-theta), np.sin(-theta)
                    R_inv = np.array([[c, -s], [s, c]])
                    local_pt = R_inv @ (pt - c1)
                    lx, ly = local_pt
                    d1 = lx - 1.0
                    d2 = 0.5*lx + (np.sqrt(3)/2)*ly - np.sqrt(3)/2
                    d3 = -0.5*lx + (np.sqrt(3)/2)*ly - np.sqrt(3)/2
                    d4 = -lx - 1.0
                    d5 = -0.5*lx - (np.sqrt(3)/2)*ly - np.sqrt(3)/2
                    d6 = 0.5*lx - (np.sqrt(3)/2)*ly - np.sqrt(3)/2
                    d_max = max(d1, d2, d3, d4, d5, d6)
                    if d_max < 0:
                        dists.append(-d_max)
                    else:
                        dists.append(0.0)
                        
                # Max violation
                if dists:
                    max_viol = max(dists)
                    if max_viol > 0:
                        penalty += 10.0 * max_viol**2

        return S + penalty

    # Initial guess: Hexagonal packing
    # Side length 1 hexagons.
    # Distance between centers for side-to-side contact is sqrt(3) ~ 1.732.
    # For a cluster of 11, a hexagonal arrangement is good.
    # Center at (0,0).
    # 1 center, 6 around, 4 more? 
    # Let's place them in a grid.
    dx = np.sqrt(3)
    dy = 1.5 # For flat top, vertical distance between rows is 1.5?
    # Flat top hex:
    # Vertices at 0, 60, 120, 180, 240, 300.
    # Width (x) = 2. Height (y) = sqrt(3).
    # Neighbors:
    # Left/Right: dx = 2, dy = 0.
    # Top/Bottom: dx = 0, dy = sqrt(3)? No.
    # Let's check vertices: (1,0), (0.5, 0.866), (-0.5, 0.866), (-1,0), (-0.5, -0.866), (0.5, -0.866).
    # Right neighbor center: (2, 0).
    # Top-right neighbor center: (1.5, 0.866*1.5)? No.
    # The distance between centers of touching hexagons is sqrt(3) for side-to-side?
    # No, the distance between centers is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
    # So neighbors are at distance sqrt(3).
    # Angles: 0, 60, 120, 180, 240, 300.
    
    centers = []
    angles = []
    # Center
    centers.append((0,0))
    angles.append(0)
    # Ring 1
    for k in range(6):
        ang = np.deg2rad(60*k)
        x = np.sqrt(3) * np.cos(ang)
        y = np.sqrt(3) * np.sin(ang)
        centers.append((x,y))
        angles.append(0) # Keep same orientation for simplicity
        
    # Ring 2 (partial, need 4 more)
    # We have 7. Need 4 more.
    # Place them in the gaps of Ring 1?
    # Gap centers are at distance 2*sqrt(3) from center?
    # Or just extend the ring.
    # Let's add 4 more at distance 2*sqrt(3) roughly.
    for k in range(4):
        ang = np.deg2rad(60*k + 30) # Offset
        x = 2.0 * np.sqrt(3) * np.cos(ang)
        y = 2.0 * np.sqrt(3) * np.sin(ang)
        centers.append((x,y))
        angles.append(0)
        
    # Shuffle to make sure we have 11
    # Actually, let's just use a simple grid initialization to be safe.
    # 11 hexagons in a 3x4 grid?
    # Let's stick to the hexagonal cluster, it's more compact.
    
    x0 = []
    for i in range(N):
        if i < len(centers):
            x0.extend([centers[i][0], centers[i][1], angles[i]])
        else:
            # Fallback
            x0.extend([0,0,0])
    x0.append(4.0) # Initial S
    
    x0 = np.array(x0)
    
    best_result = None
    best_score = 1e9
    
    # Simulated Annealing
    current_params = x0.copy()
    current_score = objective(current_params)
    
    T = 1.0
    T_min = 0.01
    alpha = 0.99
    steps = 2000
    
    # Track best
    best_params = current_params.copy()
    best_score_sa = current_score
    
    for step in range(steps):
        if time.time() > deadline - 10: # Leave time for polish
            break
            
        # Perturb
        # Randomly pick a few hexagons to move/rotate
        num_perturb = np.random.randint(1, 4)
        idxs = np.random.choice(N, num_perturb, replace=False)
        
        new_params = current_params.copy()
        for idx in idxs:
            # Move
            new_params[3*idx] += np.random.normal(0, 0.5)
            new_params[3*idx+1] += np.random.normal(0, 0.5)
            # Rotate
            new_params[3*idx+2] += np.random.normal(0, 10)
            # Adjust S
            if np.random.rand() < 0.1:
                new_params[-1] += np.random.normal(0, 0.2)
                
        new_score = objective(new_params)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_params = new_params
            current_score = new_score
            
            if current_score < best_score_sa:
                best_score_sa = current_score
                best_params = current_params.copy()
                
        T *= alpha
        if T < T_min:
            T = T_min
            
    best_result = best_params
    best_score = best_score_sa
    
    # Polish with L-BFGS
    if time.time() < deadline - 5:
        try:
            res = minimize(objective_smooth, best_result, method='L-BFGS-B', 
                           options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.fun < best_score:
                best_result = res.x
                best_score = res.fun
        except Exception:
            pass
            
    # Extract best solution
    S = best_result[-1]
    inner = []
    for i in range(N):
        x = best_result[3*i]
        y = best_result[3*i+1]
        a = best_result[3*i+2]
        inner.append([x, y, a])
        
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
