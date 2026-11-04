import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Vertices of a unit regular hexagon centered at origin, rotation 0
    base_angles = np.arange(6) * np.pi / 3
    base_verts = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    
    def get_hex_verts(cx, cy, theta):
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return cx + (base_verts @ R.T)[:, 0], cy + (base_verts @ R.T)[:, 1]
        # Wait, base_verts is (6,2). R is (2,2).
        # base_verts @ R.T -> (6,2).
        # So x = cx + (base_verts @ R.T)[:, 0]
        # y = cy + (base_verts @ R.T)[:, 1]
        
    def sat_penalty(v1, v2):
        # v1, v2 are (6,2) arrays
        # Check overlap using Separating Axis Theorem
        # Axes are normals to edges
        axes = []
        for v in (v1, v2):
            for i in range(6):
                p1 = v[i]
                p2 = v[(i+1)%6]
                edge = p2 - p1
                # Normal
                n = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-9:
                    axes.append(n / norm)
        
        min_overlap = 0.0
        for n in axes:
            proj1 = v1 @ n
            proj2 = v2 @ n
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for gap
            if max1 < min2 or max2 < min1:
                return 0.0 # Disjoint
            
            # Calculate overlap depth
            # Overlap is min(max1, max2) - max(min1, min2)
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_overlap:
                min_overlap = overlap
                
        # If we reach here, they overlap.
        # Penalty is square of the minimum penetration depth
        return (min_overlap ** 2)

    def objective(params):
        # params: x1..x10, y1..y10, theta1..theta10
        # Hex 0 is fixed at (0,0,0)
        centers = np.zeros((N, 2))
        angles = np.zeros(N)
        
        centers[0] = [0.0, 0.0]
        angles[0] = 0.0
        
        centers[1:, 0] = params[0:10]
        centers[1:, 1] = params[10:20]
        angles[1:] = params[20:30]
        
        # Calculate all vertices
        all_verts = []
        for i in range(N):
            vx, vy = get_hex_verts(centers[i, 0], centers[i, 1], angles[i])
            all_verts.append(np.column_stack((vx, vy)))
            
        # Center the cluster
        centroid = np.mean(centers, axis=0)
        
        # Shifted vertices for container calculation
        shifted_verts = []
        for v in all_verts:
            shifted_verts.append(v - centroid)
            
        # Flatten shifted vertices for container calc
        # We need max projection over all vertices and all 6 normals
        # Normals depend on phi
        # A(phi) = max_{v in V} max_{j} (v . n_j(phi))
        
        # We can pre-compute the set of all vertices
        V_all = np.vstack(shifted_verts)
        
        def calc_apothem(phi):
            # Normals for outer hexagon
            # Outer hexagon rotation phi
            # Normals are at angles phi, phi+60, ...
            # Note: The apothem direction is normal to the side.
            # If the hexagon is rotated by phi, the first normal is at angle phi.
            # Actually, for a regular hexagon, the normals are spaced by 60 deg.
            # The "flat" side is perpendicular to the normal.
            # Let's define normals n_j(phi) = (cos(phi + j*pi/3), sin(phi + j*pi/3))
            # The constraint is v . n_j <= A.
            
            max_proj = 0.0
            for j in range(6):
                angle_n = phi + j * np.pi / 3
                n = np.array([np.cos(angle_n), np.sin(angle_n)])
                projs = V_all @ n
                max_p = np.max(projs)
                if max_p > max_proj:
                    max_proj = max_p
            return max_proj
            
        # Ternary search for optimal phi in [0, pi/3]
        a, b = 0.0, np.pi / 3
        for _ in range(50):
            m1 = a + (b - a) / 3
            m2 = b - (b - a) / 3
            f1 = calc_apothem(m1)
            f2 = calc_apothem(m2)
            if f1 < f2:
                b = m2
            else:
                a = m1
        phi_opt = (a + b) / 2
        apothem = calc_apothem(phi_opt)
        
        # Calculate overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                p = sat_penalty(all_verts[i], all_verts[j])
                penalty += p
                
        # Side length S = apothem * 2 / sqrt(3)
        side = apothem * 2.0 / np.sqrt(3.0)
        
        # Total score to minimize: side + large * penalty
        # If penalty is large, side is irrelevant
        return side + 100.0 * penalty

    # Initial guess
    # Hex 0 at (0,0)
    # 1-6 in flower
    # 7-10 random nearby
    x0 = np.zeros(30)
    
    # Flower neighbors
    for k in range(6):
        angle = k * np.pi / 3
        r = np.sqrt(3)
        x0[k] = r * np.cos(angle)
        x0[6+k] = r * np.sin(angle)
        x0[12+k] = 0.0 # angles 0
        
    # Remaining 4 (indices 18-29 in x0? No, x0 has 30 elements)
    # x0[0:6] are x1..x6
    # x0[6:12] are y1..y6
    # x0[12:18] are theta1..theta6
    # x0[18:24] are x7..x10
    # x0[24:30] are y7..y10
    # Wait, my mapping in objective was:
    # centers[1:, 0] = params[0:10]
    # centers[1:, 1] = params[10:20]
    # angles[1:] = params[20:30]
    
    # So:
    # x1..x6 at x0[0:6]
    # y1..y6 at x0[6:12]
    # theta1..theta6 at x0[12:18]
    # x7..x10 at x0[18:22]
    # y7..y10 at x0[22:26]
    # theta7..theta10 at x0[26:30]
    
    # Set flower
    for k in range(6):
        angle = k * np.pi / 3
        r = np.sqrt(3)
        x0[k] = r * np.cos(angle)
        x0[6+k] = r * np.sin(angle)
        x0[12+k] = 0.0
        
    # Set 7-10
    # Place them in the gaps of the flower
    # Gaps are at angles 30, 90, 150, 210, 270, 330?
    # Distance from center to gap center?
    # Let's just place them randomly in a ring
    for k in range(4):
        angle = (2*k + 0.5) * np.pi / 3 # 30, 90, 150, 210 deg
        r = 2.0 * np.sqrt(3)
        x0[18+k] = r * np.cos(angle)
        x0[22+k] = r * np.sin(angle)
        x0[26+k] = 0.0
        
    # Optimize
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
    
    best_params = res.x
    
    # Reconstruct
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    
    centers[0] = [0.0, 0.0]
    angles[0] = 0.0
    
    centers[1:, 0] = best_params[0:10]
    centers[1:, 1] = best_params[10:20]
    angles[1:] = best_params[20:30]
    
    # Find optimal phi and side for the best solution
    all_verts = []
    for i in range(N):
        vx, vy = get_hex_verts(centers[i, 0], centers[i, 1], angles[i])
        all_verts.append(np.column_stack((vx, vy)))
        
    centroid = np.mean(centers, axis=0)
    shifted_verts = [v - centroid for v in all_verts]
    V_all = np.vstack(shifted_verts)
    
    def calc_apothem(phi):
        max_proj = 0.0
        for j in range(6):
            angle_n = phi + j * np.pi / 3
            n = np.array([np.cos(angle_n), np.sin(angle_n)])
            projs = V_all @ n
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        return max_proj
        
    a, b = 0.0, np.pi / 3
    for _ in range(100):
        m1 = a + (b - a) / 3
        m2 = b - (b - a) / 3
        f1 = calc_apothem(m1)
        f2 = calc_apothem(m2)
        if f1 < f2:
            b = m2
        else:
            a = m1
    phi_opt = (a + b) / 2
    apothem = calc_apothem(phi_opt)
    side = apothem * 2.0 / np.sqrt(3.0)
    
    # Construct inner list
    inner = []
    for i in range(N):
        inner.append([centers[i, 0], centers[i, 1], np.degrees(angles[i])])
        
    return inner, (0.0, 0.0), side, np.degrees(phi_opt)

# EVOLVE-BLOCK-END
