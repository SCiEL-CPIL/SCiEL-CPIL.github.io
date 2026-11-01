"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    sqrt3 = np.sqrt(3)

    # Define unit hexagon vertices (flat-top, angle 0)
    unit_hex_verts = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])

    def generate_cluster(config_type):
        """Generate 12 hexagon centers in a fixed configuration."""
        centers = []
        
        if config_type == 0:
            # Compact "Flower" + extras
            coords = [
                (0, 0),
                (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
                (2, 0), (2, -1), (1, -2), (0, -2), (-1, -2)
            ]
            for q, r in coords:
                x = sqrt3 * (q + r/2.0)
                y = 1.5 * r
                centers.append([x, y])
                
        elif config_type == 1:
            # 3x4 Staggered Grid
            for r in range(3):
                for c in range(4):
                    x = c * sqrt3
                    y = r * 1.5
                    if r % 2 == 1:
                        x += sqrt3 / 2
                    centers.append([x, y])
                    
        elif config_type == 2:
            # 4x3 Staggered Grid
            for r in range(4):
                for c in range(3):
                    x = c * sqrt3
                    y = r * 1.5
                    if r % 2 == 1:
                        x += sqrt3 / 2
                    centers.append([x, y])

        elif config_type == 3:
            # "Cross" shape?
            # Center + 4 arms of length 2?
            # (0,0)
            # (1,0), (2,0)
            # (-1,0), (-2,0)
            # (0,1), (0,2) -- Wait, axial coords.
            # Let's use Cartesian directly for simplicity of "Cross"
            centers = [[0,0]]
            for d in range(1, 3):
                centers.append([d*sqrt3, 0])
                centers.append([-d*sqrt3, 0])
                centers.append([0, d*1.5]) # This is not a valid hex neighbor distance for vertical
                # Vertical neighbor distance is 1.5? No.
                # Axial (0,1) -> x=0, y=1.5.
                # So yes.
                centers.append([0, -d*1.5])
            # This gives 1 + 4*2 = 9 hexes. Need 3 more.
            # Add (1,1), (-1,-1), (1,-1)?
            # (1,1) -> x=sqrt3(1.5), y=1.5
            # (-1,-1) -> x=-1.5sqrt3, y=-1.5
            # (1,-1) -> x=0.5sqrt3, y=-1.5
            centers.append([1.5*sqrt3, 1.5])
            centers.append([-1.5*sqrt3, -1.5])
            centers.append([0.5*sqrt3, -1.5])
            
        else:
            # Random-ish compact
            coords = [
                (0, 0),
                (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
                (2, -1), (1, -2), (0, -2), (-1, -1), (-2, 0)
            ]
            for q, r in coords:
                x = sqrt3 * (q + r/2.0)
                y = 1.5 * r
                centers.append([x, y])

        centers = np.array(centers)
        centers -= np.mean(centers, axis=0)
        return centers

    def compute_min_side_for_angle(phi_rad, cluster_centers):
        """
        For a fixed outer hexagon angle phi_rad, find the optimal center (cx, cy)
        that minimizes the required side length.
        Returns (min_side, cx, cy).
        """
        # Normals for outer hexagon
        angles_deg = 30 + np.arange(0, 360, 60) + np.degrees(phi_rad)
        normals_x = np.cos(np.radians(angles_deg))
        normals_y = np.sin(np.radians(angles_deg))
        
        # We want to minimize A such that for all vertices v of all inner hexes:
        # -A <= (v - c) . n_i <= A for all i
        # This is equivalent to:
        # max_i ( (v - c) . n_i ) <= A
        # min_i ( (v - c) . n_i ) >= -A
        # So A >= max( max_i(v . n_i - c . n_i), -min_i(v . n_i - c . n_i) )
        # Let P_i(v) = v . n_i.
        # Let C_i = c . n_i.
        # We need A >= max( max_v(P_i(v)) - C_i, C_i - min_v(P_i(v)) ) for all i.
        # Let MaxP_i = max_v(P_i(v)) and MinP_i = min_v(P_i(v)).
        # We need to choose C_0..C_5 (which are determined by c_x, c_y)
        # to minimize A = max_i( max(MaxP_i - C_i, C_i - MinP_i) ).
        
        # Note: C_i = c_x * nx_i + c_y * ny_i.
        # This is a convex optimization problem in c_x, c_y.
        
        # Pre-compute MaxP and MinP for each normal direction
        # We have 12 hexes, 6 verts each = 72 points.
        # We can stack all vertices.
        all_verts = []
        for center in cluster_centers:
            verts = center[None, :] + unit_hex_verts
            all_verts.append(verts)
        all_verts = np.vstack(all_verts) # (72, 2)
        
        MaxP = np.zeros(6)
        MinP = np.zeros(6)
        
        for i in range(6):
            n = np.array([normals_x[i], normals_y[i]])
            projs = all_verts @ n
            MaxP[i] = np.max(projs)
            MinP[i] = np.min(projs)
            
        # Now minimize A(cx, cy)
        def objective_c(c):
            cx, cy = c
            A = 0.0
            for i in range(6):
                C_i = cx * normals_x[i] + cy * normals_y[i]
                # Required A for this normal
                req_A = max(MaxP[i] - C_i, C_i - MinP[i])
                if req_A > A:
                    A = req_A
            return A

        # Use Nelder-Mead for 2 variables, very fast
        # Initial guess: 0,0
        res = minimize(objective_c, [0.0, 0.0], method='Nelder-Mead',
                       options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500})
        
        return res.fun, res.x[0], res.x[1]

    best_score = np.inf
    best_result = None

    # Try different cluster configurations
    for config_idx in range(4):
        if time.time() > deadline - 10:
            break
            
        centers = generate_cluster(config_idx)
        
        # Sweep angle from 0 to 60 degrees (symmetry of hexagon)
        # Step 1 degree
        for phi_deg in range(0, 61, 1):
            if time.time() > deadline - 5:
                break
                
            phi_rad = np.radians(phi_deg)
            side, cx, cy = compute_min_side_for_angle(phi_rad, centers)
            
            if side < best_score:
                best_score = side
                best_result = {
                    'config': config_idx,
                    'cx': cx,
                    'cy': cy,
                    'phi_deg': phi_deg
                }

    if best_result is None:
        return [[i * sqrt3, 0.0, 0.0] for i in range(12)], (0.0, 0.0), 20.0, 0.0

    # Refine the best angle found with smaller steps
    best_phi_deg = best_result['phi_deg']
    best_config = best_result['config']
    centers = generate_cluster(best_config)
    
    # Local search around best_phi_deg
    for phi_deg in np.arange(best_phi_deg - 2, best_phi_deg + 2, 0.1):
        if time.time() > deadline - 2:
            break
        phi_rad = np.radians(phi_deg)
        side, cx, cy = compute_min_side_for_angle(phi_rad, centers)
        if side < best_score:
            best_score = side
            best_result = {
                'config': best_config,
                'cx': cx,
                'cy': cy,
                'phi_deg': phi_deg
            }

    # Final extraction
    cx = best_result['cx']
    cy = best_result['cy']
    phi_deg = best_result['phi_deg']
    
    inner = []
    for c in centers:
        inner.append([float(c[0]), float(c[1]), 0.0])
        
    return inner, (float(cx), float(cy)), float(best_score), float(phi_deg)

# EVOLVE-BLOCK-END
