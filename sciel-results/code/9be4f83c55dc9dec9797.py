# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _edge_normals(hex_verts):
    normals = []
    centroid = hex_verts.mean(axis=0)
    for i in range(6):
        p1 = hex_verts[i]
        p2 = hex_verts[(i + 1) % 6]
        edge = p2 - p1
        n = np.array([-edge[1], edge[0]])
        mid = (p1 + p2) / 2.0
        if np.dot(n, mid - centroid) < 0:
            n = -n
        n = n / np.linalg.norm(n)
        normals.append(n)
    return np.array(normals)

def _check_separation(v1, n1, v2, n2):
    # v1, v2: (6, 2)
    # n1, n2: (6, 2)
    all_normals = np.vstack([n1, n2]) # (12, 2)
    
    # Project vertices onto normals
    # proj1: (6, 12), proj2: (6, 12)
    proj1 = v1 @ all_normals.T
    proj2 = v2 @ all_normals.T
    
    max1 = np.max(proj1, axis=0)
    min1 = np.min(proj1, axis=0)
    max2 = np.max(proj2, axis=0)
    min2 = np.min(proj2, axis=0)
    
    # Check separation condition:
    # max1 < min2 - eps OR max2 < min1 - eps
    eps = 1e-6
    sep = (max1 < min2 - eps) | (max2 < min1 - eps)
    return np.any(sep)

def _min_outer_side(inner_hexes, angle_deg):
    # inner_hexes: list of (verts, normals)
    # angle_deg: angle of outer hexagon
    # Returns minimum side length S
    a = np.deg2rad(angle_deg)
    # Outer hexagon normals are at angles a + pi/6 + k*pi/3
    # Distance from center to edge is S * sqrt(3)/2
    # Condition: v . n_k <= S * sqrt(3)/2
    # S >= (2/sqrt(3)) * max(v . n_k)
    
    normals = np.zeros((6, 2))
    for k in range(6):
        ang = a + np.pi/6 + k * np.pi/3
        normals[k] = [np.cos(ang), np.sin(ang)]
    
    max_proj = -np.inf
    for verts, _ in inner_hexes:
        projs = verts @ normals.T
        max_proj = np.maximum(max_proj, np.max(projs))
        
    return (2.0 / np.sqrt(3.0)) * max_proj

def hexagon_packing_11():
    t_start = time.time()
    deadline = 600.0
    safety_margin = 30.0
    
    # Seed arrangement: Flower pattern
    # Center at (0,0)
    # 6 neighbors at distance 2*sqrt(3)/2? No, distance between centers of touching hexagons is 2 * inradius = 2 * sqrt(3)/2 = sqrt(3) ~ 1.732?
    # Wait, side length is 1. Inradius is sqrt(3)/2. Distance between centers of touching hexagons is 2 * inradius = sqrt(3).
    # Let's verify.
    # Hexagon vertices at distance 1 from center.
    # Edge length 1.
    # Distance from center to edge is sqrt(3)/2.
    # If two hexagons touch edge-to-edge, distance between centers is 2 * (sqrt(3)/2) = sqrt(3).
    
    # Let's use a known good packing.
    # 1 in center, 6 around it.
    # The 6 around it form a ring.
    # Then 4 more?
    # Total 11.
    # Center: (0,0)
    # Ring 1: 6 hexagons at distance sqrt(3) from center, angles 0, 60, ...
    # Ring 2: 4 hexagons?
    # Maybe a hexagonal grid arrangement.
    # Coordinates in hex grid (axial):
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # This is 7 hexagons.
    # Add (2,0), (2,-1), (1,1), (0,2)?
    # Let's just use a random start or the previous seed but optimized.
    
    # The previous seed was:
    # [0, 0], [-2.5, 0], [2.5, 0], ...
    # Distance 2.5 is > sqrt(3) ~ 1.732. So they don't touch.
    # We can tighten them.
    
    # Let's generate a dense seed.
    # Axial to Cartesian:
    # x = size * (q + r/2)
    # y = size * (sqrt(3)/2 * r)
    # size = 1 (side length)
    # Distance between centers is sqrt(3).
    # My formula: x = 1.5 * q, y = sqrt(3)/2 * (q + 2r)? No.
    # Standard pointy-top:
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    # Distance between (0,0) and (1,0) is sqrt(3). Correct.
    
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        return np.array([x, y])
    
    # 7 hexagons in center cluster:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    hexes_axial = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (2,-1), (1,1), (0,2) # 4 more
    ]
    # Check if these 11 are valid (non-overlapping).
    # In hex grid, these are distinct cells, so they don't overlap.
    # They touch. We need them to be separated by > 1e-6.
    # So we should expand them slightly or move them apart.
    # Actually, the optimization will move them.
    # But starting with touching hexagons might cause issues with "separated by > 1e-6".
    # Let's scale the positions by 1.001 to ensure initial separation.
    
    seed_positions = np.array([axial_to_cartesian(q, r) for q, r in hexes_axial])
    seed_positions *= 1.001 # Ensure initial separation
    
    def _build_inner_hexes(positions, angles):
        hexes = []
        for i in range(11):
            verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            norms = _edge_normals(verts)
            hexes.append((verts, norms))
        return hexes

    def _objective(params):
        if time.time() > deadline - safety_margin:
            return 1e10
            
        positions = params[:22].reshape(11, 2)
        angles = params[22:33]
        outer_angle = params[33]
        
        inner_hexes = _build_inner_hexes(positions, angles)
        
        # Check separation
        for i in range(11):
            for j in range(i + 1, 11):
                if not _check_separation(inner_hexes[i][0], inner_hexes[i][1],
                                         inner_hexes[j][0], inner_hexes[j][1]):
                    return 1e10
        
        S = _min_outer_side(inner_hexes, outer_angle)
        if not np.isfinite(S):
            return 1e10
        return S

    # Initial guess
    x0 = np.zeros(34)
    x0[:22] = seed_positions.flatten()
    x0[22:33] = 0.0
    x0[33] = 0.0
    
    best_params = x0.copy()
    best_S = _objective(x0)
    
    # Multi-start optimization
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() > deadline - 60:
            break
            
        if r == 0:
            current_params = x0.copy()
        else:
            # Perturb best params
            current_params = best_params + np.random.normal(0, 0.5, 34)
        
        try:
            res = minimize(
                _objective,
                current_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Final result
    positions = best_params[:22].reshape(11, 2)
    angles = best_params[22:33]
    outer_angle = best_params[33]
    
    inner_hex_data = np.column_stack([positions, angles])
    outer_hex_data = np.array([0.0, 0.0, outer_angle])
    outer_hex_side_length = float(best_S)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
