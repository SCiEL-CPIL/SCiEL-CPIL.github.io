import numpy as np
import time
from scipy.optimize import minimize

N = 11
_DEADLINE = 140.0

def _hex_vertices(cx, cy, theta, side=1.0):
    angles = theta + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles),
                                cy + side * np.sin(angles)])

def _mhe_side_and_params(positions, angles, outer_center, outer_angle):
    """
    Returns side length of the minimum enclosing regular hexagon.
    positions: (N, 2) array
    angles: (N,) array
    """
    # Get all vertices
    all_verts = []
    for i in range(len(positions)):
        v = _hex_vertices(positions[i, 0], positions[i, 1], angles[i], 1.0)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    # Normals for the 6 sides of the outer hexagon
    # The outer hexagon is rotated by outer_angle.
    # The normals to the sides are at outer_angle + pi/6 + k*pi/3
    normals_angles = outer_angle + np.pi/6 + np.pi/3 * np.arange(6)
    normals = np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])
    
    # Project vertices onto normals
    proj = all_verts @ normals.T
    c_proj = outer_center @ normals.T
    diffs = np.abs(proj - c_proj)
    max_diff = diffs.max()
    side = max_diff * 2.0 / np.sqrt(3.0)
    
    return side

def _overlap_penalty(positions, angles):
    """
    Penalty for overlaps.
    Uses center distance as a proxy.
    """
    pen = 0.0
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < 2.0:
                pen += (2.0 - d) ** 2
    return pen

def _objective(v):
    """
    v: [x1, y1, a1, ..., xN, yN, aN, cx, cy, theta]
    """
    hex_vars = v[:3*N]
    outer_vars = v[3*N:]
    
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    for i in range(N):
        positions[i, 0] = hex_vars[i*3]
        positions[i, 1] = hex_vars[i*3+1]
        angles[i] = hex_vars[i*3+2]
        
    cx = outer_vars[0]
    cy = outer_vars[1]
    theta = outer_vars[2]
    
    outer_center = np.array([cx, cy])
    
    side = _mhe_side_and_params(positions, angles, outer_center, theta)
    pen = _overlap_penalty(positions, angles)
    
    return side + 10.0 * pen

def _generate_lattice_configs():
    """Generate a few hexagonal lattice configurations."""
    configs = []
    
    # Hex grid coordinates in axial system
    # Center (0,0)
    # Ring 1: 6 hexes
    # Ring 2: 12 hexes
    # We need 11 total.
    
    axial_coords = [(0,0)]
    # Ring 1
    axial_coords += [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    # Ring 2
    axial_coords += [(2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)]
    
    selected_axial = axial_coords[:N]
    
    # Convert to Cartesian
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    lattice_centers = []
    for q, r in selected_axial:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q/2)
        lattice_centers.append((x, y))
        
    positions = np.array(lattice_centers)
    angles = np.zeros(N)
    cx = np.mean(positions[:, 0])
    cy = np.mean(positions[:, 1])
    theta = 0.0
    
    v = np.concatenate([
        np.array([[x, y, a] for x, y, a in zip(positions[:,0], positions[:,1], angles)]).flatten(),
        [cx, cy, theta]
    ])
    configs.append(v)
    
    # Random perturbations
    for _ in range(5):
        v_perturbed = v + np.random.randn(len(v)) * 0.1
        configs.append(v_perturbed)
        
    return configs

def construct():
    start_time = time.time()
    best_result = None
    best_side = float('inf')
    
    configs = _generate_lattice_configs()
    
    for v0 in configs:
        if time.time() > _DEADLINE - 10:
            break
            
        try:
            res = minimize(_objective, v0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-8})
            
            if res.success or res.fun < best_side:
                v_opt = res.x
                hex_vars = v_opt[:3*N]
                outer_vars = v_opt[3*N:]
                
                positions = np.zeros((N, 2))
                angles = np.zeros(N)
                for i in range(N):
                    positions[i, 0] = hex_vars[i*3]
                    positions[i, 1] = hex_vars[i*3+1]
                    angles[i] = hex_vars[i*3+2]
                    
                cx = outer_vars[0]
                cy = outer_vars[1]
                theta = outer_vars[2]
                
                side = _mhe_side_and_params(positions, angles, np.array([cx, cy]), theta)
                
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.linalg.norm(positions[i] - positions[j])
                        if d < 2.0 - 1e-5:
                            valid = False
                            break
                    if not valid: break
                        
                if valid and side < best_side:
                    best_side = side
                    inner = [[positions[i, 0], positions[i, 1], np.degrees(angles[i])] for i in range(N)]
                    best_result = (inner, (cx, cy), side, np.degrees(theta))
                    
        except Exception as e:
            pass
            
    if best_result is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 50.0, 0.0
        
    return best_result
