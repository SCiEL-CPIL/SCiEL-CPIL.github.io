# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _all_inner_vertices_vec(positions):
    """(12,2) positions -> (72,2) all vertices (angle 0)."""
    a = np.radians(60 * np.arange(6))
    base = np.column_stack([np.cos(a), np.sin(a)])  # (6,2)
    return positions[:, None, :] + base[None, :, :]  # (12,6,2)

def _check_separation_vec(positions):
    """Vectorized SAT check for all pairs. True if all separated."""
    n = len(positions)
    verts = _all_inner_vertices_vec(positions)  # (12,6,2)
    # Edge normals for each hex: (12,6,2)
    edges = np.roll(verts, -1, axis=1) - verts  # (12,6,2)
    normals = np.column_stack([np.zeros(12*6), -edges[:, :, 0], edges[:, :, 1]]).reshape(12, 6, 2)
    # Actually recompute normals properly
    normals = np.empty((12, 6, 2))
    for i in range(12):
        for k in range(6):
            e = edges[i, k]
            normals[i, k] = [-e[1], e[0]]
    
    for i in range(n):
        vi = verts[i]  # (6,2)
        ni = normals[i]  # (6,2)
        for j in range(i+1, n):
            vj = verts[j]
            nj = normals[j]
            found = False
            # Check normals of i
            for k in range(6):
                nm = ni[k]
                pi = vi @ nm
                pj = vj @ nm
                if pi.max() < pj.min() - 1e-6 or pj.max() < pi.min() - 1e-6:
                    found = True
                    break
            if found:
                continue
            for k in range(6):
                nm = nj[k]
                pi = vi @ nm
                pj = vj @ nm
                if pi.max() < pj.min() - 1e-6 or pj.max() < pi.min() - 1e-6:
                    found = True
                    break
            if not found:
                return False
    return True

def _min_containing_hex_side(positions, ocx, ocy, angle_deg):
    all_verts = _all_inner_vertices_vec(positions).reshape(-1, 2)
    a_rad = np.radians(angle_deg)
    normal_angles = a_rad + np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    shifted = all_verts - np.array([ocx, ocy])
    dots = shifted @ normals.T
    max_dot = np.max(dots)
    s = (2.0 / np.sqrt(3.0)) * max_dot
    return s

def _objective(params):
    positions = params[:24].reshape(12, 2)
    ocx, ocy = params[24], params[25]
    angle_deg = params[26]
    if not _check_separation_vec(positions):
        return 1000.0
    s = _min_containing_hex_side(positions, ocx, ocy, angle_deg)
    return s

def _lattice_positions():
    """Hex lattice: center + 6 ring + 5 of second ring (12 total)."""
    h = 2.5
    v = 2.1650635094610965  # sqrt(75)/2
    positions = np.array([
        [0.0, 0.0],
        [-h, 0.0], [h, 0.0],
        [-h/2, v], [h/2, v],
        [-h/2, -v], [h/2, -v],
        [-3*h/2, v], [3*h/2, v],
        [-3*h/2, -v], [3*h/2, -v],
        [0.0, 2*v],
    ])
    return positions

def _circular_positions(radius=2.0):
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580
    
    np.random.seed(42)
    
    best_s = float('inf')
    best_params = None
    
    inits = []
    base = _lattice_positions()
    inits.append(base)
    inits.append(_lattice_positions() * 1.02)
    inits.append(_lattice_positions() * 0.98)
    for r in [1.8, 2.2, 2.6]:
        inits.append(_circular_positions(r))
    for _ in range(8):
        inits.append(base + np.random.randn(12, 2) * 0.4)
    for _ in range(4):
        inits.append(np.random.randn(12, 2) * 3.0)
    
    for i, init_pos in enumerate(inits):
        if time.time() > deadline - 40:
            break
        for init_angle in [0, 30]:
            if time.time() > deadline - 40:
                break
            centroid = init_pos.mean(axis=0)
            init_params = np.zeros(27)
            init_params[:24] = init_pos.flatten()
            init_params[24] = centroid[0]
            init_params[25] = centroid[1]
            init_params[26] = init_angle
            try:
                res = minimize(
                    _objective, init_params, method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
                )
            except Exception:
                continue
            if res.fun < best_s and res.fun < 100:
                best_s = res.fun
                best_params = res.x.copy()
    
    if best_params is None or best_s > 100:
        best_pos = _lattice_positions()
        best_angle = 0
        centroid = best_pos.mean(axis=0)
        all_verts = _all_inner_vertices_vec(best_pos).reshape(-1, 2)
        a_rad = np.radians(best_angle)
        normal_angles = a_rad + np.radians(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        shifted = all_verts - centroid
        dots = shifted @ normals.T
        max_dot = np.max(dots)
        best_s = (2.0 / np.sqrt(3.0)) * max_dot
        best_params = np.zeros(27)
        best_params[:24] = best_pos.flatten()
        best_params[24] = centroid[0]
        best_params[25] = centroid[1]
        best_params[26] = best_angle
    
    final_positions = best_params[:24].reshape(12, 2)
    ocx, ocy = best_params[24], best_params[25]
    final_angle = best_params[26]
    
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = final_positions[:, 0] - ocx
    inner_hex_data[:, 1] = final_positions[:, 1] - ocy
    inner_hex_data[:, 2] = 0.0
    outer_hex_data = np.array([0.0, 0.0, final_angle])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
