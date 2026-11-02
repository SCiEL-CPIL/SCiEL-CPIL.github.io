import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, s, angle_deg):
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + np.pi / 3 * i
        verts.append([cx + s * np.cos(theta), cy + s * np.sin(theta)])
    return np.array(verts)

def _point_in_hex(px, py, cx, cy, s, angle_deg):
    a = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    rx = dx * np.cos(-a) - dy * np.sin(-a)
    ry = dx * np.sin(-a) + dy * np.cos(-a)
    for k in range(6):
        theta_edge = (k + 0.5) * np.pi / 3
        nx = np.cos(theta_edge)
        ny = np.sin(theta_edge)
        if rx * nx + ry * ny > s * np.sqrt(3) / 2 + 1e-9:
            return False
    return True

def _max_containing_hex_side(positions, angles, outer_angle_deg):
    s_inner = 1.0
    outer_a = np.radians(outer_angle_deg)
    max_required = 0.0
    for i in range(len(positions)):
        cx, cy = positions[i]
        verts = _hex_vertices(cx, cy, s_inner, angles[i])
        for k in range(6):
            theta_edge = (k + 0.5) * np.pi / 3 + outer_a
            nx = np.cos(theta_edge)
            ny = np.sin(theta_edge)
            proj = verts[:, 0] * nx + verts[:, 1] * ny
            max_proj = np.max(proj)
            required_S = max_proj * 2.0 / np.sqrt(3)
            if required_S > max_required:
                max_required = required_S
    return max_required

def _check_disjoint(positions, angles, eps=1e-6):
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            vi = _hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            vj = _hex_vertices(positions[j][0], positions[j][1], 1.0, angles[j])
            for v in vi:
                if _point_in_hex(v[0], v[1], positions[j][0], positions[j][1], 1.0, angles[j]):
                    return False
            for v in vj:
                if _point_in_hex(v[0], v[1], positions[i][0], positions[i][1], 1.0, angles[i]):
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = np.sqrt(3)
    
    # Hexagonal lattice initial configuration
    positions = []
    angles = []
    
    # Center
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # Ring 1: 6 hexagons
    for k in range(6):
        theta = k * np.pi / 3
        x = sqrt3 * np.cos(theta)
        y = sqrt3 * np.sin(theta)
        positions.append([x, y])
        angles.append(0.0)
    
    # Ring 2: 4 more hexagons
    for k in [0, 1, 4, 5]:
        theta = k * np.pi / 3
        x = 2 * sqrt3 * np.cos(theta)
        y = 2 * sqrt3 * np.sin(theta)
        positions.append([x, y])
        angles.append(0.0)
    
    positions = np.array(positions)
    angles_arr = np.array(angles)
    
    def objective(params):
        pos = params[:2 * N].reshape(N, 2)
        outer_angle = params[2 * N]
        S = _max_containing_hex_side(pos.tolist(), [0.0] * N, outer_angle)
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < sqrt3 - 1e-3:
                    penalty += (sqrt3 - dist) ** 2
        return S + penalty
    
    x0 = np.concatenate([positions.flatten(), [0.0]])
    
    best_S = _max_containing_hex_side(positions.tolist(), angles_arr.tolist(), 0.0)
    best_params = x0.copy()
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B',
                         options={'maxiter': 300, 'ftol': 1e-7, 'gtol': 1e-7})
        if time.time() < deadline:
            S_candidate = result.fun
            if S_candidate < best_S:
                best_params = result.x
                best_S = S_candidate
    except Exception:
        pass
    
    best_pos = best_params[:2 * N].reshape(N, 2)
    best_outer_angle = best_params[2 * N]
    
    if not _check_disjoint(best_pos.tolist(), [0.0] * N):
        best_pos = positions
        best_outer_angle = 0.0
        best_S = _max_containing_hex_side(positions.tolist(), angles_arr.tolist(), 0.0)
    
    final_S = _max_containing_hex_side(best_pos.tolist(), [0.0] * N, best_outer_angle)
    final_S += 1e-5
    
    inner = [[float(x), float(y), 0.0] for x, y in best_pos]
    return inner, (0.0, 0.0), float(final_S), float(np.degrees(best_outer_angle))
# EVOLVE-BLOCK-END
