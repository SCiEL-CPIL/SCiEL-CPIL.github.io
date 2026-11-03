import numpy as np
from scipy.optimize import minimize

N = 12

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    dx = px - cx
    dy = py - cy
    r = side * np.sqrt(3) / 2
    angle_rad = np.radians(angle_deg)
    for i in range(6):
        theta = angle_rad + i * np.pi / 3 + np.pi / 6
        nx = np.cos(theta)
        ny = np.sin(theta)
        if dx * nx + dy * ny > r + 1e-9:
            return False
    return True

def hexagons_overlap(v1, v2):
    """Check if two hexagons overlap using SAT."""
    normals = []
    for verts in [v1, v2]:
        for i in range(6):
            j = (i + 1) % 6
            edge = verts[j] - verts[i]
            nx = -edge[1]
            ny = edge[0]
            norm = np.sqrt(nx**2 + ny**2)
            if norm > 1e-12:
                normals.append([nx / norm, ny / norm])
    for normal in normals:
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        if np.max(proj1) < np.min(proj2) - 1e-9 or np.max(proj2) < np.min(proj1) - 1e-9:
            return False
    return True

# EVOLVE-BLOCK-START
def construct():
    outer_cx, outer_cy = 0.0, 0.0
    outer_angle = 0.0
    
    # Initial guess: compact honeycomb-like arrangement
    positions = []
    angles = []
    for row in range(3):
        for col in range(4):
            x = (col - 1.5) * 2 * np.sqrt(3) + (row % 2) * np.sqrt(3)
            y = (row - 1) * 1.5
            positions.append([x, y])
            angles.append(0.0)
    
    positions = np.array(positions)
    angles = np.array(angles)
    
    x0 = np.concatenate([positions.flatten(), angles, [3.0]])
    
    def objective(x):
        pos = x[:24].reshape(12, 2)
        ang = x[24:36]
        outer_side = x[36]
        
        if outer_side < 1.0:
            return 1000.0 + (1.0 - outer_side) ** 2
        
        verts_list = [hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i]) for i in range(12)]
        
        penalty = 0.0
        for i in range(12):
            for v in verts_list[i]:
                if not point_in_hexagon(v[0], v[1], outer_cx, outer_cy, outer_side, outer_angle):
                    dx = v[0] - outer_cx
                    dy = v[1] - outer_cy
                    r = outer_side * np.sqrt(3) / 2
                    angle_rad = np.radians(outer_angle)
                    for k in range(6):
                        theta = angle_rad + k * np.pi / 3 + np.pi / 6
                        nx = np.cos(theta)
                        ny = np.sin(theta)
                        dist = dx * nx + dy * ny
                        if dist > r:
                            penalty += (dist - r) ** 2
        
        for i in range(12):
            for j in range(i + 1, 12):
                if hexagons_overlap(verts_list[i], verts_list[j]):
                    penalty += 100.0
        
        return outer_side + penalty
    
    bounds = [(-10, 10)] * 24 + [(0, 360)] * 12 + [(1.0, 20.0)]
    
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
    
    x_opt = result.x
    pos_opt = x_opt[:24].reshape(12, 2)
    ang_opt = x_opt[24:36]
    outer_side_opt = x_opt[36]
    
    inner = [[float(pos_opt[i, 0]), float(pos_opt[i, 1]), float(ang_opt[i])] for i in range(12)]
    
    return inner, (outer_cx, outer_cy), float(outer_side_opt), float(outer_angle)
# EVOLVE-BLOCK-END
