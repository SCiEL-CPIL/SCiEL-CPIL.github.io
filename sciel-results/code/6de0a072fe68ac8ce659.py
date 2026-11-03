"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, angle_deg):
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    base = np.array([[1,0],[0.5,np.sqrt(3)/2],[-0.5,np.sqrt(3)/2],[-1,0],[-0.5,-np.sqrt(3)/2],[0.5,-np.sqrt(3)/2]])
    R = np.array([[cos_a,-sin_a],[sin_a,cos_a]])
    return cx + base @ R.T, cy + base @ R.T  # This is wrong, let me fix

def _hex_vertices(cx, cy, angle_deg):
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    base = np.array([[1,0],[0.5,np.sqrt(3)/2],[-0.5,np.sqrt(3)/2],[-1,0],[-0.5,-np.sqrt(3)/2],[0.5,-np.sqrt(3)/2]])
    verts = base @ R.T
    return np.column_stack([cx + verts[:,0], cy + verts[:,1]])

def _max_support(verts_all, center, angle_deg):
    a = np.radians(angle_deg)
    normals = np.array([[np.cos(a + np.pi/6.0 + k*np.pi/3.0), np.sin(a + np.pi/6.0 + k*np.pi/3.0)] for k in range(6)])
    rel = verts_all - np.array(center)
    projections = rel @ normals.T
    max_projs = np.max(projections, axis=0)
    min_projs = np.min(projections, axis=0)
    required_apothem = np.max(np.maximum(max_projs, -min_projs))
    return 2.0 * required_apothem / np.sqrt(3.0)

def _compute_side(inner, center, angle_deg):
    all_verts = np.vstack([_hex_vertices(cx, cy, ang) for (cx, cy, ang) in inner])
    return _max_support(all_verts, center, angle_deg)

def _hex_overlap(v1, v2, tol=1e-9):
    def edge_normals(verts):
        n = len(verts)
        normals = []
        for i in range(n):
            dx = verts[(i+1)%n, 0] - verts[i, 0]
            dy = verts[(i+1)%n, 1] - verts[i, 1]
            nx, ny = dy, -dx
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                normals.append([nx/norm, ny/norm])
        return np.array(normals)
    
    n1 = edge_normals(v1)
    n2 = edge_normals(v2)
    all_normals = np.vstack([n1, n2])
    
    for normal in all_normals:
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
            return False
    return True

def _check_overlaps(inner):
    verts = [_hex_vertices(cx, cy, ang) for (cx, cy, ang) in inner]
    for i in range(N):
        for j in range(i+1, N):
            if _hex_overlap(verts[i], verts[j]):
                return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial: hexagonal lattice
    d = np.sqrt(3.0)
    positions = []
    # 4x3 grid with hex offset
    for row in range(3):
        for col in range(4):
            x = col * d + (row % 2) * d/2.0
            y = row * d * np.sqrt(3.0)/2.0
            positions.append((x, y))
    
    inner = [[x, y, 0.0] for x, y in positions]
    center = (0.0, 0.0)
    angle_deg = 0.0
    
    current_side = _compute_side(inner, center, angle_deg)
    best_inner = [list(x) for x in inner]
    best_side = current_side
    best_center = center
    best_angle = angle_deg
    
    T = 0.3
    T_min = 0.005
    cooling = 0.997
    
    np.random.seed(123)
    
    while time.time() < deadline and T > T_min:
        op = np.random.random()
        
        if op < 0.45:
            i = np.random.randint(N)
            dx = np.random.normal(0, 0.2)
            dy = np.random.normal(0, 0.2)
            old_x, old_y = inner[i][0], inner[i][1]
            inner[i][0] += dx
            inner[i][1] += dy
            revert = lambda: (inner[i].__setitem__(0, old_x), inner[i].__setitem__(1, old_y))
        elif op < 0.75:
            i = np.random.randint(N)
            da = np.random.normal(0, 10.0)
            old_a = inner[i][2]
            inner[i][2] = (old_a + da) % 360.0
            revert = lambda: inner[i].__setitem__(2, old_a)
        elif op < 0.9:
            dx = np.random.normal(0, 0.15)
            dy = np.random.normal(0, 0.15)
            old_c = center
            center = (center[0]+dx, center[1]+dy)
            revert = lambda: None  # Can't easily revert tuple
            # Store old and restore manually
            revert_center = old_c
        else:
            da = np.random.normal(0, 3.0)
            old_angle = angle_deg
            angle_deg = (old_angle + da) % 360.0
            revert_angle = old_angle
        
        new_side = _compute_side(inner, center, angle_deg)
        
        if _check_overlaps(inner):
            new_side += 5.0
        
        delta = new_side - current_side
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_side = new_side
            if new_side < best_side:
                best_side = new_side
                best_inner = [list(x) for x in inner]
                best_center = center
                best_angle = angle_deg
        else:
            if op < 0.45:
                inner[i][0] = old_x
                inner[i][1] = old_y
            elif op < 0.75:
                inner[i][2] = old_a
            elif op < 0.9:
                center = revert_center
            else:
                angle_deg = revert_angle
        
        T *= cooling
    
    return best_inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
