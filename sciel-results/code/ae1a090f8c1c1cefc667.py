import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def get_hex_vertices(center, side, angle):
        cx, cy = center
        verts = []
        for i in range(6):
            theta = angle + i * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return np.array(verts)

    def get_hex_normals(angle):
        normals = []
        for i in range(6):
            theta = angle + math.pi / 6.0 + i * math.pi / 3.0
            normals.append((math.cos(theta), math.sin(theta)))
        return np.array(normals)

    def calc_overlap_depth(verts_a, normals_a, verts_b, normals_b):
        # Compute min overlap over all axes (normals of A and B)
        min_overlap = float('inf')
        # Axes from A
        for nx, ny in normals_a:
            proj_a = verts_a[:, 0] * nx + verts_a[:, 1] * ny
            proj_b = verts_b[:, 0] * nx + verts_b[:, 1] * ny
            min_a, max_a = np.min(proj_a), np.max(proj_a)
            min_b, max_b = np.min(proj_b), np.max(proj_b)
            overlap = min(max_a, max_b) - max(min_a, min_b)
            if overlap < min_overlap:
                min_overlap = overlap
        # Axes from B
        for nx, ny in normals_b:
            proj_a = verts_a[:, 0] * nx + verts_a[:, 1] * ny
            proj_b = verts_b[:, 0] * nx + verts_b[:, 1] * ny
            min_a, max_a = np.min(proj_a), np.max(proj_a)
            min_b, max_b = np.min(proj_b), np.max(proj_b)
            overlap = min(max_a, max_b) - max(min_a, min_b)
            if overlap < min_overlap:
                min_overlap = overlap
        return min_overlap

    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10
            
        S = params[0]
        cx, cy, phi = params[1], params[2], params[3]
        
        # Outer hexagon
        outer_verts = get_hex_vertices((cx, cy), S, phi)
        outer_normals = get_hex_normals(phi)
        
        penalty = 0.0
        
        # Inner hexagons
        inner_verts_list = []
        inner_normals_list = []
        
        for i in range(N):
            ix, iy, alpha = params[4 + 3*i], params[5 + 3*i], params[6 + 3*i]
            verts = get_hex_vertices((ix, iy), 1.0, alpha)
            normals = get_hex_normals(alpha)
            inner_verts_list.append(verts)
            inner_normals_list.append(normals)
            
            # Containment penalty
            for vx, vy in verts:
                for nx, ny in outer_normals:
                    dist = (vx - cx) * nx + (vy - cy) * ny
                    # The hexagon is defined by |dist| <= S * APOTHEM
                    # Wait, the normal points to the side. The distance from center to side is S * APOTHEM.
                    # So the condition is |dist| <= S * APOTHEM.
                    # If |dist| > S * APOTHEM, it's outside.
                    if abs(dist) > S * APOTHEM:
                        penalty += (abs(dist) - S * APOTHEM) ** 2
        
        # Pairwise overlap penalty
        for i in range(N):
            for j in range(i + 1, N):
                depth = calc_overlap_depth(inner_verts_list[i], inner_normals_list[i], 
                                           inner_verts_list[j], inner_normals_list[j])
                if depth > 0:
                    penalty += depth ** 2
                    
        # We want to minimize S, but penalty must be 0.
        # Use a large weight for penalty to ensure feasibility.
        return S + 1000.0 * penalty

    # Initial guesses
    best_result = None
    best_val = float('inf')
    
    # Generate a few random starting points
    np.random.seed(42)
    
    for restart in range(10):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Initial S
        S0 = 4.0 + np.random.rand()
        # Outer center
        cx0, cy0 = 0.0, 0.0
        phi0 = 0.0
        
        params = [S0, cx0, cy0, phi0]
        
        # Inner hexagons: place in a grid
        # Approximate positions
        positions = []
        # Center
        positions.append((0, 0, 0))
        # Ring 1
        for k in range(6):
            angle = k * math.pi / 3.0
            x = 2.0 * math.cos(angle)
            y = 2.0 * math.sin(angle)
            positions.append((x, y, 0))
        # Ring 2 (partial)
        for k in range(4):
            angle = k * math.pi / 2.0 + math.pi/4
            x = 3.5 * math.cos(angle)
            y = 3.5 * math.sin(angle)
            positions.append((x, y, 0))
            
        # Add random jitter
        for i in range(N):
            x, y, a = positions[i]
            x += np.random.randn() * 0.1
            y += np.random.randn() * 0.1
            a += np.random.randn() * 0.1
            params.extend([x, y, a])
            
        res = minimize(objective, params, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.fun < best_val:
            best_val = res.fun
            best_result = res.x
            
    if best_result is None:
        # Fallback
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
        
    S = best_result[0]
    cx, cy, phi = best_result[1], best_result[2], best_result[3]
    
    inner = []
    for i in range(N):
        x, y, a = best_result[4 + 3*i], best_result[5 + 3*i], best_result[6 + 3*i]
        inner.append([float(x), float(y), float(math.degrees(a))])
        
    center = [float(cx), float(cy)]
    side = float(S)
    rotation = float(math.degrees(phi))
    
    return inner, center, side, rotation
