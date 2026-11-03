import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
D = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 10.0

    # Vertices of a unit hexagon (flat-top, side 1) centered at origin
    # Angles: 0, 60, 120, 180, 240, 300
    # Wait, flat top usually means vertical sides?
    # Standard math: Point at (1,0) is angle 0.
    # If we want "flat top" (horizontal top edge), vertices are at 30, 90, 150...
    # Let's use vertices at angles 0, 60, ... (pointy right).
    # It doesn't matter as long as we are consistent.
    hex_verts_base = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] 
        for i in range(6)
    ])

    def get_vertices(center, angle):
        # Rotate base vertices
        c = math.cos(angle)
        s = math.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return center + hex_verts_base.T @ R # (2, 6) -> (2, 6) is wrong, base is (6,2)
        # base is (6,2). R is (2,2).
        # R @ base.T -> (2,6).
        # Let's do: base @ R.T -> (6,2)
        
    def get_vertices_correct(center, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        R = np.array([[c, -s], [s, c]])
        # hex_verts_base is (6, 2)
        # rotated = hex_verts_base @ R.T
        rotated = hex_verts_base @ R.T
        return rotated + center

    def check_overlap_depth(verts1, verts2):
        # verts1, verts2 are (6, 2)
        # Get axes
        axes = []
        for i in range(6):
            v1 = verts1[i]
            v2 = verts1[(i+1)%6]
            edge = v2 - v1
            normal = np.array([-edge[1], edge[0]])
            axes.append(normal)
        for i in range(6):
            v1 = verts2[i]
            v2 = verts2[(i+1)%6]
            edge = v2 - v1
            normal = np.array([-edge[1], edge[0]])
            axes.append(normal)
            
        min_pen = float('inf')
        found_overlap = False
        for axis in axes:
            norm = np.linalg.norm(axis)
            if norm == 0: continue
            axis = axis / norm
            proj1 = np.dot(verts1, axis)
            proj2 = np.dot(verts2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 or max2 < min1:
                return 0.0 # Separated
            
            found_overlap = True
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_pen:
                min_pen = overlap
                
        if found_overlap:
            return min_pen
        return 0.0

    def objective(x):
        # x: [x1, y1, a1, x2, y2, a2, ..., x11, y11, a11, outer_angle]
        # 11 * 3 + 1 = 34 variables
        centers = x[:33].reshape(11, 3)[:, :2]
        angles = x[:33].reshape(11, 3)[:, 2]
        outer_angle = x[33]
        
        # Calculate S
        # Normals of outer hexagon
        # Outer hexagon is regular.
        # Let's define outer hexagon normals at angles outer_angle + k * 60 deg.
        # The distance from center to side is h = S * sqrt(3)/2.
        # So S = 2 * h / sqrt(3).
        
        # We need max projection of any inner vertex onto any outer normal.
        # Let's get all vertices
        all_verts = []
        for i in range(11):
            v = get_vertices_correct(centers[i], angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts) # (66, 2)
        
        # Normals
        normals = []
        for k in range(6):
            a = outer_angle + k * math.pi / 3
            normals.append([math.cos(a), math.sin(a)])
        normals = np.array(normals) # (6, 2)
        
        # Projections: (66, 2) @ (2, 6) -> (66, 6)
        projs = all_verts @ normals.T
        max_abs_proj = np.max(np.abs(projs))
        S = (2.0 / math.sqrt(3)) * max_abs_proj
        
        # Overlap Penalty
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                v1 = get_vertices_correct(centers[i], angles[i])
                v2 = get_vertices_correct(centers[j], angles[j])
                depth = check_overlap_depth(v1, v2)
                if depth > 0:
                    penalty += depth * depth
        
        return S + 100.0 * penalty

    # Initialization
    # 3x3 grid of hexagons
    # Axial to Cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    # Let's place 9 hexagons in a 3x3 block
    # q in [0,1,2], r in [0,1,2]
    # Center of block
    init_pts = []
    for q in range(3):
        for r in range(3):
            x = D * (q + r / 2.0)
            y = 1.5 * r
            init_pts.append((x, y, 0.0))
            
    # Add 2 more at (0,0)
    init_pts.append((0.0, 0.0, 0.0))
    init_pts.append((0.0, 0.0, 0.0))
    
    # Center the configuration
    xs = [p[0] for p in init_pts]
    ys = [p[1] for p in init_pts]
    cx = np.mean(xs)
    cy = np.mean(ys)
    init_pts = [(x - cx, y - cy, a) for x, y, a in init_pts]
    
    x0 = np.array([item for p in init_pts for item in p] + [0.0])
    
    best_S = float('inf')
    best_x = x0.copy()
    
    # Local Search with multiple restarts
    num_restarts = 10
    for k in range(num_restarts):
        if time.time() > deadline - safety_margin:
            break
            
        # Perturb initial guess
        if k == 0:
            x_start = x0
        else:
            x_start = x0 + np.random.normal(0, 0.5, 34)
            
        try:
            # Use Powell for efficiency
            res = minimize(objective, x_start, method='Powell', options={'maxiter': 200, 'xtol': 1e-6, 'ftol': 1e-6})
            if res.fun < best_S:
                best_S = res.fun
                best_x = res.x
        except Exception:
            pass

    # Extract best solution
    centers = best_x[:33].reshape(11, 3)[:, :2]
    angles = best_x[:33].reshape(11, 3)[:, 2]
    outer_angle = best_x[33]
    
    # Final check and cleanup
    # If best_S is still huge, fallback
    if best_S > 10:
         # Fallback to seed-like behavior or a simple grid
         inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
         return inner, (0.0, 0.0), 40.0, 0.0

    inner_rows = []
    for i in range(11):
        inner_rows.append([float(centers[i][0]), float(centers[i][1]), float(math.degrees(angles[i]))])
        
    center = (0.0, 0.0)
    side = float(best_S)
    angle_deg = float(math.degrees(outer_angle))
    
    return inner_rows, center, side, angle_deg
