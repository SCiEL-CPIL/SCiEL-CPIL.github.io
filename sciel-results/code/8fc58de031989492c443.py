import numpy as np
import math

N = 11

def construct():
    sqrt3 = math.sqrt(3)
    
    # Hexagonal lattice vectors for edge-sharing unit hexagons
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3/2, 1.5])
    
    # Generate and select 11 closest lattice points to origin
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            pt = i * v1 + j * v2
            dist2 = pt[0]**2 + pt[1]**2
            candidates.append((dist2, pt[0], pt[1]))
    
    candidates.sort()
    centers = [(c[1], c[2]) for c in candidates[:11]]
    
    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = np.array([[math.cos(angle_rad + math.radians(60*k)), 
                           math.sin(angle_rad + math.radians(60*k))] for k in range(6)])
        return verts + np.array([cx, cy])
    
    def min_enclosing_hex_side(points, outer_orient=0.0):
        """points: Nx2. Returns minimal side length of enclosing regular hexagon with given orientation."""
        normals = np.array([[math.cos(math.radians(outer_orient + 60*k)), 
                             math.sin(math.radians(outer_orient + 60*k))] for k in range(6)])
        
        # The optimal center is the one that minimizes max over normals of the "radius"
        # Let's use the fact that for a regular hexagon, the center should be near the centroid
        centroid = points.mean(axis=0)
        
        def side_for_center(C):
            max_val = 0
            for n in normals:
                proj = (points - C) @ n
                max_val = max(max_val, proj.max(), -proj.min())
            return 2.0 / sqrt3 * max_val
        
        # Local search from centroid
        C = centroid.copy()
        best = side_for_center(C)
        step = 0.1
        for _ in range(50):
            improved = False
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    new_C = C + np.array([dx, dy])
                    val = side_for_center(new_C)
                    if val < best:
                        C = new_C
                        best = val
                        improved = True
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break
        return best
    
    def total_side(rot_deg):
        all_pts = np.vstack([hex_vertices(cx, cy, rot_deg) for cx, cy in centers])
        return min_enclosing_hex_side(all_pts, 0.0)
    
    # Minimize over rotation [0, 60)
    samples = np.linspace(0, 60, 61)
    vals = [total_side(s) for s in samples]
    min_idx = int(np.argmin(vals))
    
    left = samples[max(0, min_idx - 1)]
    right = samples[min(60, min_idx + 1)]
    
    for _ in range(40):
        m1 = left + (right - left) / 3
        m2 = right - (right - left) / 3
        if total_side(m1) < total_side(m2):
            right = m2
        else:
            left = m1
    
    best_rot = (left + right) / 2
    best_side = total_side(best_rot)
    
    inner = [[cx, cy, best_rot] for cx, cy in centers]
    return inner, (0.0, 0.0), best_side, 0.0
