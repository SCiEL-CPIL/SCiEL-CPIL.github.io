import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # --- Geometry Helpers ---
    
    def hex_vertices(cx, cy, side, angle_rad):
        """Returns 6x2 array of vertices for a regular hexagon."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        """Returns 6x2 array of outward normals for a CCW polygon."""
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / (norms[:, np.newaxis] + 1e-15)
        return normals

    def sat_overlap(v1, v2):
        """
        Returns min overlap depth (negative if disjoint) using SAT.
        """
        normals1 = get_normals(v1)
        normals2 = get_normals(v2)
        normals = np.vstack([normals1, normals2])
        
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        min_overlap = np.min(overlaps)
        
        return min_overlap

    def compute_min_side(hexes):
        """
        Calculate the minimum side length of a regular hexagon (centered at origin, fixed orientation)
        that contains all given hexagons.
        """
        # The outer hexagon has fixed orientation (angle 0).
        # Its supporting lines are defined by the 6 outward normals of a hexagon with side 1, angle 0.
        # Normal angles: 30, 90, 150, 210, 270, 330 degrees.
        normal_angles = np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for h in hexes:
            proj = h @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        # For a regular hexagon of side s, the distance from center to side is s * sqrt(3) / 2.
        # So max_proj = s * sqrt(3) / 2 => s = max_proj * 2 / sqrt(3)
        s = max_proj * 2.0 / np.sqrt(3.0)
        return s

    def compute_objective(params):
        """
        params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12]
        Returns min_side + penalty for overlaps.
        """
        if time.time() > deadline:
            return 1e10
            
        # Reshape params to (12, 3)
        hex_params = params.reshape(12, 3)
        
        hexes = []
        for i in range(12):
            x, y, a = hex_params[i]
            hexes.append(hex_vertices(x, y, 1.0, a))
        
        # Calculate minimal outer side
        s_min = compute_min_side(hexes)
        
        penalty = 0.0
        
        # Check overlaps
        for i in range(12):
            for j in range(i + 1, 12):
                # Quick bounding box check
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += overlap * 10.0
                    
        return s_min + penalty

    # --- Optimization ---
    
    # Bounds for each hexagon: x, y in [-5, 5], angle in [0, pi/3]
    # Angles can be restricted to [0, pi/3] due to hexagon symmetry (60 deg periodicity)
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (0.0, np.pi/3)] * 12
    
    best_s = np.inf
    best_params = None
    
    # Use Differential Evolution for global search
    # We limit maxiter to ensure we finish in time
    try:
        res = differential_evolution(
            compute_objective,
            bounds,
            maxiter=50, # Reduced iterations to save time for local polish
            popsize=10,
            tol=1e-4,
            seed=42,
            polish=False # We will do our own polish
        )
        
        if res.fun < best_s:
            best_s = res.fun
            best_params = res.x
            
    except Exception as e:
        pass
    
    # Local Refinement: Try a few local minimizations from good starts
    # We'll try the best DE result and a few random valid-ish starts
    
    candidates = []
    if best_params is not None:
        candidates.append(best_params)
        
    # Add a known good symmetric start as a candidate
    # Simple 3-4-3-2 packing approximation or just a grid
    # Let's try a few random starts near the center
    np.random.seed(42)
    for _ in range(5):
        # Random positions in a hexagonal grid pattern
        p = np.zeros(36)
        # Generate 12 random positions
        for i in range(12):
            p[i*3] = np.random.uniform(-2, 2)
            p[i*3+1] = np.random.uniform(-2, 2)
            p[i*3+2] = np.random.uniform(0, np.pi/3)
        candidates.append(p)
        
    for p in candidates:
        if time.time() > deadline - 10:
            break
            
        try:
            res_local = minimize(
                compute_objective,
                p,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-6}
            )
            if res_local.fun < best_s:
                best_s = res_local.fun
                best_params = res_local.x
        except Exception:
            pass

    # Construct the final answer from best_params
    if best_params is None:
        # Fallback: Simple grid
        best_params = np.zeros(36)
        # Place them in a 3x4 grid roughly
        coords = []
        for i in range(3):
            for j in range(4):
                x = (j - 1.5) * 1.8
                y = (i - 1) * 1.6
                coords.append([x, y, 0.0])
        # Add 2 more? Just use the first 12 from a larger grid
        for i in range(5):
            for j in range(5):
                if len(coords) >= 12: break
                x = (j - 2) * 1.8
                y = (i - 2) * 1.6
                if not any(np.allclose(c[:2], [x,y]) for c in coords):
                    coords.append([x, y, 0.0])
        best_params = np.array(coords[:12]).flatten()
        
    x_coords = best_params[0::3]
    y_coords = best_params[1::3]
    a_coords = best_params[2::3]
    
    inner = []
    for i in range(12):
        inner.append([x_coords[i], y_coords[i], np.degrees(a_coords[i])])
        
    # Calculate final outer side length
    hexes_final = [hex_vertices(inner[i][0], inner[i][1], 1.0, np.radians(inner[i][2])) for i in range(12)]
    s_final = compute_min_side(hexes_final)
    
    outer_center = (0.0, 0.0)
    outer_side = s_final
    outer_angle_deg = 0.0
    
    return inner, outer_center, outer_side, outer_angle_deg

if __name__ == '__main__':
    inner, center, side, angle = construct()
    print(f"Side: {side}")
