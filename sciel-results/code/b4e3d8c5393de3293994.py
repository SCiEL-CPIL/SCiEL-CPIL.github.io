import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    candidates = []

    # Heuristic: Compact shapes
    # 1. Flower (1+6+4)
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    # 2. 3x3 block minus 2 corners (7+4)
    # (0,0) to (2,2) is 9 hexagons.
    # Add 2 more.
    axial_2 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (3, 0), (-1, 1)
    ]
    candidates.append(axial_2)

    # 3. Linear-ish but bent
    axial_3 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2)
    ]
    candidates.append(axial_3)

    # 4. More compact: 1+6+4 but different 4
    axial_4 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 2), (2, -2)
    ]
    candidates.append(axial_4)

    # 5. 2x2x3 block?
    axial_5 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2)
    ]
    candidates.append(axial_5)

    # 6. Diamond shape
    axial_6 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, 1), (1, -1), (-1, -1),
        (2, 0), (-2, 0)
    ]
    candidates.append(axial_6)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def get_hex_vertices(cx, cy, angle_deg):
        """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation angle_deg."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3.0
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return np.array(verts)

    def calc_S_continuous(params):
        """
        params: [cx1, cy1, a1, cx2, cy2, a2, ..., cx11, cy11, a11, outer_angle_deg]
        Returns: Side length S of the enclosing hexagon.
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        
        n_hex = 11
        outer_angle_deg = params[-1]
        outer_angle_rad = math.radians(outer_angle_deg)
        
        # Normals of the outer hexagon
        # The sides are perpendicular to the normals.
        # A regular hexagon with rotation phi has normals at phi + pi/6 + k*pi/3.
        # Wait, the standard orientation (flat top) has normals at 30, 90, 150, 210, 270, 330?
        # Let's stick to the definition: vertices at 0, 60, 120...
        # If vertices are at 0, 60, 120, 180, 240, 300, the normals (outward) are at 30, 90, 150, 210, 270, 330.
        # So normals = outer_angle_rad + pi/6 + k*pi/3.
        
        max_proj = 0.0
        
        for i in range(n_hex):
            cx = params[3*i]
            cy = params[3*i + 1]
            a_deg = params[3*i + 2]
            
            verts = get_hex_vertices(cx, cy, a_deg)
            
            for k in range(6):
                normal_angle = outer_angle_rad + math.pi / 6.0 + k * math.pi / 3.0
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                
                # Project all vertices onto this normal
                projs = verts[:, 0] * nx + verts[:, 1] * ny
                max_p = np.max(projs)
                min_p = np.min(projs)
                
                # The constraint is that the projection must fit within [-S*apothem, S*apothem]
                # So S*apothem >= max(|max_p|, |min_p|)
                current_max = max(abs(max_p), abs(min_p))
                if current_max > max_proj:
                    max_proj = current_max
        
        S = max_proj / APOTHEM
        return S

    def optimize_shape_continuous(pts):
        """
        Optimize positions and rotations of hexagons starting from initial pts.
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None, None, None
        
        n_hex = len(pts)
        # Initial params: centers are pts, angles are 0, outer angle is 0
        x0 = []
        for x, y in pts:
            x0.append(x)
            x0.append(y)
            x0.append(0.0)
        x0.append(0.0) # outer angle
        
        x0 = np.array(x0)
        
        best_S = float('inf')
        best_params = x0
        
        # Use Nelder-Mead for robustness against non-differentiability
        try:
            res = minimize(calc_S_continuous, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass
        
        # Try Powell as well, it might handle the box-like constraints better
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res2 = minimize(calc_S_continuous, best_params, method='Powell',
                                options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
                if res2.fun < best_S:
                    best_S = res2.fun
                    best_params = res2.x
        except Exception:
            pass
            
        # Extract results
        centers = []
        angles = []
        for i in range(n_hex):
            centers.append((best_params[3*i], best_params[3*i+1]))
            angles.append(best_params[3*i+2])
        outer_angle = best_params[-1]
        
        return best_S, centers, angles, outer_angle, best_params

    best_global_S = float('inf')
    best_global_centers = None
    best_global_angles = None
    best_global_outer_angle = 0.0

    # Sort by heuristic
    def axial_heuristic(axial):
        rows = set(r for _, r in axial)
        cols = set(q for q, _ in axial)
        return (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
    
    sorted_candidates = sorted(candidates, key=axial_heuristic)
    
    for axial in sorted_candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, centers, angles, outer_angle, _ = optimize_shape_continuous(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_centers = centers
            best_global_angles = angles
            best_global_outer_angle = outer_angle

    if best_global_centers is None or best_global_S == float('inf'):
        # Fallback
        best_global_centers = [tuple(p) for p in axial_to_cartesian(candidates[0])]
        best_global_angles = [0.0] * 11
        best_global_outer_angle = 0.0
        # Calculate S for fallback
        params = []
        for c, a in zip(best_global_centers, best_global_angles):
            params.extend([c[0], c[1], a])
        params.append(best_global_outer_angle)
        best_global_S = calc_S_continuous(params)
    
    # Final verification and scaling
    # The optimizer might find a configuration that is slightly invalid due to numerical errors
    # We scale the inner hexagons by a tiny factor to ensure they fit inside.
    # But we also need to return the correct side length.
    # The score is the side length.
    
    inner = []
    for i in range(11):
        x, y = best_global_centers[i]
        a = best_global_angles[i]
        inner.append([float(x), float(y), float(a)])
        
    # The center of the outer hexagon is (0,0) in our optimization frame?
    # No, the optimization minimized the max projection relative to the origin.
    # This implies the outer hexagon is centered at the origin.
    # So center is (0,0).
    center = [0.0, 0.0]
    side = float(best_global_S)
    rotation = float(best_global_outer_angle)
    
    return inner, center, side, rotation
