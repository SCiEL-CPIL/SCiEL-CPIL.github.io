import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 145.0  # Safety margin below 150s
APOTHEM_UNIT = math.sqrt(3) / 2.0

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _get_axes(verts):
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        # Normal vector
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _project(verts, axis):
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _sat_overlap_depth(verts1, verts2, axes1, axes2):
    """
    Returns the overlap depth if polygons overlap, 0 otherwise.
    Positive value indicates overlap.
    """
    min_overlap = float('inf')
    
    for axis in axes1:
        min1, max1 = _project(verts1, axis)
        min2, max2 = _project(verts2, axis)
        
        if max1 <= min2 or max2 <= min1:
            return 0.0 # Separated
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            
    for axis in axes2:
        min1, max1 = _project(verts1, axis)
        min2, max2 = _project(verts2, axis)
        
        if max1 <= min2 or max2 <= min1:
            return 0.0 # Separated
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            
    return min_overlap

def _objective(params):
    # Unpack params
    xs = params[0:N]
    ys = params[N:2*N]
    angles = params[2*N:3*N]
    
    verts_list = []
    axes_list = []
    
    for i in range(N):
        v = _hex_vertices(xs[i], ys[i], angles[i])
        verts_list.append(v)
        axes_list.append(_get_axes(v))
        
    # Calculate required outer side length
    # Outer hexagon normals (assuming outer is fixed at angle 0, center 0,0)
    # We can optimize outer angle later or fix it. 
    # Due to symmetry, fixing outer angle to 0 is WLOG for the *relative* configuration?
    # No, the inner hexagons can rotate. The outer hexagon's orientation matters relative to them.
    # However, we can fix the outer hexagon to angle 0 and center (0,0) and let the inner ones move.
    # This is valid because the problem asks for the minimum side length, and the outer hexagon's position/orientation
    # can be adjusted to best fit the inner cluster. 
    # Wait, if we fix the outer to (0,0,0), we are constraining the solution.
    # But the score is just the side length.
    # The evaluator allows the outer hexagon to have any center and angle.
    # So we should minimize the side length of the *minimum enclosing regular hexagon* of the inner set.
    
    # The minimum enclosing regular hexagon problem is non-trivial.
    # Approximation: The side length is determined by the max projection onto the 6 normals.
    # But the normals depend on the outer angle.
    # Let's assume the outer hexagon is centered at the centroid of the inner hexagons?
    # Or better: We can add the outer angle and center to the optimization variables?
    # That increases dimensionality.
    # Alternative: For a given configuration of inner hexagons, the minimum enclosing hexagon side length can be found by checking 6 orientations?
    # Actually, for a fixed set of points, the minimum enclosing regular hexagon is unique up to symmetry.
    # A simple bound: The side length is at least the max distance from the centroid to any vertex divided by the apothem ratio?
    
    # Let's stick to the inspiration's approach: 
    # The objective is likely just the max projection onto the 6 fixed normals of a reference hexagon, 
    # OR we optimize the outer angle as well.
    
    # Let's optimize the outer angle and center as well to be safe.
    # But that makes the objective non-smooth potentially? No, it's smooth.
    # However, adding 3 more variables (cx, cy, theta_outer) is fine.
    
    # For now, let's use a simplified objective that assumes the outer hexagon is centered at (0,0) and angle 0.
    # This might be suboptimal but is a good start.
    # Actually, the "flower" pattern is symmetric, so center (0,0) and angle 0 is likely optimal for the best packings.
    
    max_proj = 0.0
    for v in verts_list:
        for (vx, vy) in v:
            # Project onto 6 normals of unit hexagon (angle 0)
            # Normals are at 30, 90, 150, 210, 270, 330 degrees
            for k in range(6):
                theta_n = math.radians(30 + 60 * k)
                nx = math.cos(theta_n)
                ny = math.sin(theta_n)
                proj = vx * nx + vy * ny
                if proj > max_proj:
                    max_proj = proj
                    
    side_length = max_proj / APOTHEM_UNIT
    
    # Calculate overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(verts_list[i], verts_list[j], axes_list[i], axes_list[j])
            if depth > 0:
                penalty += depth * depth
                
    return side_length + 10.0 * penalty

def construct():
    t0 = time.time()
    
    # Initial configurations
    def make_flower():
        pts = []
        pts.append([0.0, 0.0, 0.0])
        r1 = 2 * math.sqrt(3)
        for i in range(6):
            angle = math.radians(60 * i)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            pts.append([x, y, 0.0])
        r2 = r1 * 1.732
        angles = [30, 90, 150, 210, 270]
        for deg in angles:
            rad = math.radians(deg)
            x = r2 * math.cos(rad)
            y = r2 * math.sin(rad)
            pts.append([x, y, 0.0])
        return np.array([p[0] for p in pts] + [p[1] for p in pts] + [p[2] for p in pts])

    def make_lattice():
        pts = []
        row_height = math.sqrt(3)
        col_width = 2.0
        for c in range(3):
            x = c * col_width - col_width
            y = -row_height
            pts.append([x, y, 0.0])
        for c in range(4):
            x = c * col_width - col_width - col_width/2
            y = 0.0
            pts.append([x, y, 0.0])
        for c in range(3):
            x = c * col_width - col_width
            y = row_height
            pts.append([x, y, 0.0])
        pts.append([-col_width*2 - col_width/2 - col_width, 0.0, 0.0])
        pts.append([col_width*2 - col_width/2 + col_width, 0.0, 0.0])
        return np.array([p[0] for p in pts] + [p[1] for p in pts] + [p[2] for p in pts])

    initializers = [make_flower, make_lattice]
    
    best_side = float('inf')
    best_params = None
    
    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 60:
            break
            
        x0 = init_func()
        
        # Run Nelder-Mead
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res.fun < best_side:
                # Check validity of the result
                params = res.x
                xs = params[0:N]
                ys = params[N:2*N]
                angles = params[2*N:3*N]
                
                verts_list = [_hex_vertices(xs[i], ys[i], angles[i]) for i in range(N)]
                axes_list = [_get_axes(v) for v in verts_list]
                
                is_valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        depth = _sat_overlap_depth(verts_list[i], verts_list[j], axes_list[i], axes_list[j])
                        if depth > 1e-6:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                
                if is_valid:
                    # Calculate precise side length
                    max_proj = 0.0
                    for v in verts_list:
                        for (vx, vy) in v:
                            for k in range(6):
                                theta_n = math.radians(30 + 60 * k)
                                nx = math.cos(theta_n)
                                ny = math.sin(theta_n)
                                proj = vx * nx + vy * ny
                                if proj > max_proj:
                                    max_proj = proj
                    side = max_proj / APOTHEM_UNIT
                    if side < best_side:
                        best_side = side
                        best_params = params
        except Exception as e:
            pass

    # If no valid solution found, use a fallback
    if best_params is None:
        best_params = make_flower()
        best_side = _objective(best_params)
        
    # Final refinement: try to optimize outer hexagon angle and center?
    # For now, we return the configuration assuming outer is (0,0,0).
    # The evaluator will find the best outer hexagon for the given inner set.
    # So we just need to return the inner positions.
    
    inner = []
    for i in range(N):
        x = float(best_params[i])
        y = float(best_params[N + i])
        a = float(best_params[2*N + i])
        inner.append([x, y, a])
        
    # We must return: inner, center, side, angle
    # The "side" and "angle" and "center" in the return value are for the OUTER hexagon.
    # The evaluator uses these to check validity and calculate score?
    # "score is the outer side length"
    # "inner rows (x, y, angle_degrees) ... outer hexagon center (x, y), side length and rotation"
    # So we should return the optimal outer hexagon parameters for the best inner configuration.
    
    # Let's find the best outer hexagon for the best inner configuration.
    cx_best, cy_best, angle_best, side_best = 0.0, 0.0, 0.0, best_side
    
    # Simple optimization of outer hexagon parameters
    # We can use a grid search or local optimization for the outer hexagon.
    # Since the inner configuration is fixed, the minimum enclosing hexagon can be found.
    
    # Function to get min enclosing hexagon side length for given outer params
    def _min_enclosing_hex_side(xs, ys, angles, cx, cy, angle_deg):
        a = math.radians(angle_deg)
        max_proj = 0.0
        for i in range(N):
            v = _hex_vertices(xs[i], ys[i], angles[i])
            for (vx, vy) in v:
                for k in range(6):
                    theta_n = a + math.radians(30 + 60 * k)
                    nx = math.cos(theta_n)
                    ny = math.sin(theta_n)
                    proj = (vx - cx) * nx + (vy - cy) * ny
                    if proj > max_proj:
                        max_proj = proj
        return max_proj / APOTHEM_UNIT

    # Optimize outer hexagon
    # Start with (0,0,0)
    current_side = _min_enclosing_hex_side(best_params[0:N], best_params[N:2*N], best_params[2*N:3*N], 0.0, 0.0, 0.0)
    
    # Try a few angles and centers
    for angle_deg in range(0, 60, 5):
        for cx in [-0.5, 0, 0.5]:
            for cy in [-0.5, 0, 0.5]:
                s = _min_enclosing_hex_side(best_params[0:N], best_params[N:2*N], best_params[2*N:3*N], cx, cy, float(angle_deg))
                if s < current_side:
                    current_side = s
                    cx_best, cy_best, angle_best = cx, cy, float(angle_deg)
                    
    side_best = current_side
    
    return inner, (cx_best, cy_best), side_best, angle_best

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
