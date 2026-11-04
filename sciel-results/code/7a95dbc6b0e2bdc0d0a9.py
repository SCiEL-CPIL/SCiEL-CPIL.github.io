import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 140.0  # Safety margin below 150s
APOTHEM = math.sqrt(3) / 2.0
MIN_DIST_CIRCLE = 2 * APOTHEM

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

def _project_polygon(verts, axis):
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    """
    Fast validity check using distance-based pre-filter + SAT fallback.
    """
    n = len(cx_arr)
    min_dist_sq = (MIN_DIST_CIRCLE * MIN_DIST_CIRCLE) - 1e-9

    # Fast pre-check: Circle packing
    for i in range(n):
        dx = cx_arr[i] - cx_arr[i+1:]
        dy = cy_arr[i] - cy_arr[i+1:]
        dist_sq = dx*dx + dy*dy
        if np.any(dist_sq < min_dist_sq):
            return False

    # Exact check: SAT
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    
    axes_list = []
    for i in range(n):
        axes_list.append(_axes_from_polygon(verts_list[i]))
    
    for i in range(n):
        vi = verts_list[i]
        ai = axes_list[i]
        for j in range(i + 1, n):
            vj = verts_list[j]
            aj = axes_list[j]
            
            dx = cx_arr[i] - cx_arr[j]
            dy = cy_arr[i] - cy_arr[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq > (2 * math.sqrt(3)) ** 2:
                continue
            
            overlap = True
            for axis in ai:
                minA, maxA = _project_polygon(vi, axis)
                minB, maxB = _project_polygon(vj, axis)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    overlap = False
                    break
            
            if overlap:
                for axis in aj:
                    minA, maxA = _project_polygon(vi, axis)
                    minB, maxB = _project_polygon(vj, axis)
                    if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                        overlap = False
                        break
            
            if overlap:
                return False
    
    return True

def _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for i in range(len(cx_arr)):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def construct():
    t0 = time.time()
    random.seed(42)
    
    def make_lattice_initial():
        """
        Creates the 3-4-3 lattice configuration.
        This forms a hexagon of side length 4.
        """
        pts = []
        row_height = math.sqrt(3)
        col_width = 2.0
        
        # Row 1 (top): 3 hexagons
        for c in range(3):
            x = c * col_width - col_width
            y = row_height
            pts.append([x, y, 0.0])
        
        # Row 2 (middle): 4 hexagons
        for c in range(4):
            x = c * col_width - col_width - col_width/2
            y = 0.0
            pts.append([x, y, 0.0])
            
        # Row 3 (bottom): 3 hexagons
        for c in range(3):
            x = c * col_width - col_width
            y = -row_height
            pts.append([x, y, 0.0])
            
        return pts

    # Start with the known optimal lattice
    cfg = make_lattice_initial()
    
    cx_arr = np.array([p[0] for p in cfg])
    cy_arr = np.array([p[1] for p in cfg])
    angle_arr = np.array([p[2] for p in cfg])
    
    best_side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
    best_cfg = [[cx_arr[i], cy_arr[i], angle_arr[i]] for i in range(N)]
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Objective function for Nelder-Mead
    def objective(params):
        cx_p = params[0:N]
        cy_p = params[N:2*N]
        ang_p = params[2*N:3*N]
        
        # Check validity
        if not _is_valid_fast(cx_p, cy_p, ang_p):
            # Return a large penalty if invalid
            return 100.0
            
        # Compute side length with fixed outer center (0,0) and angle (0) for speed
        # We will optimize outer angle/center separately later
        side = _outer_hex_side(cx_p, cy_p, ang_p, (0.0, 0.0), 0.0)
        return side

    # Initial parameters
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = cx_arr[i]
        x0[3*i+1] = cy_arr[i]
        x0[3*i+2] = angle_arr[i]
        
    # Run Nelder-Mead to refine positions and rotations
    # We give it a decent time budget
    if time.time() - t0 < TIME_LIMIT - 20:
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'fatol': 1e-8, 'xtol': 1e-6, 'disp': False}
            )
            
            if result.fun < best_side:
                opt_params = result.x
                opt_cfg = []
                for i in range(N):
                    opt_cfg.append([opt_params[3*i], opt_params[3*i+1], opt_params[3*i+2]])
                
                opt_cx = np.array([p[0] for p in opt_cfg])
                opt_cy = np.array([p[1] for p in opt_cfg])
                opt_ang = np.array([p[2] for p in opt_cfg])
                
                if _is_valid_fast(opt_cx, opt_cy, opt_ang):
                    side = _outer_hex_side(opt_cx, opt_cy, opt_ang, (0.0, 0.0), 0.0)
                    if side < best_side:
                        best_side = side
                        best_cfg = opt_cfg
        except Exception:
            pass

    # Refine outer hexagon center and angle for the best inner configuration
    cx_best = np.array([p[0] for p in best_cfg])
    cy_best = np.array([p[1] for p in best_cfg])
    ang_best = np.array([p[2] for p in best_cfg])
    
    # Optimize outer angle
    # The lattice is symmetric, but slight rotations might help if the inner hexes are rotated
    # Try a few angles
    best_outer_angle = 0.0
    best_outer_center = (0.0, 0.0)
    best_final_side = best_side
    
    for angle_deg in [0, 5, 10, 15, 20, 25, 30]:
        if time.time() - t0 > TIME_LIMIT - 5:
            break
            
        # For a given angle, find the best center
        # The center that minimizes the max projection is the centroid of the "extreme" points
        # But since it's a hexagon, we can just grid search the center near the centroid
        centroid_x = np.mean(cx_best)
        centroid_y = np.mean(cy_best)
        
        found_better = False
        for dx in np.linspace(-0.5, 0.5, 5):
            for dy in np.linspace(-0.5, 0.5, 5):
                center = (centroid_x + dx, centroid_y + dy)
                side = _outer_hex_side(cx_best, cy_best, ang_best, center, angle_deg)
                if side < best_final_side:
                    best_final_side = side
                    best_outer_angle = angle_deg
                    best_outer_center = center
                    found_better = True
        
        # If we found a better angle, we could potentially re-run NM with this angle fixed
        # But for now, just updating the outer parameters is a good improvement
        
    # Final verification
    cx_final = np.array([p[0] for p in best_cfg])
    cy_final = np.array([p[1] for p in best_cfg])
    angle_final = np.array([p[2] for p in best_cfg])
    
    if not _is_valid_fast(cx_final, cy_final, angle_final):
        # Fallback to lattice
        cfg = make_lattice_initial()
        best_cfg = cfg
        cx_final = np.array([p[0] for p in cfg])
        cy_final = np.array([p[1] for p in cfg])
        angle_final = np.array([p[2] for p in cfg])
        best_final_side = _outer_hex_side(cx_final, cy_final, angle_final, (0.0, 0.0), 0.0)
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
    else:
        # Recompute side with final center/angle to be sure
        best_final_side = _outer_hex_side(cx_final, cy_final, angle_final, best_outer_center, best_outer_angle)
        
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, best_outer_center, best_final_side, best_outer_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
