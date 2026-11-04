"""Pack 11 unit hexagons inside a minimal regular hexagon."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 11
DEADLINE = 115.0  # safety margin before 120s limit

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), rotation in degrees."""
    a = np.deg2rad(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _min_enclosing_hex_side(composite_verts, outer_angle_deg, outer_center):
    """
    Compute minimum side length S of outer hexagon (centered at outer_center, rotated outer_angle_deg)
    to contain all points in composite_verts.
    """
    angle_rad = np.deg2rad(outer_angle_deg)
    normals = []
    for i in range(6):
        theta = angle_rad + np.pi / 6 + i * np.pi / 3
        normals.append([np.cos(theta), np.sin(theta)])
    normals = np.array(normals)
    
    rel_verts = composite_verts - np.array(outer_center)
    dots = rel_verts @ normals.T
    max_proj = np.max(np.abs(dots))
    return 2 * max_proj / np.sqrt(3)

def _compute_composite_hull(centers, angles):
    """Compute convex hull of all vertices of all inner hexagons."""
    all_verts = []
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i][0], centers[i][1], angles[i])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    try:
        hull = ConvexHull(all_verts)
        hull_verts = all_verts[hull.vertices]
        return hull_verts
    except Exception:
        return all_verts

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return DEADLINE - elapsed()
    
    # Initial configuration: 1 center + 6 ring + 4 outer ring
    # This is a known good packing for 11 hexagons
    centers_list = []
    angles_list = []
    
    # Center hexagon
    centers_list.append([0.0, 0.0])
    angles_list.append(0.0)
    
    # Inner ring: 6 hexagons at distance sqrt(3) from center
    for k in range(6):
        angle = k * np.pi / 3
        r = np.sqrt(3)
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
        angles_list.append(0.0)
    
    # Outer ring: 4 hexagons at distance 2*sqrt(3) from center, at 30, 90, 150, 210 degrees
    for k in range(4):
        angle = np.pi / 6 + k * np.pi / 3
        r = 2 * np.sqrt(3)
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
        angles_list.append(0.0)
        
    centers = np.array(centers_list)
    angles = np.array(angles_list)
    
    # Strategy: Treat the cluster as a rigid body and optimize the outer hexagon's
    # center and rotation to minimize the enclosing side length.
    # This exploits the symmetry of the packing.
    
    def outer_objective(params):
        """Objective: minimize outer hexagon side length given cluster positions."""
        ocx, ocy, oangle = params
        composite = _compute_composite_hull(centers, angles)
        S = _min_enclosing_hex_side(composite, oangle, (ocx, ocy))
        return S
    
    # Multi-start optimization for outer hexagon
    best_S = float('inf')
    best_params = None
    
    # Try multiple initial rotations
    init_angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
    
    for init_angle in init_angles:
        if time_left() < 5.0:
            break
        
        x0 = np.array([0.0, 0.0, init_angle])
        
        try:
            res = minimize(outer_objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # If we have time, also try to optimize inner positions slightly
    # to further reduce the enclosing hexagon size
    if time_left() > 20.0 and best_params is not None:
        ocx, ocy, oangle = best_params
        
        def full_objective(params):
            """Objective: minimize outer hexagon side length, optimizing inner positions and outer params."""
            n = len(params) // 3
            # First 3 params are outer center and angle
            ocx_f, ocy_f, oangle_f = params[0], params[1], params[2]
            
            # Remaining params are inner centers and angles
            inner_centers = []
            inner_angles = []
            for i in range(n):
                cx = params[3 + 3*i]
                cy = params[3 + 3*i + 1]
                angle = params[3 + 3*i + 2]
                inner_centers.append([cx, cy])
                inner_angles.append(angle)
            
            inner_centers = np.array(inner_centers)
            inner_angles = np.array(inner_angles)
            
            composite = _compute_composite_hull(inner_centers, inner_angles)
            S = _min_enclosing_hex_side(composite, oangle_f, (ocx_f, ocy_f))
            return S
        
        # Initial params: outer params + inner params
        init_params = np.array([ocx, ocy, oangle])
        for i in range(N):
            init_params = np.append(init_params, [centers[i][0], centers[i][1], angles[i]])
        
        try:
            res = minimize(full_objective, init_params, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Extract final parameters
    if best_params is None:
        ocx, ocy, oangle = 0.0, 0.0, 0.0
        final_centers = centers
        final_angles = angles
    else:
        if len(best_params) == 3:
            ocx, ocy, oangle = best_params
            final_centers = centers
            final_angles = angles
        else:
            ocx, ocy, oangle = best_params[0], best_params[1], best_params[2]
            final_centers = []
            final_angles = []
            for i in range(N):
                cx = best_params[3 + 3*i]
                cy = best_params[3 + 3*i + 1]
                angle = best_params[3 + 3*i + 2]
                final_centers.append([cx, cy])
                final_angles.append(angle)
    
    # Compute final side length
    final_centers = np.array(final_centers)
    final_angles = np.array(final_angles)
    composite = _compute_composite_hull(final_centers, final_angles)
    side = _min_enclosing_hex_side(composite, oangle, (ocx, ocy))
    
    # Build inner list
    inner = []
    for i in range(N):
        inner.append([final_centers[i][0], final_centers[i][1], final_angles[i]])
    
    return inner, (ocx, ocy), side, oangle
