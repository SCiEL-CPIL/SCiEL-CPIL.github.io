import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _hex_vertices(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + np.cos(theta)
        y = cy + np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def _get_outer_normals(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    normals = []
    for i in range(6):
        theta = angle_rad + np.pi/6 + i * np.pi / 3
        normals.append((np.cos(theta), np.sin(theta)))
    return np.array(normals)

def _min_enclosing_hex_side(inner_verts, outer_angle_deg):
    normals = _get_outer_normals(outer_angle_deg)
    max_proj = 0.0
    for verts in inner_verts:
        dots = np.abs(verts @ normals.T)
        max_proj = max(max_proj, np.max(dots))
    return 2 * max_proj / np.sqrt(3)

def _check_validity(centers, min_dist=np.sqrt(3)):
    """Check if configuration is valid (no overlaps)."""
    n = len(centers)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist - 1e-9:
                return False
    return True

def _compute_S(centers, angles, outer_angle):
    """Compute enclosing hex side length."""
    verts_list = []
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i][0], centers[i][1], angles[i])
        verts_list.append(verts)
    return _min_enclosing_hex_side(verts_list, outer_angle)

def _penalized_objective(vars, angles, min_dist=np.sqrt(3)):
    """Objective with penalty for overlaps."""
    c = vars[:22].reshape(11, 2)
    o_ang = vars[22]
    
    # Overlap penalty
    pen = 0.0
    for i in range(11):
        for j in range(i+1, 11):
            d = np.linalg.norm(c[i] - c[j])
            if d < min_dist:
                pen += (min_dist - d)**2
    
    if pen > 1e-8:
        return 1e6 + pen
    
    return _compute_S(c, angles, o_ang)

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    # Initial configuration: 11 hexagons in a valid hexagonal pattern
    # For 11 unit hexagons, the minimum enclosing hexagon has side length 3
    # Configuration: center + 6 in first ring + 4 in second ring
    eps = 1e-9
    centers_list = []
    centers_list.append([0.0, 0.0])
    
    # First ring: 6 hexagons at distance sqrt(3) from center
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Second ring: 4 hexagons at distance 3 from center (at 30, 90, 150, 210 degrees)
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + eps
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
        
    centers = np.array(centers_list)
    angles = np.array([30.0] * 11)
    outer_angle = 30.0
    
    # Verify initial configuration is valid
    if not _check_validity(centers):
        # If invalid, add more spacing
        centers[1:] *= 1.01
    
    initial_S = _compute_S(centers, angles, outer_angle)
    best_S = initial_S
    best_centers = centers.copy()
    best_o_ang = outer_angle
    
    # Strategy 1: Differential evolution for global optimization
    if time.time() < DEADLINE - 30:
        try:
            bounds = [(-5, 5)] * 22 + [(-90, 90)]
            
            def de_objective(vars):
                if time.time() > DEADLINE - 5:
                    return 1e9
                return _penalized_objective(vars, angles)
            
            res = differential_evolution(
                de_objective, bounds, 
                maxiter=50, popsize=15,
                tol=1e-6, seed=42,
                polish=False
            )
            
            if res.fun < best_S and res.fun < 1e5:
                best_vars = res.x
                best_centers = best_vars[:22].reshape(11, 2)
                best_o_ang = best_vars[22]
                best_S = res.fun
        except Exception:
            pass
    
    # Strategy 2: Nelder-Mead local refinement
    if time.time() < DEADLINE - 15:
        try:
            x0 = np.concatenate([best_centers.flatten(), [best_o_ang]])
            
            def nm_objective(vars):
                if time.time() > DEADLINE - 5:
                    return 1e9
                return _penalized_objective(vars, angles)
            
            res = minimize(
                nm_objective, x0, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res.fun < best_S and res.fun < 1e5:
                best_vars = res.x
                best_centers = best_vars[:22].reshape(11, 2)
                best_o_ang = best_vars[22]
                best_S = res.fun
        except Exception:
            pass
    
    # Strategy 3: Coordinate descent polish
    if time.time() < DEADLINE - 5:
        min_dist = np.sqrt(3)
        improved = True
        max_iter = 50
        
        for it in range(max_iter):
            if time.time() > DEADLINE - 2:
                break
            if not improved:
                break
            improved = False
            
            # Optimize outer angle
            for delta in [0.05, -0.05, 0.01, -0.01, 0.001, -0.001]:
                new_ang = best_o_ang + delta
                if _check_validity(best_centers):
                    S = _compute_S(best_centers, angles, new_ang)
                    if S < best_S - 1e-10:
                        best_S = S
                        best_o_ang = new_ang
                        improved = True
            
            # Optimize each center
            for i in range(11):
                for coord in range(2):
                    for delta in [0.01, -0.01, 0.001, -0.001]:
                        new_c = best_centers.copy()
                        new_c[i, coord] += delta
                        if _check_validity(new_c):
                            S = _compute_S(new_c, angles, best_o_ang)
                            if S < best_S - 1e-10:
                                best_S = S
                                best_centers = new_c
                                improved = True
        
    # Final verification
    if not _check_validity(best_centers):
        # Fallback to initial valid configuration
        best_centers = centers.copy()
        best_o_ang = outer_angle
        best_S = _compute_S(best_centers, angles, best_o_ang)
    
    # Final computation
    side = _compute_S(best_centers, angles, best_o_ang)
    
    inner = []
    for i in range(11):
        inner.append([best_centers[i][0], best_centers[i][1], angles[i]])
    
    return inner, (0.0, 0.0), side, best_o_ang
