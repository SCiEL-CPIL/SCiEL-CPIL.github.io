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
    n = len(centers)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist - 1e-9:
                return False
    return True

def _compute_S(centers, angles, outer_angle):
    verts_list = []
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i][0], centers[i][1], angles[i])
        verts_list.append(verts)
    return _min_enclosing_hex_side(verts_list, outer_angle)

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    n = 11
    min_dist = np.sqrt(3)
    
    # Initial configuration: hexagonal lattice pattern
    # Center hexagon at origin
    # First ring: 6 hexagons at distance sqrt(3)
    # Second ring: 4 hexagons at specific positions
    
    centers_list = []
    centers_list.append([0.0, 0.0])
    
    # First ring: 6 hexagons
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = min_dist
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Second ring: 4 hexagons (positions that complete the pattern)
    # In a hexagonal lattice, the second ring has 12 positions,
    # but we only need 4 more to make 11 total
    # These are at angles 30, 90, 150, 210 degrees at distance 3
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
        
    centers = np.array(centers_list)
    angles = np.array([30.0] * n)
    outer_angle = 30.0
    
    # Verify initial configuration is valid
    if not _check_validity(centers):
        # If invalid, add small perturbation
        centers[1:] *= 1.001
    
    best_S = _compute_S(centers, angles, outer_angle)
    best_centers = centers.copy()
    best_angles = angles.copy()
    best_o_ang = outer_angle
    
    # Strategy 1: Golden section search over outer rotation angle
    # This is a 1D optimization that's very fast
    def S_for_angle(o_ang, c, ang):
        return _compute_S(c, ang, o_ang)
    
    # Coarse search over outer angle
    if time.time() < DEADLINE - 30:
        # Search over a range of outer angles
        ang_min = -30.0
        ang_max = 30.0
        n_coarse = 61
        
        for i in range(n_coarse):
            o_ang = ang_min + (ang_max - ang_min) * i / (n_coarse - 1)
            S = S_for_angle(o_ang, best_centers, best_angles)
            if S < best_S:
                best_S = S
                best_o_ang = o_ang
    
    # Strategy 2: Fine golden section search around best angle
    if time.time() < DEADLINE - 20:
        # Golden section search
        phi = (1 + np.sqrt(5)) / 2
        invphi = 1.0 / phi
        invphi2 = 1.0 / (phi * phi)
        
        a = best_o_ang - 2.0
        b = best_o_ang + 2.0
        
        for _ in range(50):
            if time.time() > DEADLINE - 15:
                break
            c = b - invphi * (b - a)
            d = a + invphi * (b - a)
            
            fc = S_for_angle(c, best_centers, best_angles)
            fd = S_for_angle(d, best_centers, best_angles)
            
            if fc < fd:
                b = d
            else:
                a = c
            
            if abs(b - a) < 1e-6:
                break
        
        best_o_ang = (a + b) / 2
        best_S = S_for_angle(best_o_ang, best_centers, best_angles)
    
    # Strategy 3: Nelder-Mead optimization of centers and outer angle
    if time.time() < DEADLINE - 15:
        try:
            # Parameterize: 11*2 center coords + 1 outer angle = 23 variables
            # Keep angles fixed at 30 degrees for now
            
            x0 = np.concatenate([best_centers.flatten(), [best_o_ang]])
            
            def objective(vars):
                if time.time() > DEADLINE - 5:
                    return 1e9
                c = vars[:22].reshape(11, 2)
                o_ang = vars[22]
                
                # Check validity
                if not _check_validity(c):
                    # Compute penalty
                    pen = 0.0
                    for i in range(11):
                        for j in range(i+1, 11):
                            d = np.linalg.norm(c[i] - c[j])
                            if d < min_dist:
                                pen += (min_dist - d)**2
                    return 1e6 + pen
                
                return _compute_S(c, angles, o_ang)
            
            res = minimize(
                objective, x0, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res.fun < best_S and res.fun < 1e5:
                best_centers = res.x[:22].reshape(11, 2)
                best_o_ang = res.x[22]
                best_S = res.fun
        except Exception:
            pass
    
    # Strategy 4: Coordinate descent polish
    if time.time() < DEADLINE - 5:
        improved = True
        max_iter = 100
        
        for it in range(max_iter):
            if time.time() > DEADLINE - 2:
                break
            if not improved:
                break
            improved = False
            
            # Optimize outer angle with fine steps
            for delta in [0.01, -0.01, 0.001, -0.001, 0.0001, -0.0001]:
                new_ang = best_o_ang + delta
                if _check_validity(best_centers):
                    S = _compute_S(best_centers, angles, new_ang)
                    if S < best_S - 1e-12:
                        best_S = S
                        best_o_ang = new_ang
                        improved = True
            
            # Optimize each center coordinate
            for i in range(11):
                for coord in range(2):
                    for delta in [0.005, -0.005, 0.001, -0.001, 0.0001, -0.0001]:
                        new_c = best_centers.copy()
                        new_c[i, coord] += delta
                        if _check_validity(new_c):
                            S = _compute_S(new_c, angles, best_o_ang)
                            if S < best_S - 1e-12:
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
