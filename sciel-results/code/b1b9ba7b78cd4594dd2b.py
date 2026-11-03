import numpy as np
from scipy.optimize import minimize

N = 12

def _hex_normals():
    """Return the 6 outward normals of a unit regular hexagon (flat-side aligned with x-axis at 0 deg).
    If rotation is 0, normals are at 0, 60, 120, 180, 240, 300.
    """
    angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _support_hex(u, rot_rad):
    """
    Compute support function of a unit regular hexagon (side 1) with rotation rot_rad in direction u.
    The hexagon has flat sides normal to angles 0, 60, ... when rot=0.
    Vertices are at 30, 90, ... when rot=0.
    Support in direction u (angle phi) is max dot product of u with vertices.
    Vertices v_k have angles rot_rad + 30 + 60k.
    v_k = [cos(rot+30+60k), sin(rot+30+60k)].
    h(u) = max_k u . v_k.
    """
    phi = np.arctan2(u[1], u[0])
    # Angles of vertices relative to u
    # We want to maximize cos(phi - (rot + 30 + 60k))
    # Max is 1 if phi == rot + 30 + 60k
    # Generally, the max is cos(delta) where delta is min angle diff to nearest vertex angle.
    # But we can just compute the 6 dots.
    # Optimization: The support function of a regular polygon is periodic and piecewise cosine.
    # For a hexagon, h(theta) = cos(3*(theta - theta_vertex_mod))? No.
    # Let's just compute the 6 vertices dots. It's fast enough for 3 params.
    vertex_angles = rot_rad + np.deg2rad(30) + np.deg2rad(60) * np.arange(6)
    vertices = np.column_stack((np.cos(vertex_angles), np.sin(vertex_angles)))
    dots = vertices @ u
    return np.max(dots)

def construct():
    sqrt3 = np.sqrt(3.0)
    normals_container = _hex_normals() # 0, 60, ...
    
    # Topology: 1 center, 5 inner, 6 outer
    # Parameters: r_in, r_out, theta_global
    # Inner ring angles: theta_global + [0, 72, 144, 216, 288]
    # Outer ring angles: theta_global + [0, 60, 120, 180, 240, 300]
    # Orientation: Flat side out.
    # If flat side is out, the normal aligns with the radial vector.
    # So rotation = radial angle.
    
    def get_centers_and_rots(params):
        r_in, r_out, theta_glob = params
        theta_glob_rad = np.deg2rad(theta_glob)
        
        centers = np.zeros((N, 2))
        rots = np.zeros(N)
        
        # Center
        centers[0] = [0.0, 0.0]
        rots[0] = 0.0 # Center rotation doesn't matter for extent, but matters for separation. 
                      # Let's assume center is aligned with container (0).
        
        # Inner ring (5)
        inner_angles_deg = np.array([0, 72, 144, 216, 288])
        inner_angles_rad = np.deg2rad(inner_angles_deg) + theta_glob_rad
        for k in range(5):
            centers[1 + k] = [r_in * np.cos(inner_angles_rad[k]), r_in * np.sin(inner_angles_rad[k])]
            rots[1 + k] = inner_angles_rad[k] # Flat side out
            
        # Outer ring (6)
        outer_angles_deg = np.array([0, 60, 120, 180, 240, 300])
        outer_angles_rad = np.deg2rad(outer_angles_deg) + theta_glob_rad
        for k in range(6):
            centers[6 + k] = [r_out * np.cos(outer_angles_rad[k]), r_out * np.sin(outer_angles_rad[k])]
            rots[6 + k] = outer_angles_rad[k] # Flat side out
            
        return centers, rots

    def objective(params):
        centers, rots = get_centers_and_rots(params)
        
        # Calculate required L
        # L >= (N_cont . c_i + h_i(N_cont)) / h_cont(N_cont)
        # h_cont(N_cont) = sqrt(3)/2
        max_extent = 0.0
        for i in range(N):
            c = centers[i]
            rot = rots[i]
            for nk in normals_container:
                # Support of inner hex in direction nk
                h_in = _support_hex(nk, rot)
                extent = nk[0]*c[0] + nk[1]*c[1] + h_in
                if extent > max_extent:
                    max_extent = extent
        L = 2.0 * max_extent / sqrt3
        
        # Calculate separation violations
        # For each pair, check if they overlap.
        # Overlap if max_u (h_i(u) + h_j(-u) - u . (c_j - c_i)) > 0
        # We check the 12 normals of the Minkowski sum.
        # Normals of Minkowski sum of two hexagons with rotations r1, r2:
        # Union of normals of hex1 and hex2.
        # Hex1 normals: r1 + [0, 60, ...]
        # Hex2 normals: r2 + [0, 60, ...]
        # The Minkowski sum has normals at the union of these angles (sorted).
        
        max_viol = 0.0
        for i in range(N):
            for j in range(i+1, N):
                c_diff = centers[j] - centers[i]
                r1 = rots[i]
                r2 = rots[j]
                
                # Generate candidate normals
                n1 = r1 + np.deg2rad(np.arange(0, 360, 60))
                n2 = r2 + np.deg2rad(np.arange(0, 360, 60))
                all_n = np.concatenate([n1, n2])
                all_n = np.sort(all_n % (2*np.pi))
                
                viol = 0.0
                for ang in all_n:
                    u = np.array([np.cos(ang), np.sin(ang)])
                    # Support of i in u
                    h_i = _support_hex(u, r1)
                    # Support of j in -u
                    # h_j(-u) = support of j in direction -u
                    # Note: _support_hex expects a unit vector.
                    # -u is direction ang + pi.
                    h_j = _support_hex(-u, r2)
                    
                    # Separation condition: u . c_diff > h_i + h_j
                    # Violation: h_i + h_j - u . c_diff > 0
                    sep = u[0]*c_diff[0] + u[1]*c_diff[1]
                    v = h_i + h_j - sep
                    if v > viol:
                        viol = v
                        
                if viol > max_viol:
                    max_viol = viol
                    
        # Penalty
        # Use a high penalty to enforce separation
        penalty = 100.0 * max_viol
        return L + penalty

    # Optimization
    best_L = float('inf')
    best_params = None
    
    # Initial guesses
    # r_in ~ 1.732 (touching center if flat-side-in? No, center is rot 0. 
    # Center has flat side at 0. Inner hex has flat side at alpha.
    # If alpha=0, inner hex flat side is at 0 (right). Center flat side is at 0.
    # They touch if distance is 2 * apothem? No.
    # Center support in x is sqrt(3)/2. Inner support in -x is sqrt(3)/2.
    # So distance sqrt(3) ~ 1.732.
    
    # r_out ~ 3.46
    
    for seed in range(50):
        np.random.seed(seed)
        r_in = np.random.uniform(1.6, 2.0)
        r_out = np.random.uniform(3.2, 3.8)
        theta = np.random.uniform(0, 360)
        
        x0 = np.array([r_in, r_out, theta])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            params = result.x
            
            # Verify
            centers, rots = get_centers_and_rots(params)
            max_viol = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    c_diff = centers[j] - centers[i]
                    r1 = rots[i]
                    r2 = rots[j]
                    n1 = r1 + np.deg2rad(np.arange(0, 360, 60))
                    n2 = r2 + np.deg2rad(np.arange(0, 360, 60))
                    all_n = np.concatenate([n1, n2])
                    all_n = np.sort(all_n % (2*np.pi))
                    for ang in all_n:
                        u = np.array([np.cos(ang), np.sin(ang)])
                        h_i = _support_hex(u, r1)
                        h_j = _support_hex(-u, r2)
                        sep = u[0]*c_diff[0] + u[1]*c_diff[1]
                        v = h_i + h_j - sep
                        if v > max_viol:
                            max_viol = v
                            
            if max_viol > 1e-4:
                continue
                
            # Calculate L
            max_extent = 0.0
            for i in range(N):
                c = centers[i]
                rot = rots[i]
                for nk in normals_container:
                    h_in = _support_hex(nk, rot)
                    extent = nk[0]*c[0] + nk[1]*c[1] + h_in
                    if extent > max_extent:
                        max_extent = extent
            L_final = 2.0 * max_extent / sqrt3
            
            if L_final < best_L:
                best_L = L_final
                best_params = params.copy()
                
        except Exception:
            continue
            
    if best_params is None:
        # Fallback
        return [[0.0, 0.0, 0.0]] + [[1.732, 0.0, 0.0]] * 5 + [[3.464, 0.0, 0.0]] * 6, (0.0, 0.0), 8.0, 0.0
        
    centers, rots = get_centers_and_rots(best_params)
    inner = []
    for i in range(N):
        inner.append([float(centers[i, 0]), float(centers[i, 1]), float(np.rad2deg(rots[i]))])
        
    return inner, (0.0, 0.0), best_L, 0.0
