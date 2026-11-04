import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # 1. Generate 12 centers in a 3x4 hexagonal lattice
    # Flat-top hexagons
    # Lattice vectors for touching flat sides
    # v1 = (1.5, sqrt(3)/2), v2 = (1.5, -sqrt(3)/2) is for pointy?
    # Let's use standard axial to cartesian for flat-top
    # q, r -> x = 1.5 * q, y = sqrt(3) * (r + q/2)
    
    centers = []
    # 3 columns (q = -1, 0, 1), 4 rows (r = -1, 0, 1, 2) to center it roughly
    # Let's use q in [-1, 0, 1] and r in [-1, 0, 1, 2]
    # Total 12 hexagons
    for q in range(-1, 2):
        for r in range(-1, 3):
            x = 1.5 * q
            y = np.sqrt(3) * (r + q / 2.0)
            centers.append((x, y))
            
    # 2. Generate vertices for each unit hexagon
    # Unit flat-top hexagon vertices relative to center
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    verts_rel = np.array([(np.cos(a), np.sin(a)) for a in angles])
    
    all_verts = []
    for cx, cy in centers:
        verts = verts_rel + np.array([cx, cy])
        all_verts.extend(verts)
        
    all_verts = np.array(all_verts)
    
    # 3. Find minimal enclosing regular hexagon
    # We search for center (cx, cy) and angle theta
    # The constraint is: for all vertices p, and for all 6 normal directions n_k(theta):
    # | (p - c) . n_k | <= s * sqrt(3) / 2
    
    centroid = np.mean(all_verts, axis=0)
    
    best_s = float('inf')
    best_c = centroid
    best_theta = 0.0
    best_inner = None
    
    # Search theta
    # Symmetry period is 60 degrees
    for theta_deg in range(0, 60, 2):
        if time.time() > deadline:
            break
        theta = np.deg2rad(theta_deg)
        # Normals at theta, theta+60, ...
        normals = []
        for k in range(6):
            ang = theta + np.deg2rad(60 * k)
            normals.append((np.cos(ang), np.sin(ang)))
        normals = np.array(normals)
        
        # Search center in a grid around centroid
        # Range: roughly 3 units
        grid_size = 10
        range_val = 3.0
        step = 2.0 * range_val / grid_size
        
        for gx in range(grid_size + 1):
            if time.time() > deadline:
                break
            cx = centroid[0] - range_val + gx * step
            for gy in range(grid_size + 1):
                cy = centroid[1] - range_val + gy * step
                c = np.array([cx, cy])
                
                # Calculate required s
                # s = max over all vertices p and normals n of | (p-c).n | / (sqrt(3)/2)
                # Vectorized:
                # diffs = all_verts - c (72 x 2)
                # dots = diffs @ normals.T (72 x 6)
                # abs_dots = np.abs(dots)
                # max_dot = np.max(abs_dots)
                # s_candidate = max_dot / (np.sqrt(3) / 2)
                
                diffs = all_verts - c
                dots = diffs @ normals.T
                max_dot = np.max(np.abs(dots))
                s_candidate = max_dot / (np.sqrt(3) / 2)
                
                if s_candidate < best_s:
                    best_s = s_candidate
                    best_c = c
                    best_theta = theta_deg
                    # Store the configuration
                    # Inner hexagons are at centers, angle 0 (flat-top)
                    # Outer hexagon center is best_c, side best_s, angle best_theta
                    inner = []
                    for cx_h, cy_h in centers:
                        inner.append([cx_h, cy_h, 0.0])
                    best_inner = inner

    # If no valid search completed, fallback to known good value
    if best_s == float('inf'):
        best_s = 5.0
        best_c = (0.0, 0.0)
        best_theta = 0.0
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    else:
        inner = best_inner
        # Ensure the inner hexagons are disjoint (they are by construction in lattice)
        # The outer hexagon parameters
        center = tuple(best_c)
        side = best_s
        angle = best_theta

    return inner, center, side, angle

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    import numpy as np
    import time

    N = 12
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # 1. Generate 12 centers in a 3x4 hexagonal lattice
    # Flat-top hexagons
    centers = []
    for q in range(-1, 2):
        for r in range(-1, 3):
            x = 1.5 * q
            y = np.sqrt(3) * (r + q / 2.0)
            centers.append((x, y))
            
    # 2. Generate vertices for each unit hexagon
    angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    verts_rel = np.array([(np.cos(a), np.sin(a)) for a in angles])
    
    all_verts = []
    for cx, cy in centers:
        verts = verts_rel + np.array([cx, cy])
        all_verts.extend(verts)
        
    all_verts = np.array(all_verts)
    
    # 3. Find minimal enclosing regular hexagon
    centroid = np.mean(all_verts, axis=0)
    
    best_s = float('inf')
    best_c = centroid
    best_theta = 0.0
    best_inner = None
    
    # Search theta
    for theta_deg in range(0, 60, 2):
        if time.time() > deadline:
            break
        theta = np.deg2rad(theta_deg)
        normals = []
        for k in range(6):
            ang = theta + np.deg2rad(60 * k)
            normals.append((np.cos(ang), np.sin(ang)))
        normals = np.array(normals)
        
        # Search center in a grid around centroid
        grid_size = 10
        range_val = 3.0
        step = 2.0 * range_val / grid_size
        
        for gx in range(grid_size + 1):
            if time.time() > deadline:
                break
            cx = centroid[0] - range_val + gx * step
            for gy in range(grid_size + 1):
                cy = centroid[1] - range_val + gy * step
                c = np.array([cx, cy])
                
                diffs = all_verts - c
                dots = diffs @ normals.T
                max_dot = np.max(np.abs(dots))
                s_candidate = max_dot / (np.sqrt(3) / 2)
                
                if s_candidate < best_s:
                    best_s = s_candidate
                    best_c = c
                    best_theta = theta_deg
                    inner = []
                    for cx_h, cy_h in centers:
                        inner.append([cx_h, cy_h, 0.0])
                    best_inner = inner

    if best_s == float('inf'):
        # Fallback to a very loose but valid configuration
        # Just place them in a line with huge s
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    else:
        inner = best_inner
        center = tuple(best_c)
        side = best_s
        angle = best_theta
        return inner, center, side, angle
# EVOLVE-BLOCK-END
