import numpy as np
import time
import scipy.optimize

N = 11

def hex_vertices(x, y, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    verts = verts @ R.T
    verts += np.array([x, y])
    return verts

def construct():
    start_time = time.time()
    deadline = 145.0
    
    d = np.sqrt(3)
    
    # Lattice vectors
    u = np.array([1.5, d/2])
    v = np.array([0.0, d])
    
    # 7 core hexagons
    core_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
    ]
    
    # 4 outer hexagons. 
    # To minimize the bounding box, we pick 4 from the second ring.
    # The second ring has 12 points.
    # Symmetric choice: (2,0), (0,2), (-2,0), (0,-2) ? 
    # Or (1,1), (-1,-1), (1,-1)?? No.
    # Let's try the 4 "diagonal" ones: (1,1), (-1,-1) are opposite.
    # (2,-1) and (-2,1) are opposite.
    # Let's try: (2,0), (-2,0), (1,1), (-1,-1).
    # This is not fully symmetric.
    # Better: (2,0), (0,2), (-2,0), (0,-2).
    outer_coords = [
        (2, 0), (0, 2), (-2, 0), (0, -2)
    ]
    
    all_coords = core_coords + outer_coords
    cluster_centers = np.array([i * u + j * v for i, j in all_coords])
    
    # Scale up slightly for disjointness
    cluster_centers *= 1.001
    
    # Pre-compute vertices
    base_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    all_base_verts = np.tile(base_verts, (N, 1, 1))
    cluster_verts = all_base_verts + cluster_centers[:, np.newaxis, :]
    flat_verts = cluster_verts.reshape(-1, 2)
    
    def compute_side(angle_cluster_deg, phi_deg):
        angle_rad = np.deg2rad(angle_cluster_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        rotated = flat_verts @ R.T
        
        phi_rad = np.deg2rad(phi_deg)
        normals = np.array([
            [np.cos(phi_rad + k * np.pi / 3), np.sin(phi_rad + k * np.pi / 3)]
            for k in range(6)
        ])
        projs = rotated @ normals.T
        max_proj = np.max(projs)
        return 2.0 * max_proj / np.sqrt(3)
    
    # Optimize angles
    best_side = float('inf')
    best_theta = 0.0
    best_phi = 0.0
    
    # Grid search
    for theta in np.linspace(0, 60, 61, endpoint=False):
        for phi in np.linspace(0, 60, 61, endpoint=False):
            s = compute_side(theta, phi)
            if s < best_side:
                best_side = s
                best_theta = theta
                best_phi = phi
                
    # Refine
    if time.time() < deadline - 5:
        def obj(params):
            return compute_side(params[0], params[1])
        
        try:
            res = scipy.optimize.minimize(
                obj,
                np.array([best_theta, best_phi]),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res.x[0] >= 0 and res.x[1] >= 0:
                s_res = compute_side(res.x[0], res.x[1])
                if s_res < best_side:
                    best_side = s_res
                    best_theta = res.x[0]
                    best_phi = res.x[1]
        except Exception:
            pass
            
    # Build inner hexagons
    angle_rad = np.deg2rad(best_theta)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    rotated_centers = cluster_centers @ R.T
    
    inner = []
    for i in range(N):
        x = float(rotated_centers[i, 0])
        y = float(rotated_centers[i, 1])
        ang = float(best_theta)
        inner.append([x, y, ang])
        
    return inner, (0.0, 0.0), float(best_side), float(best_phi)
