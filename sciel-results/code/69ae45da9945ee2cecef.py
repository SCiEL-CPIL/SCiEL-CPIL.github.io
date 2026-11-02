import numpy as np
from scipy.optimize import minimize
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Define lattice vectors
    a = np.array([np.sqrt(3), 0])
    b = np.array([np.sqrt(3)/2, 1.5])
    
    # Define unit hexagon vertices (flat-top, centered at origin)
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.deg2rad(np.arange(0, 360, 60))
    unit_hex_verts = np.array([np.cos(angles), np.sin(angles)]).T
    
    # Generate candidate clusters
    clusters = []
    
    # Cluster 1: 3x4 Parallelogram
    pts1 = []
    for i in range(3):
        for j in range(4):
            pts1.append(i*a + j*b)
    clusters.append(np.array(pts1))
    
    # Cluster 2: 2x6 Parallelogram
    pts2 = []
    for i in range(2):
        for j in range(6):
            pts2.append(i*a + j*b)
    clusters.append(np.array(pts2))
    
    # Cluster 3: Compact 12 (1 center + 6 ring 1 + 5 ring 2)
    # Axial coords (q,r)
    axial = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2), (-2,1)
    ]
    pts3 = []
    for q, r in axial:
        pts3.append(q*a + r*b)
    clusters.append(np.array(pts3))
    
    # Cluster 4: 3x3 Rhombus + 3 extra
    # 3x3 rhombus: i in 0..2, j in 0..2
    pts4 = []
    for i in range(3):
        for j in range(3):
            pts4.append(i*a + j*b)
    # Add 3 more adjacent to the end of the rhombus
    # End of rhombus is around (2,2) or (0,2)?
    # Let's just add (2,3), (1,3), (0,3) to make a 3x4 but missing corner?
    # Actually, let's just use a 4x3 parallelogram (same as 3x4 rotated)
    # Let's try a "staircase"
    pts4.append(2*a + 3*b)
    pts4.append(1*a + 3*b)
    pts4.append(0*a + 3*b)
    clusters.append(np.array(pts4))

    best_S = float('inf')
    best_params = None
    best_cluster_idx = 0
    
    # Outer hexagon normals (flat-top, side S)
    # Normals are perpendicular to sides.
    # Sides are at 30, 90, 150, 210, 270, 330 degrees?
    # If vertices are at 0, 60..., sides are at 30, 90...
    # Let's verify.
    # Vertex (1,0). Side connects (1,0) and (1/2, sqrt(3)/2).
    # Midpoint (3/4, sqrt(3)/4). Normal is (3, 1) normalized?
    # Angle 30 deg.
    # So normals are at 30, 90, 150, 210, 270, 330.
    n_angles = np.deg2rad(np.arange(30, 360, 60))
    normals = np.array([np.cos(n_angles), np.sin(n_angles)]).T
    
    for idx, cluster in enumerate(clusters):
        if time.time() > deadline - 5.0:
            break
            
        def objective(params):
            x0, y0, theta = params
            c = np.array([x0, y0])
            R = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
            
            # Rotate and translate centers
            centers_rot = (R @ cluster.T).T + c
            
            # All vertices
            # For each center, add rotated unit hex vertices
            # unit_hex_verts is fixed relative to cluster orientation
            # So rotate unit_hex_verts by theta as well?
            # Yes, the whole cluster rotates.
            verts_rot = (R @ unit_hex_verts.T).T
            
            # Generate all 72 vertices
            # centers_rot (12,2) + verts_rot (6,2)
            all_verts = (centers_rot[:, np.newaxis, :] + verts_rot[np.newaxis, :, :]).reshape(-1, 2)
            
            # Project onto 6 normals
            # all_verts (72, 2) @ normals.T (2, 6) -> (72, 6)
            projections = all_verts @ normals.T
            max_projs = np.max(projections, axis=0) # (6,)
            
            # S = 2/sqrt(3) * max(max_projs)
            S_val = (2.0 / np.sqrt(3.0)) * np.max(max_projs)
            return S_val
        
        # Initial guess
        # Center the cluster at origin for x0, y0
        cluster_center = np.mean(cluster, axis=0)
        x0_init = -cluster_center[0]
        y0_init = -cluster_center[1]
        theta_init = 0.0
        
        res = minimize(objective, [x0_init, y0_init, theta_init], method='Nelder-Mead',
                       options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 1000})
        
        if res.success:
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
                best_cluster_idx = idx
                
    # Reconstruct the best solution
    if best_params is None:
        # Fallback
        return [[0,0,0] for _ in range(N)], (0,0), 10.0, 0.0
        
    x0, y0, theta = best_params
    cluster = clusters[best_cluster_idx]
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    centers_rot = (R @ cluster.T).T + np.array([x0, y0])
    
    # Convert to output format: (x, y, angle_degrees)
    inner = []
    angle_deg = np.rad2deg(theta)
    for c in centers_rot:
        inner.append([c[0], c[1], angle_deg])
        
    # Outer hexagon: center (0,0), side best_S, angle 0
    # The problem asks for outer hexagon center, side, angle.
    # We assumed outer is at (0,0) angle 0.
    # So return (0,0), best_S, 0.0
    
    return inner, (0.0, 0.0), best_S, 0.0
