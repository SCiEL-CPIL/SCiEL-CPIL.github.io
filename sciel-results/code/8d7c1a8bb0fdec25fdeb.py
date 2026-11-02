import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Define unit hexagon vertices (flat-top, side 1)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    # Radius 1
    hex_verts = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(0, 360, 60)
    ])
    
    # Topologies: list of (x, y) centers
    topologies = []
    
    # 1. 3x4 Grid
    centers_34 = []
    for i in range(3):
        for j in range(4):
            x = j * np.sqrt(3) + (i % 2) * np.sqrt(3) / 2
            y = i * 1.5
            centers_34.append([x, y])
    topologies.append(np.array(centers_34))
    
    # 2. 2x6 Grid
    centers_26 = []
    for i in range(2):
        for j in range(6):
            x = j * np.sqrt(3) + (i % 2) * np.sqrt(3) / 2
            y = i * 1.5
            centers_26.append([x, y])
    topologies.append(np.array(centers_26))
    
    # 3. 4x3 Grid (same as 3x4 but transposed, effectively rotated 90 deg in grid space, but we search rotation anyway)
    # Skip if similar to 3x4.
    
    # 4. "Hex" Cluster (1 center, 6 ring, 5 extra)
    # Let's try to make a more compact shape.
    # A 3x3 block is 9 hexes. Add 3 more.
    # 3x3 block:
    centers_33 = []
    for i in range(3):
        for j in range(3):
            x = j * np.sqrt(3) + (i % 2) * np.sqrt(3) / 2
            y = i * 1.5
            centers_33.append([x, y])
    # Add 3 more to the right side of the middle row?
    # Middle row is i=1. j=0,1,2.
    # Add j=3,4,5 to i=1? That makes it 3x3 + 3 = 12.
    # This is basically a 3x6 strip with top and bottom rows shortened.
    # Let's add them to make it more square-ish.
    # Add (3, 1), (3, 0), (3, 2)? No, (3,1) is neighbor of (2,1).
    # Let's just use the 3x4 and 2x6. They are standard.
    
    best_S = float('inf')
    best_config = None
    
    for centers in topologies:
        if time.time() > deadline:
            break
            
        # Center the cluster to minimize bounding box
        centers = centers - np.mean(centers, axis=0)
        
        # Pre-compute all vertices for this cluster at rotation 0
        # Actually, we need to rotate the whole cluster.
        # For a given rotation theta, the vertices are:
        # v_rot = R(theta) * (centers[:, None, :] + hex_verts[None, :, :])
        
        # We want to minimize S(theta)
        # S(theta) = 2/sqrt(3) * max_k max_p ( p_rot . n_k )
        
        # n_k are normals to the sides of the outer hexagon (orientation 0)
        # Outer hexagon: flat-top?
        # If we use the same orientation for inner and outer, the normals are at 30, 90, 150, 210, 270, 330 degrees.
        # Let's use normals at angles 30 + k*60.
        normals = np.array([
            [np.cos(np.deg2rad(30 + k*60)), np.sin(np.deg2rad(30 + k*60))] for k in range(6)
        ])
        
        # Scan theta
        best_theta = 0
        best_M = float('inf')
        
        # Coarse scan
        for theta_deg in np.arange(0, 60, 1.0):
            if time.time() > deadline:
                break
            theta = np.deg2rad(theta_deg)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]])
            
            # Rotate centers
            rc = centers @ R.T
            # Rotate hex vertices
            rhv = hex_verts @ R.T
            
            # All vertices
            # Shape: (12, 6, 2)
            # centers: (12, 2) -> (12, 1, 2)
            # hex_verts: (6, 2) -> (1, 6, 2)
            verts = rc[:, None, :] + rhv[None, :, :]
            
            # Project onto normals
            # verts: (12, 6, 2)
            # normals: (6, 2)
            # We need max over all vertices and all normals.
            # proj = np.einsum('ijk,k->ij', verts, normals) # (12, 6)
            # M = np.max(proj)
            
            # Optimization: 
            # p . n = x*n_x + y*n_y
            # We can compute this vectorized.
            proj = np.einsum('ijk,k->ij', verts, normals)
            M = np.max(proj)
            
            if M < best_M:
                best_M = M
                best_theta = theta
        
        # Refine around best_theta
        # Golden section or just finer grid
        theta_range = np.arange(best_theta - 1.0, best_theta + 1.0, 0.01)
        for theta_deg in theta_range:
            if time.time() > deadline:
                break
            theta = np.deg2rad(theta_deg)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]])
            
            rc = centers @ R.T
            rhv = hex_verts @ R.T
            verts = rc[:, None, :] + rhv[None, :, :]
            proj = np.einsum('ijk,k->ij', verts, normals)
            M = np.max(proj)
            
            if M < best_M:
                best_M = M
                best_theta = theta
        
        S = 2.0 * best_M / np.sqrt(3)
        
        if S < best_S:
            best_S = S
            # Store best config
            # inner: list of [x, y, angle]
            # angle is best_theta in degrees
            theta_deg = np.rad2deg(best_theta)
            c, s = np.cos(best_theta), np.sin(best_theta)
            R = np.array([[c, -s], [s, c]])
            rc = centers @ R.T
            
            inner = []
            for i in range(len(rc)):
                inner.append([rc[i][0], rc[i][1], theta_deg])
            
            best_config = (inner, [0.0, 0.0], S, 0.0)
            
    if best_config is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    return best_config

# EVOLVE-BLOCK-START
def construct():
    # Use the logic from above
    pass
# EVOLVE-BLOCK-END
