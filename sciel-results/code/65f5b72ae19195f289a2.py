import numpy as np
import math
import time

N = 12

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # Define unit hexagon vertices
    # Orientation: Vertex at (1, 0)
    hex_verts = np.array([
        [1.0, 0.0],
        [0.5, math.sqrt(3)/2],
        [-0.5, math.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -math.sqrt(3)/2],
        [0.5, -math.sqrt(3)/2]
    ])
    
    # Outer hexagon normals (edge normals)
    # Angles: 30, 90, 150, 210, 270, 330
    outer_normals = np.array([
        [math.cos(math.radians(30 + k*60)), math.sin(math.radians(30 + k*60))]
        for k in range(6)
    ])
    
    best_side = float('inf')
    best_config = None
    
    # Try different W x H honeycomb blocks
    # W, H are dimensions in hexagon units
    # Total hexagons = W * H = 12
    dims = [(1, 12), (2, 6), (3, 4), (4, 3), (6, 2), (12, 1)]
    
    for w, h in dims:
        # Generate centers for w x h honeycomb
        # Center the block at (0,0)
        # i in 0..w-1, j in 0..h-1
        # Standard honeycomb offset
        centers = []
        for i in range(w):
            for j in range(h):
                x = i * math.sqrt(3) + (j % 2) * math.sqrt(3)/2
                y = j * 1.5
                centers.append([x, y])
        
        centers = np.array(centers)
        # Center the block
        centroid = np.mean(centers, axis=0)
        centers = centers - centroid
        
        # Optimize rotation theta
        # Search theta in [0, 30 deg]
        best_theta = 0.0
        best_L = float('inf')
        
        # Grid search
        # Use 121 steps for better initial resolution
        for theta_deg in np.linspace(0, 30, 121):
            theta = math.radians(theta_deg)
            c, s = math.cos(theta), math.sin(theta)
            R = np.array([[c, -s], [s, c]])
            
            # Rotate centers
            rot_centers = centers @ R.T
            
            max_proj = 0.0
            for n in outer_normals:
                # Project rotated centers onto normal n
                dots = rot_centers @ n
                max_c_dot = np.max(dots)
                
                # Add support function of unit hexagon in direction n
                h_val = np.max(hex_verts @ n)
                
                total_proj = max_c_dot + h_val
                if total_proj > max_proj:
                    max_proj = total_proj
            
            L = (2.0 / math.sqrt(3)) * max_proj
            if L < best_L:
                best_L = L
                best_theta = theta
        
        # Refine with ternary search around best_theta
        def eval_L(theta):
            c, s = math.cos(theta), math.sin(theta)
            R = np.array([[c, -s], [s, c]])
            rot_centers = centers @ R.T
            max_proj = 0.0
            for n in outer_normals:
                dots = rot_centers @ n
                max_c_dot = np.max(dots)
                h_val = np.max(hex_verts @ n)
                total_proj = max_c_dot + h_val
                if total_proj > max_proj:
                    max_proj = total_proj
            return (2.0 / math.sqrt(3)) * max_proj
        
        a = max(0.0, best_theta - 1.0)
        b = min(math.radians(30), best_theta + 1.0)
        
        # Ternary search
        for _ in range(100):
            m1 = a + (b - a) / 3.0
            m2 = b - (b - a) / 3.0
            if eval_L(m1) < eval_L(m2):
                b = m2
            else:
                a = m1
        
        theta_opt = (a + b) / 2.0
        L_opt = eval_L(theta_opt)
        
        if L_opt < best_side:
            best_side = L_opt
            # Construct inner list
            inner = []
            c, s = math.cos(theta_opt), math.sin(theta_opt)
            R = np.array([[c, -s], [s, c]])
            rot_centers = centers @ R.T
            for c_pt in rot_centers:
                inner.append([c_pt[0], c_pt[1], 0.0])
            
            best_config = (inner, (0.0, 0.0), L_opt, 0.0)
            
        # Check time
        if time.time() - start_time > deadline - 5.0:
            break
            
    if best_config is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    return best_config
