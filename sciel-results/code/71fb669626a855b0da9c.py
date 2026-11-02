import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0
    
    # Constants
    hex_spacing = np.sqrt(3)
    
    # Initialize positions
    # 7 hexagons in a cluster
    centers_7 = [
        (0.0, 0.0),
        (hex_spacing, 0.0),
        (hex_spacing/2, 1.5),
        (-hex_spacing/2, 1.5),
        (-hex_spacing, 0.0),
        (-hex_spacing/2, -1.5),
        (hex_spacing/2, -1.5)
    ]
    # 4 additional hexagons, placed roughly outside
    # Let's place them at the "tips" of the 6 arms, but we only have 4.
    # Maybe place them at angles 0, 120, 240, 180?
    # Or just randomize around the 7-cluster.
    
    def get_initial_positions():
        pos = []
        for p in centers_7:
            pos.extend([p[0], p[1]])
        # Add 4 more
        for k in range(4):
            angle = 2 * np.pi * k / 4
            r = 2 * hex_spacing
            pos.extend([r * np.cos(angle), r * np.sin(angle)])
        return np.array(pos)

    def objective(x):
        centers = x.reshape(-1, 2)
        # Penalty for overlap
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < hex_spacing:
                    penalty += (hex_spacing - d) ** 2
        
        # Calculate bounding hexagon side length
        # Vertices of unit hexagon
        verts = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        
        # Normals to sides of outer hexagon
        normals = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        
        max_proj = 0.0
        for c in centers:
            # All vertices of this hexagon
            vs = c + verts
            # Projections onto normals
            projs = np.abs(vs @ normals.T)
            max_proj = max(max_proj, np.max(projs))
        
        side_len = 2.0 * max_proj / np.sqrt(3)
        
        return side_len + 10.0 * penalty

    best_score = 100.0
    best_x = None
    
    # Try a few random restarts
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        x0 = get_initial_positions()
        # Add random noise
        x0 += np.random.normal(0, 0.5, x0.shape)
        
        # Nelder-Mead
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x
        except Exception:
            pass

    if best_x is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    centers = best_x.reshape(-1, 2)
    
    # Recalculate optimal side length and center
    verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    normals = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    max_proj = 0.0
    min_x, min_y = np.inf, np.inf
    max_x, max_y = -np.inf, -np.inf
    
    for c in centers:
        vs = c + verts
        min_x = min(min_x, np.min(vs[:,0]))
        max_x = max(max_x, np.max(vs[:,0]))
        min_y = min(min_y, np.min(vs[:,1]))
        max_y = max(max_y, np.max(vs[:,1]))
        
        projs = np.abs(vs @ normals.T)
        max_proj = max(max_proj, np.max(projs))
        
    # The center of the bounding box might not be (0,0) if the shape is asymmetric.
    # But for a hexagon, the "center" is the center of the circumscribing hexagon.
    # We found max_proj assuming center (0,0).
    # If the shape is shifted, we should re-center.
    # Let's re-center the configuration to the centroid of the vertices?
    # Or just calculate the minimal enclosing hexagon.
    # For simplicity, assume the optimal outer hexagon is centered at the centroid of the inner centers.
    cx = np.mean(centers[:,0])
    cy = np.mean(centers[:,1])
    
    # Shift centers to center of mass
    centers_shifted = centers - np.array([cx, cy])
    
    # Recalculate max_proj with shifted centers
    max_proj = 0.0
    for c in centers_shifted:
        vs = c + verts
        projs = np.abs(vs @ normals.T)
        max_proj = max(max_proj, np.max(projs))
        
    side_len = 2.0 * max_proj / np.sqrt(3)
    
    inner = []
    for c in centers_shifted:
        inner.append([c[0], c[1], 0.0])
        
    return inner, (0.0, 0.0), side_len, 0.0
# EVOLVE-BLOCK-END
