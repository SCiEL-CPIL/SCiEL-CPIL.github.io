import numpy as np

N = 12

def construct():
    # Generate 12 hexagon centers
    # 1 center
    centers = [(0.0, 0.0)]
    
    # 6 Ring 1
    for k in range(6):
        angle = k * np.pi / 3
        centers.append((np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)))
        
    # 5 Ring 2 (consecutive)
    # Ring 2 angles are offset by 30 degrees from Ring 1?
    # Ring 1 angles: 0, 60, 120...
    # Ring 2 angles: 0, 30, 60...
    # Let's verify.
    # Center (0,0).
    # (sqrt(3), 0) is Ring 1.
    # (2sqrt(3), 0) is Ring 2.
    # (sqrt(3)/2, 1.5) is Ring 1.
    # (3sqrt(3)/2, 3) is Ring 2?
    # dist^2 = 27/4 + 9 = 27/4 + 36/4 = 63/4. sqrt(63)/2 = 3sqrt(7)/2 != 2sqrt(3).
    # 2sqrt(3) = sqrt(12) = sqrt(48/4).
    # So (3sqrt(3)/2, 3) is NOT Ring 2.
    
    # Correct Ring 2 generation:
    # The 19 centers are at (i, j) in axial coords.
    # Let's just use the polar coords of the 12 outer ring.
    # The 12 outer centers of the 19-hex cluster are at distance 2*sqrt(3).
    # Their angles are 0, 30, 60, ... 330.
    
    for k in range(5):
        angle = k * np.pi / 6
        centers.append((2 * np.sqrt(3) * np.cos(angle), 2 * np.sqrt(3) * np.sin(angle)))
        
    # We have 1 + 6 + 5 = 12 centers.
    
    # Generate vertices for each hexagon
    # Hexagon vertices relative to center:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    
    all_vertices = []
    for cx, cy in centers:
        for i in range(6):
            angle = i * np.pi / 3
            vx = cx + np.cos(angle)
            vy = cy + np.sin(angle)
            all_vertices.append((vx, vy))
            
    all_vertices = np.array(all_vertices)
    
    # Optimize rotation theta
    # L(theta) = (2/sqrt(3)) * max_{v, k} (v . n_k(theta))
    # n_k(theta) = (cos(theta + k*pi/3), sin(theta + k*pi/3))
    
    def calc_L(theta):
        max_proj = 0.0
        for k in range(6):
            nx = np.cos(theta + k * np.pi / 3)
            ny = np.sin(theta + k * np.pi / 3)
            projs = all_vertices[:, 0] * nx + all_vertices[:, 1] * ny
            max_proj = max(max_proj, np.max(projs))
        return (2.0 / np.sqrt(3.0)) * max_proj

    # Grid search for theta
    best_L = float('inf')
    best_theta = 0.0
    # 6-fold symmetry, so 0 to pi/3
    thetas = np.linspace(0, np.pi/3, 1000)
    for th in thetas:
        L = calc_L(th)
        if L < best_L:
            best_L = L
            best_theta = th
            
    # Refine with ternary search? Grid might be enough.
    # Let's do a finer grid around best_theta
    thetas = np.linspace(best_theta - 0.01, best_theta + 0.01, 1000)
    for th in thetas:
        L = calc_L(th)
        if L < best_L:
            best_L = L
            best_theta = th
            
    # Construct inner hexagons
    inner = []
    for cx, cy in centers:
        # Rotate centers by -best_theta?
        # The container is rotated by best_theta.
        # The cluster is fixed.
        # So we return the cluster as is, and the container rotated by best_theta.
        # The interface asks for inner (x,y,angle) and outer (center, side, angle).
        # Inner angle is 0 (standard orientation).
        inner.append([cx, cy, 0.0])
        
    # Outer hexagon
    # Center (0,0)
    # Side best_L
    # Angle best_theta
    
    return inner, (0.0, 0.0), best_L, np.degrees(best_theta)
