import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 140.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: simple grid
        pts = []
        for i in range(4):
            for j in range(3):
                if i+j < 4:
                    q = i
                    r = j
                    x = math.sqrt(3) * (q + r / 2.0)
                    y = 1.5 * r
                    pts.append((x, y))
        if len(pts) < N:
            # Fill remaining
            for i in range(5):
                for j in range(3):
                    if len(pts) >= N: break
                    q = i
                    r = j
                    x = math.sqrt(3) * (q + r / 2.0)
                    y = 1.5 * r
                    if (x, y) not in pts:
                        pts.append((x, y))
        pts = np.array(pts[:N])
        # Center
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        S = (max_proj + APOTHEM) / APOTHEM
        inner = [[float(x), float(y), 0.0] for x, y in pts]
        return inner, [float(cx), float(cy)], float(S), 0.0

    # Precompute rotation matrices for 6 orientations? No, we fix orientation to 0 for simplicity in this first pass
    # The prompt allows rotation, but fixed orientation is a valid subset.
    # We will optimize positions.
    
    # Objective function
    def objective(params):
        # params: 2*N + 1 (x1..xN, y1..yN, S)
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        
        pts = params[:2*N].reshape(N, 2)
        S = params[2*N]
        
        # Check overlap
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = np.sum((pts[i] - pts[j])**2)
                if dist_sq < (2.0 - 1e-9)**2:
                    return 1e10 + (2.0 - math.sqrt(dist_sq)) # Penalty for overlap
            
        # Check containment
        # Outer hexagon centered at (0,0) for simplicity? 
        # Actually, we can fix the center of the outer hexagon to (0,0) by translating pts.
        # But the outer hexagon can be rotated. 
        # To keep it simple and fast, let's assume outer hexagon is axis-aligned (rotation 0) 
        # and centered at (0,0). We can always translate the whole configuration.
        # So we minimize S such that all pts are inside hexagon of side S centered at origin.
        
        # Hexagon constraints: |proj| <= apothem_outer
        # apothem_outer = S * sqrt(3) / 2
        apothem_req = S * APOTHEM
        
        # Normals for axis aligned hexagon (flat top)
        # Angles: 30, 90, 150, 210, 270, 330 degrees
        # Or simpler: 
        # x-axis: |x| <= S
        # y-axis: |y| <= S * sqrt(3) / 2 ? No.
        
        # Regular hexagon with side S, flat top (vertex at 0, S*sqrt(3)/2? No, flat top means horizontal edges).
        # Vertices at (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
        # Wait, if side is S, distance from center to vertex is S.
        # Apothem is S * sqrt(3) / 2.
        # If flat top, the horizontal extent is from -S to S? No, vertices at (S,0) and (-S,0). So width 2S.
        # Vertical extent: y goes from -S*sqrt(3)/2 to S*sqrt(3)/2.
        # Constraints:
        # 1. |x| <= S
        # 2. |y| <= S * sqrt(3) / 2
        # 3. |x + y/sqrt(3)| <= S ? No.
        # Let's use the support function approach.
        # h(n) <= apothem
        # Normals n_k for k=0..5
        # n_0 = (1, 0) -> |x| <= apothem? No, if vertex is at (S,0), normal is (1,0), support is S.
        # So constraint is n . p <= S.
        # Wait, for a regular hexagon with side S, the distance from center to vertex is S.
        # The distance from center to edge (apothem) is S * sqrt(3)/2.
        # The support function in direction of vertex normal is S.
        # The support function in direction of edge normal is S * sqrt(3)/2.
        
        # Let's stick to one specific orientation for the outer hexagon to avoid non-convexity in S.
        # If we allow rotation, it becomes non-convex. 
        # Given time limits, we will fix outer hexagon rotation to 0 degrees (flat top).
        # And we will center the inner hexagons around (0,0) by subtracting their centroid? 
        # No, we can just let the optimizer find positions. 
        # To reduce variables, we can fix the centroid of the inner hexagons to (0,0).
        # But that's an extra constraint. 
        # Simpler: Just optimize positions and S.
        
        # Constraints for outer hexagon (flat top, side S, centered at 0,0):
        # Vertices at (S, 0), (S/2, S*sqrt(3)/2), ...
        # Edges are perpendicular to angles 0, 60, 120, 180, 240, 300.
        # Wait, if vertex is at angle 0, edge normal is at 30?
        # Let's check: Vertex (S, 0). Edge connects (S,0) and (S/2, S*sqrt(3)/2).
        # Midpoint: (3S/4, S*sqrt(3)/4). Normal direction: (1, 1/sqrt(3))?
        # Actually, for a regular hexagon, the normals to the edges are at 30, 90, 150, 210, 270, 330 degrees.
        # The distance from center to each edge is the apothem A = S * sqrt(3) / 2.
        # So constraints: |p . n_k| <= A for k=0..5.
        
        A = S * APOTHEM
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            for i in range(N):
                proj = pts[i, 0] * nx + pts[i, 1] * ny
                if proj > A + 1e-9:
                    return 1e10 + (proj - A)
                if -proj > A + 1e-9:
                    return 1e10 + (-proj - A)
        
        return S

    # Initial guesses
    best_S = float('inf')
    best_pts = None
    
    # Generate several initial configurations
    # 1. Hexagonal lattice
    # 2. Random
    
    configs = []
    
    # Config 1: Cluster
    pts = []
    used = set()
    neighbors = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    pts.append((0,0))
    used.add((0,0))
    for _ in range(N-1):
        base = pts[-1]
        random.shuffle(neighbors)
        for d in neighbors:
            new_pt = (base[0] + d[0], base[1] + d[1])
            if new_pt not in used:
                pts.append(new_pt)
                used.add(new_pt)
                break
    axial_to_cart = lambda q, r: (math.sqrt(3) * (q + r/2.0), 1.5 * r)
    pts_cart = np.array([axial_to_cart(q, r) for q, r in pts])
    configs.append(pts_cart)
    
    # Config 2: Grid
    pts = []
    for i in range(4):
        for j in range(3):
            if i+j < 4:
                pts.append((i, j))
    while len(pts) < N:
        i = len(pts) // 3
        j = len(pts) % 3
        pts.append((i, j))
    pts_cart = np.array([axial_to_cart(q, r) for q, r in pts[:N]])
    configs.append(pts_cart)
    
    # Config 3: Random
    np.random.seed(42)
    pts_cart = np.random.uniform(-2, 2, (N, 2))
    configs.append(pts_cart)
    
    for init_pts in configs:
        # Center init
        init_pts = init_pts - np.mean(init_pts, axis=0)
        
        # Initial S estimate
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = init_pts[:, 0] * nx + init_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        init_S = (max_proj + APOTHEM) / APOTHEM + 0.5
        
        x0 = np.concatenate([init_pts.flatten(), [init_S]])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_S and res.fun < 100:
                best_S = res.fun
                best_pts = res.x[:2*N].reshape(N, 2)
        except Exception:
            pass
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break

    if best_pts is None:
        # Fallback to first config
        best_pts = configs[0] - np.mean(configs[0], axis=0)
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_pts[:, 0] * nx + best_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_S = (max_proj + APOTHEM) / APOTHEM

    # Final polish with higher precision
    x0 = np.concatenate([best_pts.flatten(), [best_S]])
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < best_S:
            best_S = res.fun
            best_pts = res.x[:2*N].reshape(N, 2)
    except Exception:
        pass

    # Return format
    # Inner: rows of (x, y, angle_degrees)
    # Center: (x, y) of outer hexagon (we assumed 0,0)
    # Side: S
    # Rotation: 0.0
    
    inner = []
    for i in range(N):
        inner.append([float(best_pts[i, 0]), float(best_pts[i, 1]), 0.0])
    
    center = [0.0, 0.0]
    side = float(best_S)
    rotation = 0.0
    
    return inner, center, side, rotation

