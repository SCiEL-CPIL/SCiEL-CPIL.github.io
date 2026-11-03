import math
import numpy as np
import time
import random

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Axial coordinate to Cartesian
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Generate a dense hexagonal cluster of N=11 hexagons
    # We use a "flower" pattern plus extensions
    base_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # center + 6 neighbors
        (2, 0), (-2, 0), (0, 2), (1, 2)  # 4 more to make 11
    ]
    
    # Alternative: a more compact 3x4 rectangle-like shape
    alt1 = [
        (0,0), (1,0), (2,0), (3,0),
        (0,1), (1,1), (2,1), (3,1),
        (0,2), (1,2), (2,2)
    ]
    
    # Another: cross shape
    alt2 = [
        (0,0),
        (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1),
        (2,0), (-2,0), (0,2), (0,-2)
    ]
    
    # Diamond
    alt3 = [
        (0,0),
        (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1),
        (2,1), (-2,1), (1,-1), (-1,-1)
    ]
    
    # Elongated
    alt4 = [
        (0,0), (1,0), (2,0), (3,0), (4,0),
        (0,1), (1,1), (2,1), (3,1),
        (1,2), (2,2)
    ]
    
    # Compact triangle
    alt5 = [
        (0,0), (1,0), (2,0), (3,0),
        (0,1), (1,1), (2,1),
        (0,2), (1,2),
        (0,3)
    ]
    
    candidates_axial = [base_axial, alt1, alt2, alt3, alt4, alt5]

    def compute_min_enclosing_hexagon(points, beta):
        """
        Given points and rotation angle beta of the enclosing hexagon,
        compute the minimal side length S such that all points fit inside.
        The enclosing hexagon is centered at (cx, cy) which we optimize.
        
        For a regular hexagon with apothem A and rotation beta, the constraints are:
        For each of 6 directions n_k (unit normals), |p · n_k - c · n_k| <= A
        where A = S * apothem_unit (apothem of unit hexagon = sqrt(3)/2)
        
        So S = (max_k max_i |p_i · n_k - c · n_k| + APOTHEM) / APOTHEM
        We minimize over c = (cx, cy).
        """
        n_points = len(points)
        pts = np.array(points)
        
        # 6 normal directions
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # For a given center (cx, cy), the required apothem is:
        # A_req = max over all k and i of |(px_i - cx)*nx_k + (py_i - cy)*ny_k|
        # S = (A_req + APOTHEM) / APOTHEM
        
        # To minimize A_req over (cx, cy), we use the fact that this is a convex problem.
        # The optimal center minimizes the maximum of linear functions, which is a LP.
        # But for speed, we use a simple approach: the optimal center is the center of the
        # minimal bounding hexagon, which can be found by noting that at optimum, at least
        # 3 constraints are tight.
        
        # Fast approach: use the centroid as initial guess, then do a few iterations
        # of minimizing the max projection.
        
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        
        # Iterate: compute max violation direction, move center toward reducing it
        for _ in range(50):
            max_proj = 0
            for nx, ny in normals:
                projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
                max_p = np.max(projs)
                min_p = np.min(projs)
                if max_p > max_proj:
                    max_proj = max_p
                    best_n = (nx, ny)
                    best_sign = 1
                if -min_p > max_proj:
                    max_proj = -min_p
                    best_n = (nx, ny)
                    best_sign = -1
            
            if max_proj < 1e-10:
                break
            
            # Move center slightly toward the worst point
            # The worst point is the one with max projection in direction best_n
            projs = (pts[:, 0] - cx) * best_n[0] + (pts[:, 1] - cy) * best_n[1]
            if best_sign == 1:
                idx = np.argmax(projs)
            else:
                idx = np.argmin(projs)
            
            # Move 10% toward that point
            cx += 0.1 * (pts[idx, 0] - cx)
            cy += 0.1 * (pts[idx, 1] - cy)
        
        # Final computation
        max_proj = 0
        for nx, ny in normals:
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        
        S = (max_proj + APOTHEM) / APOTHEM
        return S, (cx, cy), max_proj

    def optimize_beta(points):
        """Find the best rotation angle beta for the enclosing hexagon."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        # Search over beta in [0, pi/6)
        n_samples = 73
        for i in range(n_samples):
            beta = i * (math.pi / 6) / n_samples
            S, center, _ = compute_min_enclosing_hexagon(points, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine around best
        for i in range(21):
            beta = best_beta + (i - 10) * (math.pi / 6) / 73 / 10
            if beta < 0:
                beta += math.pi / 6
            if beta >= math.pi / 6:
                beta -= math.pi / 6
            S, center, _ = compute_min_enclosing_hexagon(points, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        return best_S, best_beta, best_center

    def force_directed_relaxation(initial_points, iterations=2000):
        """
        Relax the points using force-directed method to minimize the enclosing hexagon size.
        Points represent centers of unit hexagons.
        """
        pts = np.array(initial_points, dtype=float)
        n = len(pts)
        
        # We want to minimize the enclosing hexagon side length.
        # Use gradient-free approach: try random perturbations and keep if better.
        
        best_S, best_beta, best_center = optimize_beta(pts.tolist())
        best_pts = pts.copy()
        
        # Simulated annealing style
        T = 0.1
        for it in range(iterations):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Pick a random point and perturb it
            idx = random.randint(0, n - 1)
            dx = random.gauss(0, T)
            dy = random.gauss(0, T)
            
            new_pts = best_pts.copy()
            new_pts[idx, 0] += dx
            new_pts[idx, 1] += dy
            
            new_S, new_beta, new_center = optimize_beta(new_pts.tolist())
            
            if new_S < best_S:
                best_S = new_S
                best_pts = new_pts
                best_beta = new_beta
                best_center = new_center
                T = max(T * 0.99, 0.001)
            else:
                T = max(T * 0.98, 0.001)
        
        return best_S, best_beta, best_center, best_pts

    # Try each candidate pattern
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for axial_list in candidates_axial:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Convert to Cartesian
        pts = [axial_to_cartesian(q, r) for q, r in axial_list]
        pts_arr = np.array(pts)
        
        # First, get the direct enclosing hexagon
        S_direct, beta_direct, center_direct = optimize_beta(pts)
        
        if S_direct < best_global_S:
            best_global_S = S_direct
            best_global_pts = pts_arr.copy()
            best_global_beta = beta_direct
            best_global_center = center_direct
        
        # Then try force-directed relaxation
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            continue
        
        S_fd, beta_fd, center_fd, pts_fd = force_directed_relaxation(pts, iterations=1000)
        
        if S_fd < best_global_S:
            best_global_S = S_fd
            best_global_pts = pts_fd
            best_global_beta = beta_fd
            best_global_center = center_fd

    if best_global_center is None or best_global_S == float('inf'):
        # Fallback
        pts = [axial_to_cartesian(q, r) for q, r in candidates_axial[0]]
        best_global_pts = np.array(pts)
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        best_global_S = 5.0

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Compute final side length with small safety margin
    pts_scaled = best_global_pts
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    S_final = (max_proj + APOTHEM) / APOTHEM
    
    inner = []
    for i in range(len(pts_scaled)):
        inner.append([float(pts_scaled[i, 0]), float(pts_scaled[i, 1]), float(angle_deg)])
    
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
