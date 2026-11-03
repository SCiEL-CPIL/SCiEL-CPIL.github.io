import math
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

# Pre-compute hexagon vertices for a unit hexagon centered at origin, angle 0
# A regular hexagon with side length 1 has vertices at distance 1 from center
# Vertices at angles 0, 60, 120, 180, 240, 300 degrees
HEX_VERTS = np.array([
    [math.cos(math.radians(a)), math.sin(math.radians(a))]
    for a in range(0, 360, 60)
])

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(q, r):
        """Convert axial coordinates to Cartesian for hex grid with flat-top orientation."""
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    def generate_hex_clusters(n_cells, max_span=4):
        """Generate all connected hex clusters of size n_cells using DFS."""
        clusters = set()
        
        def canonicalize(cells):
            min_q = min(q for q, r in cells)
            min_r = min(r for q, r in cells)
            shifted = frozenset((q - min_q, r - min_r) for q, r in cells)
            return shifted

        def dfs(cells, last):
            if len(cells) == n_cells:
                clusters.add(canonicalize(cells))
                return
            
            q, r = last
            # 6 hex neighbors in axial coordinates
            neighbors = [
                (q+1, r), (q-1, r),
                (q, r+1), (q, r-1),
                (q+1, r-1), (q-1, r+1)
            ]
            
            for nq, nr in neighbors:
                if (nq, nr) not in cells:
                    # Prune by bounding box
                    qs = [c[0] for c in cells] + [nq]
                    rs = [c[1] for c in cells] + [nr]
                    span_q = max(qs) - min(qs)
                    span_r = max(rs) - min(rs)
                    if span_q > max_span or span_r > max_span:
                        continue
                    dfs(cells | {(nq, nr)}, (nq, nr))
        
        dfs({(0, 0)}, (0, 0))
        return list(clusters)

    def get_cluster_points(cluster):
        """Get Cartesian coordinates of cluster centers."""
        pts = []
        for q, r in cluster:
            x, y = axial_to_cartesian(q, r)
            pts.append([x, y])
        return np.array(pts)

    def compute_all_vertices(pts, angle_deg):
        """Compute all vertices of all hexagons given centers and angle."""
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        all_verts = []
        for pt in pts:
            rotated_verts = (HEX_VERTS @ rot_matrix.T) + pt
            all_verts.append(rotated_verts)
        
        return np.vstack(all_verts)

    def enclosing_hexagon_side(vertices, angle_deg):
        """Compute minimum side length of enclosing hexagon with given orientation."""
        angle_rad = math.radians(angle_deg)
        
        # The 6 normal directions for the enclosing hexagon
        normals = []
        for k in range(6):
            ang = angle_rad + math.pi / 6.0 + k * math.pi / 3.0
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        # Project all vertices onto each normal
        projs = vertices @ normals.T  # shape (n_verts, 6)
        
        # For each normal, find max and min projection
        max_projs = np.max(projs, axis=0)
        min_projs = np.min(projs, axis=0)
        
        # The width in each direction is max - min
        # For a regular hexagon with apothem D, the width in direction n is 2*D
        # So D_i = (max_i - min_i) / 2
        D_values = (max_projs - min_projs) / 2.0
        D = np.max(D_values)
        
        # Side length S = (D + APOTHEM) / APOTHEM
        S = (D + APOTHEM) / APOTHEM
        return S

    def enclosing_hexagon_side_center(vertices, angle_deg):
        """Compute minimum side length and center of enclosing hexagon."""
        angle_rad = math.radians(angle_deg)
        
        normals = []
        for k in range(6):
            ang = angle_rad + math.pi / 6.0 + k * math.pi / 3.0
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        projs = vertices @ normals.T
        max_projs = np.max(projs, axis=0)
        min_projs = np.min(projs, axis=0)
        
        # The center of the enclosing hexagon is the midpoint of the projections
        # For each direction, the center projection is (max + min) / 2
        # We need to find a point (cx, cy) such that for all normals n:
        # n . (v - c) <= D for all v, and -n . (v - c) <= D for all v
        
        # Use LP to find optimal center and D
        from scipy.optimize import linprog
        
        n_verts = len(vertices)
        # Variables: cx, cy, D
        c_obj = np.array([0.0, 0.0, 1.0])
        
        A_ub = np.zeros((12 * n_verts, 3))
        b_ub = np.zeros(12 * n_verts)
        
        idx = 0
        for nx, ny in normals:
            for px, py in vertices:
                proj = px * nx + py * ny
                # n . (p - c) <= D  =>  -n . c - D <= -n . p
                A_ub[idx, 0] = -nx
                A_ub[idx, 1] = -ny
                A_ub[idx, 2] = -1.0
                b_ub[idx] = -proj
                idx += 1
                # -n . (p - c) <= D  =>  n . c - D <= n . p
                A_ub[idx, 0] = nx
                A_ub[idx, 1] = ny
                A_ub[idx, 2] = -1.0
                b_ub[idx] = proj
                idx += 1
        
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                cx, cy, D = res.x
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy)
            else:
                return float('inf'), None
        except Exception:
            return float('inf'), None

    # Generate candidate clusters
    clusters = generate_hex_clusters(N, max_span=4)
    
    # Score clusters by a quick estimate
    def quick_score(cluster):
        pts = get_cluster_points(cluster)
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        all_verts = compute_all_vertices(pts, 0.0)
        S = enclosing_hexagon_side(all_verts, 0.0)
        return S
    
    scored_clusters = [(quick_score(c), c) for c in clusters]
    scored_clusters.sort(key=lambda x: x[0])
    
    # Take top candidates
    K = min(50, len(scored_clusters))
    candidates = [c for _, c in scored_clusters[:K]]
    
    best_S = float('inf')
    best_pts = None
    best_angle = 0.0
    best_center = None
    
    for cluster in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        pts = get_cluster_points(cluster)
        
        # Optimize angle using Nelder-Mead
        def objective(angle_params):
            angle_deg = float(angle_params[0])
            all_verts = compute_all_vertices(pts, angle_deg)
            S, center = enclosing_hexagon_side_center(all_verts, angle_deg)
            return S
        
        # Try multiple starting angles
        best_local_S = float('inf')
        best_local_angle = 0.0
        best_local_center = None
        
        for start_angle in [0, 10, 20, 30, 40, 50]:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            try:
                res = minimize(objective, [start_angle], method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
                
                if res.fun < best_local_S:
                    best_local_S = res.fun
                    best_local_angle = float(res.x[0]) % 60.0
                    # Get center from the LP
                    all_verts = compute_all_vertices(pts, best_local_angle)
                    _, center = enclosing_hexagon_side_center(all_verts, best_local_angle)
                    if center is not None:
                        best_local_center = center
            except Exception:
                continue
        
        if best_local_S < best_S:
            best_S = best_local_S
            best_pts = pts
            best_angle = best_local_angle
            best_center = best_local_center
    
    # Fallback if nothing found
    if best_center is None:
        cluster = candidates[0]
        best_pts = get_cluster_points(cluster)
        best_angle = 0.0
        all_verts = compute_all_vertices(best_pts, best_angle)
        best_S, best_center = enclosing_hexagon_side_center(all_verts, best_angle)
        if best_center is None:
            best_center = (float(np.mean(best_pts[:, 0])), float(np.mean(best_pts[:, 1])))
            best_S = enclosing_hexagon_side(all_verts, best_angle)
    
    cx, cy = best_center
    angle_deg = best_angle % 60.0
    
    # Compute final vertices and verify
    all_verts = compute_all_vertices(best_pts, angle_deg)
    final_S, final_center = enclosing_hexagon_side_center(all_verts, angle_deg)
    
    if final_center is not None:
        cx, cy = final_center
        final_S = final_S
    
    # Build output
    inner = []
    for x, y in best_pts:
        inner.append([float(x), float(y), float(angle_deg)])
    
    center = [float(cx), float(cy)]
    side = float(final_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
