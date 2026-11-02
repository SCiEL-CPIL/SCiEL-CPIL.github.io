import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Initial configuration: 11 hexagons in a hexagonal cluster
    # Using axial coordinates for initial placement
    axial_init = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    
    # Convert axial to Cartesian
    def axial_to_cartesian(axial):
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)
    
    pts_init = axial_to_cartesian(axial_init)
    
    # Compute initial center
    cx_init = float(np.mean(pts_init[:, 0]))
    cy_init = float(np.mean(pts_init[:, 1]))
    
    # Parameter vector: [x1, y1, theta1, x2, y2, theta2, ..., x11, y11, theta11, cx, cy, beta, S]
    # Total: 11*3 + 4 = 37 parameters
    
    def get_positions(params):
        """Extract positions and orientations from parameters."""
        positions = params[:N*3].reshape(N, 3)
        cx = params[N*3]
        cy = params[N*3+1]
        beta = params[N*3+2]
        S = params[N*3+3]
        return positions, cx, cy, beta, S
    
    def hexagon_vertices(cx, cy, side, angle):
        """Get vertices of a regular hexagon centered at (cx, cy) with given side and angle."""
        vertices = []
        for k in range(6):
            ang = angle + k * math.pi / 3.0
            x = cx + side * math.cos(ang)
            y = cy + side * math.sin(ang)
            vertices.append((x, y))
        return np.array(vertices)
    
    def hexagon_normals(angle):
        """Get outward normals of a regular hexagon with given angle."""
        normals = []
        for k in range(6):
            ang = angle + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        return np.array(normals)
    
    def minkowski_sum_constraint(pos1, theta1, pos2, theta2, epsilon=1e-6):
        """
        Compute the minimum separation between two hexagons using Minkowski sum.
        Returns a scalar constraint value (>=0 means non-overlapping).
        """
        # For two convex polygons, they don't overlap iff there exists a separating axis
        # For hexagons, we check all 12 edge normals (6 from each)
        
        normals1 = hexagon_normals(theta1)
        normals2 = hexagon_normals(theta2)
        
        # All candidate separating axes
        all_normals = np.vstack([normals1, normals2])
        
        # For each normal, compute the gap
        min_gap = float('inf')
        for nx, ny in all_normals:
            # Project both hexagons onto this axis
            verts1 = hexagon_vertices(pos1[0], pos1[1], 1.0, theta1)
            verts2 = hexagon_vertices(pos2[0], pos2[1], 1.0, theta2)
            
            proj1 = verts1[:, 0] * nx + verts1[:, 1] * ny
            proj2 = verts2[:, 0] * nx + verts2[:, 1] * ny
            
            # Gap is the distance between the projections
            gap = max(np.min(proj2) - np.max(proj1), np.min(proj1) - np.max(proj2))
            min_gap = min(min_gap, gap)
        
        return min_gap
    
    def objective(params):
        """Minimize outer hexagon side length S."""
        positions, cx, cy, beta, S = get_positions(params)
        
        # Check outer containment: all inner hexagons must fit inside outer hexagon
        outer_verts = hexagon_vertices(cx, cy, S, beta)
        outer_normals = hexagon_normals(beta)
        
        # For each inner hexagon, check if all its vertices are inside the outer hexagon
        for i in range(N):
            pos_i = positions[i]
            theta_i = positions[i, 2]
            inner_verts = hexagon_vertices(pos_i[0], pos_i[1], 1.0, theta_i)
            
            # Check each vertex against each outer edge
            for vx, vy in inner_verts:
                for nx, ny in outer_normals:
                    # Distance from center to edge along normal
                    edge_dist = S * APOTHEM
                    # Projection of vertex onto normal
                    proj = (vx - cx) * nx + (vy - cy) * ny
                    # Vertex must be within edge distance
                    if proj > edge_dist + 1e-9:
                        return S + 1000.0 * (proj - edge_dist)
        
        # Check non-overlap between all pairs of inner hexagons
        for i in range(N):
            for j in range(i+1, N):
                gap = minkowski_sum_constraint(
                    positions[i, :2], positions[i, 2],
                    positions[j, :2], positions[j, 2]
                )
                if gap < 1e-6:
                    return S + 1000.0 * (1e-6 - gap)
        
        return S
    
    def gradient_approx(params, eps=1e-5):
        """Approximate gradient using finite differences."""
        grad = np.zeros_like(params)
        f0 = objective(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            f_plus = objective(params_plus)
            grad[i] = (f_plus - f0) / eps
        return grad
    
    # Initial parameters
    params_init = np.zeros(N*3 + 4)
    for i in range(N):
        params_init[i*3] = pts_init[i, 0]
        params_init[i*3+1] = pts_init[i, 1]
        params_init[i*3+2] = 0.0  # initial orientation
    params_init[N*3] = cx_init
    params_init[N*3+1] = cy_init
    params_init[N*3+2] = 0.0  # outer rotation
    params_init[N*3+3] = 4.0  # initial outer side length estimate
    
    # Optimize using Nelder-Mead
    best_S = float('inf')
    best_params = params_init.copy()
    
    start_time = time.time()
    max_iter = 5000
    
    # First, do a coarse optimization
    res = minimize(objective, params_init, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
    
    if res.fun < best_S:
        best_S = res.fun
        best_params = res.x
    
    # Refine with more iterations if time permits
    if time.time() - start_time < TIME_LIMIT - START_TIME - SAFETY_MARGIN:
        res2 = minimize(objective, best_params, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False})
        if res2.fun < best_S:
            best_S = res2.fun
            best_params = res2.x
    
    # Extract final solution
    positions, cx, cy, beta, S = get_positions(best_params)
    
    # Verify and compute actual S
    max_proj = 0
    outer_normals = hexagon_normals(beta)
    for i in range(N):
        pos_i = positions[i]
        theta_i = positions[i, 2]
        inner_verts = hexagon_vertices(pos_i[0], pos_i[1], 1.0, theta_i)
        for vx, vy in inner_verts:
            for nx, ny in outer_normals:
                proj = (vx - cx) * nx + (vy - cy) * ny
                max_proj = max(max_proj, abs(proj))
    
    # S is determined by max_proj / APOTHEM
    S_actual = max_proj / APOTHEM if max_proj > 0 else 1.0
    
    # Ensure S_actual is at least the minimum needed
    S_final = max(S_actual, S)
    
    # Build inner list
    inner = []
    for i in range(N):
        x = float(positions[i, 0])
        y = float(positions[i, 1])
        angle_deg = float(math.degrees(positions[i, 2]))
        inner.append([x, y, angle_deg])
    
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(math.degrees(beta))
    
    return inner, center, side, rotation
