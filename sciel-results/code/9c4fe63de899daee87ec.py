import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Precompute unit hexagon vertices for rotation 0
    # Vertices at angles 0, 60, ..., 300
    base_vertices = np.array([
        [math.cos(math.radians(60 * k)), math.sin(math.radians(60 * k))] 
        for k in range(6)
    ])

    def get_hex_vertices(cx, cy, theta):
        """Get vertices of a unit hexagon centered at (cx, cy) with rotation theta."""
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        verts = base_vertices @ R.T + np.array([cx, cy])
        return verts

    def min_L_for_config(centers, thetas, outer_theta):
        """
        Compute the minimal side length L of a regular hexagon 
        with rotation outer_theta that encloses all given hexagons.
        """
        # Outer hexagon normal directions
        normal_angles = np.array([math.radians(30 + 60 * j + math.degrees(outer_theta)) for j in range(6)])
        normal_cos = np.cos(normal_angles)
        normal_sin = np.sin(normal_angles)
        
        max_proj = 0.0
        for i in range(len(centers)):
            verts = get_hex_vertices(centers[i][0], centers[i][1], thetas[i])
            # Project vertices onto normal directions
            # proj = v . n
            projs = verts[:, 0] * normal_cos + verts[:, 1] * normal_sin
            # We need the max projection relative to the center of the outer hexagon.
            # The outer hexagon is centered at (0,0) for calculation simplicity, 
            # but we can shift it. The minimal enclosing hexagon is centered such that
            # the max and min projections are balanced? 
            # Actually, for a fixed orientation, the minimal enclosing regular hexagon 
            # is determined by the max projection distance from the origin if we can translate it?
            # No, the outer hexagon can also translate.
            # Let's assume the outer hexagon is centered at (0,0). 
            # Then the constraint is that all vertices must be inside.
            # A point (x,y) is inside a regular hexagon of side L centered at origin with rotation phi
            # if max_j |proj_j(x,y)| <= L * sqrt(3) / 2.
            # So L >= max_j |proj_j(x,y)| / (sqrt(3)/2) for all points.
            # We want to minimize L. We can also choose the center of the outer hexagon.
            # To minimize L for a fixed orientation, the center of the outer hexagon should be 
            # the center of the bounding box in each normal direction.
            # i.e., for each normal j, let min_proj_j and max_proj_j be the min and max projections of all vertices.
            # The width in direction j is max_proj_j - min_proj_j.
            # The required L must satisfy L * sqrt(3) >= max_proj_j - min_proj_j for all j?
            # No. The distance from center to side is L * sqrt(3) / 2.
            # If the center is at C, then for normal j, the vertices must satisfy:
            # -L*sqrt(3)/2 <= proj_j(v - C) <= L*sqrt(3)/2
            # This implies proj_j(C) - L*sqrt(3)/2 <= proj_j(v) <= proj_j(C) + L*sqrt(3)/2
            # To minimize L, we can choose proj_j(C) to be the midpoint of min_proj_j and max_proj_j.
            # Then the required half-width is (max_proj_j - min_proj_j) / 2.
            # So L * sqrt(3) / 2 >= (max_proj_j - min_proj_j) / 2
            # L >= (max_proj_j - min_proj_j) / sqrt(3)
            
            # Let's compute min and max projections for all vertices of all hexagons.
            # We can do this outside the loop for efficiency.
            pass
            
        # Efficient calculation:
        all_verts = []
        for i in range(len(centers)):
            verts = get_hex_vertices(centers[i][0], centers[i][1], thetas[i])
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        # Projections onto 6 normals
        projs = all_verts[:, 0] * normal_cos[np.newaxis, :] + all_verts[:, 1] * normal_sin[np.newaxis, :]
        min_projs = np.min(projs, axis=0)
        max_projs = np.max(projs, axis=0)
        
        widths = max_projs - min_projs
        max_width = np.max(widths)
        L = max_width / SQRT3
        return L

    def check_overlap(centers, thetas):
        """Check if any two hexagons overlap."""
        verts_list = []
        for i in range(len(centers)):
            verts_list.append(get_hex_vertices(centers[i][0], centers[i][1], thetas[i]))
        
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                if polygons_overlap(verts_list[i], verts_list[j]):
                    return True
        return False

    def polygons_overlap(poly1, poly2):
        """Check if two convex polygons overlap using Separating Axis Theorem."""
        # Get all edge normals
        axes = []
        for poly in (poly1, poly2):
            for k in range(len(poly)):
                p1 = poly[k]
                p2 = poly[(k + 1) % len(poly)]
                edge = p2 - p1
                # Normal is perpendicular to edge
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-9:
                    axes.append(normal / norm)
        
        for axis in axes:
            # Project poly1 and poly2 onto axis
            proj1 = np.dot(poly1, axis)
            proj2 = np.dot(poly2, axis)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for gap
            if max1 < min2 or max2 < min1:
                return False
        return True

    def objective(vars):
        # Unpack
        centers = [(vars[i * 3], vars[i * 3 + 1]) for i in range(N)]
        thetas = [vars[i * 3 + 2] for i in range(N)]
        outer_theta = vars[N * 3]
        
        # Check overlap
        if check_overlap(centers, thetas):
            return 1000.0  # Large penalty
        
        L = min_L_for_config(centers, thetas, outer_theta)
        return L

    # Initial guesses
    best_L = float('inf')
    best_vars = None
    
    # We expect L to be around 3-4.
    # Start with a grid of initial positions.
    
    # Try a few random restarts
    np.random.seed(42)
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Generate random initial positions
        # Place them roughly in a hexagonal pattern
        # or just randomly in a box
        centers_init = []
        thetas_init = []
        
        # Try to place them in a side-3 hexagon pattern initially, then perturb
        # This is a good heuristic
        # Centers of side-3 hexagon
        init_centers = []
        for q in range(-2, 3):
            for rr in range(-2, 3):
                if abs(q + rr) <= 2:
                    x = 1.5 * q
                    y = SQRT3 * (rr + q / 2.0)
                    init_centers.append((x, y))
        
        # Pick 12 random centers from the 19
        indices = np.random.choice(len(init_centers), N, replace=False)
        for idx in indices:
            cx, cy = init_centers[idx]
            # Add small noise
            cx += np.random.uniform(-0.1, 0.1)
            cy += np.random.uniform(-0.1, 0.1)
            centers_init.append((cx, cy))
            thetas_init.append(np.random.uniform(0, 2 * math.pi))
        
        outer_theta_init = 0.0
        
        vars_init = []
        for i in range(N):
            vars_init.append(centers_init[i][0])
            vars_init.append(centers_init[i][1])
            vars_init.append(thetas_init[i])
        vars_init.append(outer_theta_init)
        
        # Optimize
        try:
            res = minimize(objective, vars_init, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < best_L and res.fun < 100.0:
                # Verify validity
                centers = [(res.x[i * 3], res.x[i * 3 + 1]) for i in range(N)]
                thetas = [res.x[i * 3 + 2] for i in range(N)]
                if not check_overlap(centers, thetas):
                    best_L = res.fun
                    best_vars = res.x
        except Exception as e:
            pass

    if best_vars is None:
        # Fallback to seed-like solution
        # Use the grid centers
        init_centers = []
        for q in range(-2, 3):
            for rr in range(-2, 3):
                if abs(q + rr) <= 2:
                    x = 1.5 * q
                    y = SQRT3 * (rr + q / 2.0)
                    init_centers.append((x, y))
        
        # Take first 12
        centers = init_centers[:12]
        thetas = [0.0] * 12
        L = min_L_for_config(centers, thetas, 0.0)
        
        inner = [[cx, cy, 0.0] for cx, cy in centers]
        return inner, (0.0, 0.0), L, 0.0

    # Extract best solution
    centers = [(best_vars[i * 3], best_vars[i * 3 + 1]) for i in range(N)]
    thetas = [best_vars[i * 3 + 2] for i in range(N)]
    outer_theta = best_vars[N * 3]
    
    # The outer hexagon center should be the centroid of the bounding box?
    # No, the objective function assumed the outer hexagon is centered at (0,0) 
    # and calculated L based on widths.
    # The actual center of the minimal enclosing hexagon is the midpoint of the min/max projections.
    # We need to return the center of the outer hexagon.
    
    # Recalculate center
    normal_angles = np.array([math.radians(30 + 60 * j + math.degrees(outer_theta)) for j in range(6)])
    normal_cos = np.cos(normal_angles)
    normal_sin = np.sin(normal_angles)
    
    all_verts = []
    for i in range(N):
        verts = get_hex_vertices(centers[i][0], centers[i][1], thetas[i])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    projs = all_verts[:, 0] * normal_cos[np.newaxis, :] + all_verts[:, 1] * normal_sin[np.newaxis, :]
    min_projs = np.min(projs, axis=0)
    max_projs = np.max(projs, axis=0)
    
    # The center of the outer hexagon in the projection space is (min+max)/2
    # We need to find the point (cx, cy) such that its projections match these midpoints.
    # This is a linear system.
    # Let M be the matrix of normals (6x2).
    # We want M @ [cx, cy]^T = mid_projs
    # M is 6x2, mid_projs is 6x1. Least squares solution.
    M = np.column_stack((normal_cos, normal_sin))
    mid_projs = (min_projs + max_projs) / 2.0
    
    # Solve M @ center = mid_projs
    center, residuals, rank, s = np.linalg.lstsq(M, mid_projs, rcond=None)
    
    inner = [[centers[i][0], centers[i][1], math.degrees(thetas[i])] for i in range(N)]
    
    return inner, (center[0], center[1]), best_L, math.degrees(outer_theta)

