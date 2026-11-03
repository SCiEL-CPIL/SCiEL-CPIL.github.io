import math
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # We will optimize the positions of the 11 hexagons and the outer hexagon parameters.
    # Variables:
    # - 11 centers (x, y) -> 22 vars
    # - Outer center (cx, cy) -> 2 vars
    # - Outer side length S -> 1 var
    # - Outer rotation angle theta -> 1 var
    # Total: 26 variables
    
    # However, fixing the symmetry might be better. 
    # Let's assume the outer hexagon is centered at (0,0) and has rotation 0.
    # This reduces variables to 22 (positions) + 1 (S) = 23 variables.
    # The inner hexagons can rotate. Since they are regular, rotation doesn't change the set of vertices, 
    # but it changes the orientation relative to the grid. 
    # Wait, the inner hexagons are "disjoint". If we allow rotation, the packing might be denser.
    # However, for regular hexagons of the same size, the optimal packing is usually aligned or with specific angles.
    # Given the complexity, let's first try to find a good alignment where all inner hexagons have the same orientation as the outer one (or 0 degrees).
    # Actually, the problem allows rotation. But let's start with a strong heuristic: 
    # Assume all inner hexagons are aligned (angle=0) and optimize their centers and the outer size S.
    # If this yields a good result, we can try to add rotations later if time permits.
    
    # Let's define the vertices of a unit hexagon centered at origin, aligned with axes (flat top/bottom? or pointy top?)
    # Standard regular hexagon with side 1:
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees.
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    
    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    def check_containment(vertices, outer_center, outer_side, outer_angle_deg):
        # Check if all vertices are inside the outer hexagon
        # Outer hexagon is defined by center, side S, angle theta.
        # The apothem of the outer hexagon is A_out = S * sqrt(3) / 2
        # The distance from center to each side must be <= A_out
        # The sides are at angles theta + 30 + k*60
        cx, cy = outer_center
        A_out = outer_side * math.sqrt(3) / 2
        theta_rad = math.radians(outer_angle_deg)
        
        for vx, vy in vertices:
            # Vector from center
            dx = vx - cx
            dy = vy - cy
            # Project onto normal of each side
            for k in range(6):
                normal_angle = theta_rad + math.pi / 6 + k * math.pi / 3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                dist = abs(dx * nx + dy * ny)
                if dist > A_out + 1e-9:
                    return False
        return True

    def check_disjointness(centers, angles, margin=1e-7):
        # Two hexagons of side 1 are disjoint if the distance between centers is >= 2 * apothem?
        # No. The minimum distance between two disjoint regular hexagons of side 1 depends on orientation.
        # If aligned, the minimum distance is 2 * apothem = sqrt(3) ~ 1.732.
        # If rotated 30 degrees, the minimum distance is 2 * radius = 2.0? No.
        # The distance between centers must be at least the sum of the "reach" in the direction of the center-to-center vector.
        # This is complex. A simpler check: compute the convex hull intersection or polygon intersection.
        # For speed, we can use a conservative bound: distance > 2 * sqrt(3)/2 * 2? No.
        # The maximum distance from center to vertex is 1.0.
        # So if distance > 2.0, they are definitely disjoint.
        # If distance < sqrt(3) ~ 1.732, they might overlap.
        # Let's use a polygon intersection test if distance < 2.0.
        
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                xi, yi = centers[i]
                xj, yj = centers[j]
                dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                if dist >= 2.0:
                    continue
                # Check polygon intersection
                v1 = hex_vertices(xi, yi, angles[i])
                v2 = hex_vertices(xj, yj, angles[j])
                if polygons_intersect(v1, v2):
                    return False
        return True

    def polygons_intersect(poly1, poly2):
        # Separating Axis Theorem
        def get_edges(poly):
            edges = []
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                edges.append((x2 - x1, y2 - y1))
            return edges

        def project(poly, axis):
            min_val = float('inf')
            max_val = -float('inf')
            for x, y in poly:
                p = x * axis[0] + y * axis[1]
                if p < min_val: min_val = p
                if p > max_val: max_val = p
            return min_val, max_val

        edges1 = get_edges(poly1)
        edges2 = get_edges(poly2)
        
        for edge in edges1 + edges2:
            # Normal to edge
            nx = -edge[1]
            ny = edge[0]
            # Normalize
            length = math.sqrt(nx*nx + ny*ny)
            if length < 1e-9: continue
            nx /= length
            ny /= length
            
            min1, max1 = project(poly1, (nx, ny))
            min2, max2 = project(poly2, (nx, ny))
            
            if max1 < min2 or max2 < min1:
                return False
        return True

    # Objective function for optimization
    # We want to minimize S.
    # We must satisfy:
    # 1. All inner hexagons are inside the outer hexagon.
    # 2. All inner hexagons are disjoint.
    
    def objective(params):
        # params: [x1, y1, a1, x2, y2, a2, ..., x11, y11, a11, cx, cy, S, theta]
        # Let's fix cx=0, cy=0, theta=0 for simplicity first.
        # Then params are [x1, y1, a1, ..., x11, y11, a11, S]
        # 33 + 1 = 34 variables.
        
        # Extract centers and angles
        centers = []
        angles = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            centers.append((x, y))
            angles.append(a)
        
        S = params[N * 3]
        
        # Check disjointness
        if not check_disjointness(centers, angles):
            return 1e6
        
        # Check containment
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        if not check_containment([v for c, a in zip(centers, angles) for v in hex_vertices(c[0], c[1], a)], outer_center, S, outer_angle):
            return 1e6
        
        return S

    # Initial guess: Place hexagons in a honeycomb pattern.
    # Honeycomb spacing: sqrt(3) in one direction, 1.5 in the other?
    # Axial to cartesian:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    # Let's generate a few initial configurations.
    # Config 1: 3x3 grid centered
    init_configs = []
    
    # 3x3 grid of hexagons
    # q in [-1, 0, 1], r in [-1, 0, 1]
    # This gives 9 hexagons. We need 11.
    # Add (2, 0) and (-2, 0)?
    
    # Let's use the axial coordinates from the previous program's best candidates as initial guesses.
    # But we need to convert them to the parameter vector format.
    
    # Let's just run differential evolution with a broad range.
    # Bounds:
    # x, y: [-3, 3]
    # angle: [0, 60] (degrees)
    # S: [3, 6]
    
    bounds = []
    for i in range(N):
        bounds.append((-3, 3)) # x
        bounds.append((-3, 3)) # y
        bounds.append((0, 60)) # angle
    bounds.append((3, 6)) # S
    
    # Differential Evolution is robust for global optimization.
    try:
        result = differential_evolution(objective, bounds, maxiter=1000, popsize=15, tol=1e-6, mutation=(0.5, 1.5), recombination=0.7, seed=42, polish=False, init='latinhypercube')
        
        if result.success:
            best_params = result.x
            best_S = result.fun
        else:
            # Fallback to a simple grid
            best_S = 5.0
            best_params = [0.0] * (N * 3 + 1)
            # Place in a rough grid
            for i in range(N):
                q = (i % 3) - 1
                r = (i // 3) - 1
                x = math.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                best_params[i * 3] = x
                best_params[i * 3 + 1] = y
                best_params[i * 3 + 2] = 0.0
            best_params[N * 3] = 5.0
            
    except Exception:
        best_S = 5.0
        best_params = [0.0] * (N * 3 + 1)
        for i in range(N):
            q = (i % 3) - 1
            r = (i // 3) - 1
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            best_params[i * 3] = x
            best_params[i * 3 + 1] = y
            best_params[i * 3 + 2] = 0.0
        best_params[N * 3] = 5.0

    # Refine with local optimization (Nelder-Mead)
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            result_ref = minimize(objective, best_params, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            if result_ref.fun < best_S:
                best_S = result_ref.fun
                best_params = result_ref.x
    except Exception:
        pass

    # Extract final parameters
    centers = []
    angles = []
    for i in range(N):
        x = best_params[i * 3]
        y = best_params[i * 3 + 1]
        a = best_params[i * 3 + 2]
        centers.append((x, y))
        angles.append(a)
    
    S = best_params[N * 3]
    
    # Verify validity
    if not check_disjointness(centers, angles):
        # Fallback to a known valid configuration
        # Simple 3x3 + 2
        centers = []
        angles = []
        axial = [
            (0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (-2, 0), (0, 2), (0, -2)
        ]
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            centers.append((x, y))
            angles.append(0.0)
        
        # Calculate S for this configuration
        max_proj = 0
        for cx, cy in centers:
            for vx, vy in hex_vertices(cx, cy, 0.0):
                for k in range(6):
                    ang = math.pi / 6.0 + k * math.pi / 3.0
                    nx = math.cos(ang)
                    ny = math.sin(ang)
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        S = (max_proj + APOTHEM) / APOTHEM
        best_S = S

    if not check_containment([v for c, a in zip(centers, angles) for v in hex_vertices(c[0], c[1], a)], (0.0, 0.0), S, 0.0):
        # Expand S if necessary
        max_proj = 0
        for cx, cy in centers:
            for vx, vy in hex_vertices(cx, cy, 0.0):
                for k in range(6):
                    ang = math.pi / 6.0 + k * math.pi / 3.0
                    nx = math.cos(ang)
                    ny = math.sin(ang)
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        S = (max_proj + APOTHEM) / APOTHEM

    inner = []
    for (x, y), a in zip(centers, angles):
        inner.append([float(x), float(y), float(a)])
    
    center = [0.0, 0.0]
    side = float(S)
    rotation = 0.0
    
    return inner, center, side, rotation
