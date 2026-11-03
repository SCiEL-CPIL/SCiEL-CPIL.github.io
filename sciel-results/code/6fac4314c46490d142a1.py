import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # --- Hexagon geometry helpers ---
    
    def hex_vertices(cx, cy, angle_deg, side=1.0):
        """Return 6 vertices of a regular hexagon centered at (cx,cy) with given angle (degrees)."""
        verts = []
        for k in range(6):
            ang = math.radians(angle_deg) + k * math.pi / 3.0
            vx = cx + side * math.cos(ang)
            vy = cy + side * math.sin(ang)
            verts.append((vx, vy))
        return np.array(verts)
    
    def hex_apothem_norms(angle_deg):
        """Return 6 unit normal vectors to the sides of a regular hexagon."""
        normals = []
        base = math.radians(angle_deg) + math.pi / 6.0
        for k in range(6):
            ang = base + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        return np.array(normals)
    
    def all_vertices_of_hexes(positions, angles):
        """Given list of (x,y,angle) for unit hexagons, return all 6n vertices."""
        verts = []
        for (x, y, a) in positions:
            v = hex_vertices(x, y, a)
            verts.extend(v)
        return np.array(verts)
    
    # --- Core LP: find minimum enclosing hexagon for fixed vertices and rotation ---
    
    def min_enclosing_hexagon(verts, beta):
        """
        Given fixed vertices and outer hexagon rotation beta (radians),
        find optimal center and apothem via LP.
        Returns (S, center, apothem) where S = (D+APOTHEM)/APOTHEM.
        """
        normals = hex_apothem_norms(math.degrees(beta))
        
        n = len(verts)
        # Variables: cx, cy, D
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for (nx, ny) in normals:
            for px, py in verts:
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0.0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                cx, cy, D = res.x
                S = (D + APOTHEM) / APOTHEM
                return S, (float(cx), float(cy)), float(D)
            else:
                return float('inf'), (0.0, 0.0), float('inf')
        except Exception:
            return float('inf'), (0.0, 0.0), float('inf')
    
    # --- Disjointness check: all 6n hexagons must be pairwise disjoint ---
    
    def are_disjoint(positions, angles, eps=1e-9):
        """Check that all hexagons are pairwise disjoint (no overlap, may touch)."""
        n = len(positions)
        verts_list = []
        for i in range(n):
            verts_list.append(hex_vertices(positions[i][0], positions[i][1], angles[i]))
        
        for i in range(n):
            for j in range(i + 1, n):
                if hexes_overlap(verts_list[i], verts_list[j], eps):
                    return False
        return True
    
    def hexes_overlap(v1, v2, eps=1e-9):
        """Check if two convex hexagons overlap (interior intersection)."""
        # Separating axis theorem: for convex polygons, if there exists an axis
        # where projections don't overlap, they are disjoint.
        # Try all edges of both polygons as axes.
        for verts in [v1, v2]:
            for k in range(6):
                # Edge normal
                p1 = verts[k]
                p2 = verts[(k + 1) % 6]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                # Normal to edge
                nx = -dy
                ny = dx
                norm = math.sqrt(nx*nx + ny*ny)
                if norm < 1e-15:
                    continue
                nx /= norm
                ny /= norm
                
                proj1 = v1 @ np.array([nx, ny])
                proj2 = v2 @ np.array([nx, ny])
                
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                
                if max1 <= min2 + eps or max2 <= min1 + eps:
                    return False  # Separated on this axis
        
        # If no separating axis found, check if one is inside the other
        # For convex hexagons, if no edge separates them, they overlap
        # But we need to be careful: touching counts as non-overlapping
        # Let's check if any vertex of one is strictly inside the other
        for vi in v1:
            if point_strictly_inside_hex(vi, v2, eps):
                return True
        for vi in v2:
            if point_strictly_inside_hex(vi, v1, eps):
                return True
        return True  # No separation axis found => overlap
    
    def point_strictly_inside_hex(p, verts, eps=1e-9):
        """Check if point p is strictly inside convex hexagon verts."""
        # Use cross products: point is inside if it's on the same side of all edges
        signs = []
        for k in range(6):
            p1 = verts[k]
            p2 = verts[(k + 1) % 6]
            cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
            signs.append(cross)
        
        # All must be positive or all negative (strictly)
        if all(s > eps for s in signs):
            return True
        if all(s < -eps for s in signs):
            return True
        return False
    
    # --- Main optimization: Sequential Convex Optimization ---
    
    def optimize_sequential(initial_positions, initial_angles, time_budget):
        """
        Sequential convex optimization:
        - Start with initial positions/angles
        - Iteratively: fix all angles, solve LP for positions (via relaxation), 
          then refine each angle individually
        """
        n = len(initial_positions)
        positions = list(initial_positions)
        angles = list(initial_angles)
        
        best_S = float('inf')
        best_positions = None
        best_angles = None
        
        def evaluate(current_positions, current_angles):
            """Evaluate current configuration: check disjointness, compute S."""
            if not are_disjoint(current_positions, current_angles):
                return float('inf'), None, None
            
            all_verts = all_vertices_of_hexes(current_positions, current_angles)
            
            # Try multiple outer rotations
            best_S = float('inf')
            best_center = None
            best_beta = 0.0
            for i in range(12):
                beta = i * math.pi / 6.0 / 11.0
                S, center, D = min_enclosing_hexagon(all_verts, beta)
                if S < best_S:
                    best_S = S
                    best_center = center
                    best_beta = beta
            
            # Refine beta
            try:
                res = minimize_scalar(
                    lambda b: min_enclosing_hexagon(all_verts, b)[0],
                    bounds=(0, math.pi / 6), method='bounded',
                    options={'maxiter': 100, 'xatol': 1e-10}
                )
                if res.fun < best_S:
                    best_S = res.fun
                    _, best_center, _ = min_enclosing_hexagon(all_verts, res.x)
                    best_beta = res.x
            except Exception:
                pass
            
            return best_S, best_center, best_beta
        
        # Initial evaluation
        S, center, beta = evaluate(positions, angles)
        if S < best_S:
            best_S = S
            best_positions = list(positions)
            best_angles = list(angles)
        
        # Iterative refinement: adjust positions using LP-based relaxation
        # For fixed angles, the non-overlap constraints between hexagons are convex
        # in position variables (for each pair, the minimum distance constraint is convex)
        # But solving the full LP with all pairs is expensive. Instead, use a 
        # simpler approach: gradient descent with collision avoidance.
        
        iterations = 0
        max_iter = 30
        step_size = 0.05
        
        while iterations < max_iter:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Compute gradient of S with respect to positions (approximate)
            grad_S = np.zeros((n, 2))
            verts = all_vertices_of_hexes(positions, angles)
            
            # For the enclosing hexagon, the gradient of D w.r.t. each vertex
            # points in the direction of the active normal
            normals = hex_apothem_norms(math.degrees(beta))
            
            # Find the most constraining normal direction
            max_proj = 0
            active_normal_idx = 0
            for k in range(6):
                nx, ny = normals[k]
                projs = verts @ np.array([nx, ny])
                max_p = np.max(projs)
                min_p = np.min(projs)
                if max_p > max_proj:
                    max_proj = max_p
                    active_normal_idx = k
                if -min_p > max_proj:
                    max_proj = -min_p
                    active_normal_idx = k + 6  # negative side
            
            # For each hexagon, compute the direction to move
            for i in range(n):
                vi = hex_vertices(positions[i][0], positions[i][1], angles[i])
                # Project onto active normal
                if active_normal_idx < 6:
                    nx, ny = normals[active_normal_idx]
                else:
                    nx, ny = -normals[active_normal_idx - 6]
                proj = vi @ np.array([nx, ny])
                max_proj_i = np.max(proj)
                
                # Move in direction opposite to active normal to reduce D
                grad_S[i] = np.array([nx, ny]) * (max_proj_i - max_proj)
            
            # Apply gradient descent with collision avoidance
            moved = False
            for i in range(n):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                new_pos = [positions[i][0] - step_size * grad_S[i][0],
                          positions[i][1] - step_size * grad_S[i][1]]
                
                # Check if moving maintains disjointness
                test_positions = list(positions)
                test_positions[i] = new_pos
                if are_disjoint(test_positions, angles):
                    S_test, _, _ = evaluate(test_positions, angles)
                    if S_test < best_S - 1e-12:
                        positions = test_positions
                        moved = True
                        if S_test < best_S:
                            best_S = S_test
                            best_positions = list(positions)
                            best_angles = list(angles)
            
            # Also try perturbing angles
            for i in range(n):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                for delta_a in [0.5, -0.5, 1.0, -1.0, 2.0, -2.0]:
                    test_angles = list(angles)
                    test_angles[i] = angles[i] + delta_a
                    S_test, _, _ = evaluate(positions, test_angles)
                    if S_test < best_S - 1e-12:
                        angles = test_angles
                        if S_test < best_S:
                            best_S = S_test
                            best_positions = list(positions)
                            best_angles = list(angles)
            
            if not moved:
                break
            iterations += 1
        
        return best_S, best_positions, best_angles, center, beta
    
    # --- Generate diverse initial configurations ---
    
    def generate_axial_config(axial_list):
        """Convert axial coords to Cartesian positions with 0-degree angles."""
        positions = []
        angles = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            positions.append((x, y))
            angles.append(0.0)
        return positions, angles
    
    # Candidate axial configurations
    candidate_axials = [
        # Ring + center (7 hexagons) + 4 more
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(0,2),(-2,2),(1,-2)],
        # 3x3 grid minus corners
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,-1),(0,-1)],
        # Linear arrangement
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(2,2)],
        # Diamond
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(2,0),(-1,-1),(0,2),(-2,0),(1,-1)],
        # Two rows of 5
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(2,2)],
        # T-shape
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(0,2),(0,-2),(1,-1),(-1,-1)],
        # Compact cluster
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(2,2),(3,1)],
        # Staggered
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)],
        # L-shape
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(0,2),(0,3),(0,4),(1,1),(2,1)],
        # H-shape
        [(0,0),(1,0),(2,0),(3,0),(4,0),(2,1),(2,2),(0,2),(4,2),(0,3),(4,3)],
    ]
    
    # Also add some with non-zero angles
    candidate_configs = []
    for ax in candidate_axials:
        positions, angles = generate_axial_config(ax)
        candidate_configs.append((positions, angles))
    
    # Add configurations with some rotated hexagons
    for ax in candidate_axials[:5]:
        positions, angles = generate_axial_config(ax)
        # Rotate every other hexagon by 30 degrees
        for i in range(1, len(positions), 2):
            angles[i] = 30.0
        candidate_configs.append((positions, angles))
        
        # Rotate hexagons at even positions by 60 degrees
        positions2, angles2 = generate_axial_config(ax)
        for i in range(0, len(positions2), 2):
            angles2[i] = 60.0
        candidate_configs.append((positions2, angles2))
    
    # --- Evaluate all candidates ---
    
    best_global_S = float('inf')
    best_global_positions = None
    best_global_angles = None
    best_global_center = None
    best_global_beta = 0.0
    
    for positions, angles in candidate_configs:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        S, pos_out, ang_out, center, beta = optimize_sequential(positions, angles, 10.0)
        if S < best_global_S:
            best_global_S = S
            best_global_positions = pos_out
            best_global_angles = ang_out
            best_global_center = center
            best_global_beta = beta
    
    # Fallback: simple grid
    if best_global_positions is None or best_global_S == float('inf'):
        axial = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(0,2),(-2,2),(1,-2)]
        positions, angles = generate_axial_config(axial)
        all_verts = all_vertices_of_hexes(positions, angles)
        best_global_S = float('inf')
        for i in range(12):
            beta = i * math.pi / 6.0 / 11.0
            S, center, D = min_enclosing_hexagon(all_verts, beta)
            if S < best_global_S:
                best_global_S = S
                best_global_center = center
                best_global_beta = beta
        best_global_positions = positions
        best_global_angles = angles
    
    # --- Final polish: fine-tune outer hexagon rotation ---
    
    if best_global_positions is not None and best_global_S < float('inf'):
        all_verts = all_vertices_of_hexes(best_global_positions, best_global_angles)
        try:
            res = minimize_scalar(
                lambda b: min_enclosing_hexagon(all_verts, b)[0],
                bounds=(0, math.pi / 6), method='bounded',
                options={'maxiter': 200, 'xatol': 1e-12}
            )
            if res.fun < best_global_S:
                best_global_S = res.fun
                _, best_global_center, _ = min_enclosing_hexagon(all_verts, res.x)
                best_global_beta = res.x
        except Exception:
            pass
    
    # --- Construct final answer ---
    
    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Apply tiny scale factor for safety
    SCALE = 1.0 + 1e-10
    pts_scaled = []
    for (x, y), a in zip(best_global_positions, best_global_angles):
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy, a))
    
    # Recompute S with scaled positions
    all_verts = all_vertices_of_hexes(pts_scaled, best_global_angles)
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (all_verts[:, 0] - cx) * nx + (all_verts[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    D_scaled = max_proj
    S_final = (D_scaled + APOTHEM) / APOTHEM
    
    inner = []
    for x, y, a in pts_scaled:
        inner.append([float(x), float(y), float(a)])
    
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
