import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Precompute hexagon vertices for unit hexagon
    # Vertices relative to center, rotated by 30 deg (flat top)
    # Side length 1. Apothem sqrt(3)/2.
    # Angle 0: (1, 0) is vertex? No, usually flat top means vertices at 0, 60, 120...
    # Let's use standard: vertices at angles 0, 60, 120, 180, 240, 300 degrees?
    # If flat top, top edge is horizontal. Vertices at 30, 90, 150, 210, 270, 330?
    # Let's stick to one convention. 
    # Unit hexagon side 1.
    # Vertices: (cos(theta), sin(theta)) for theta in [0, 60, ..., 300] gives pointy top.
    # Vertices: (cos(theta+30), sin(theta+30)) gives flat top.
    # The problem allows rotation, so the base shape doesn't matter as long as consistent.
    # Let's use pointy top for vertices: angles 0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3.
    base_vertices = np.array([
        [math.cos(math.pi/3 * k), math.sin(math.pi/3 * k)] for k in range(6)
    ])

    def get_hexagon_vertices(center, angle_rad):
        """
        Returns 6x2 array of vertices for a unit hexagon.
        center: (x, y)
        angle_rad: rotation in radians
        """
        cx, cy = center
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        return base_vertices @ R.T + np.array([cx, cy])

    def check_overlap(v1, v2, tol=1e-8):
        """
        Check if two convex polygons (hexagons) overlap.
        Uses Separating Axis Theorem (SAT).
        v1, v2: 6x2 arrays of vertices.
        Returns True if they overlap (intersect interiors), False if disjoint.
        """
        # For SAT, we need to check if there is a separating axis.
        # If no separating axis is found, they overlap.
        # The potential separating axes are the normals to the edges of both polygons.
        
        # Get normals for v1
        axes = []
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            edge = p2 - p1
            # Normal is perpendicular to edge
            normal = np.array([-edge[1], edge[0]])
            # Normalize (length of edge is 1, so length of normal is 1)
            axes.append(normal)
        
        # Get normals for v2
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i+1)%6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            axes.append(normal)

        # Project both polygons onto each axis
        for axis in axes:
            proj1 = v1 @ axis
            proj2 = v2 @ axis
            
            min1 = np.min(proj1)
            max1 = np.max(proj1)
            min2 = np.min(proj2)
            max2 = np.max(proj2)
            
            # Check for gap
            if max1 < min2 + tol or max2 < min1 + tol:
                return False # Disjoint
        
        return True # Overlap

    def calculate_enclosing_hexagon_size(positions, angles):
        """
        Calculate the minimum side length of a regular hexagon enclosing the given hexagons.
        positions: list of (x, y)
        angles: list of angles in radians
        
        Returns: (side_length, center, rotation_angle)
        """
        # We need to find the minimum S such that there exists a center C and rotation R
        # such that all hexagons are inside the regular hexagon of side S, center C, rotation R.
        
        # This is a non-convex optimization problem in (Cx, Cy, R, S).
        # However, for a fixed R, it is a Linear Program in (Cx, Cy, S).
        
        # Strategy:
        # 1. Iterate over possible rotations R (0 to 60 deg).
        # 2. For each R, solve LP to find min S and corresponding C.
        # 3. Return global min.
        
        # To speed up, we can sample R.
        # The optimal R is likely aligned with one of the inner hexagons or the container.
        
        best_S = float('inf')
        best_C = (0,0)
        best_R = 0.0
        
        # Sample rotations. 
        # Since the container is symmetric under 60 deg, we only need to check 0..60.
        # However, the inner hexagons have specific orientations.
        # The optimal container rotation will likely be such that one of its sides is parallel to a side of an inner hexagon.
        # So we should check rotations aligned with the inner hexagons' orientations.
        
        candidate_rotations = set()
        for ang in angles:
            # Align container sides with inner hexagon sides
            # Inner hexagon sides are at ang + 0, ang + 30, ang + 60... ?
            # Base vertices at 0, 60... Edges are at 30, 90...
            # So normals to edges are at 0, 60...
            # Container normals are at R + 30, R + 90... (if flat top) or R + 0... (if pointy top)
            # Let's assume container normals are at R + 30 + k*60.
            # To align, R + 30 + k*60 should be equal to ang + m*60.
            # So R = ang + (m-k)*60 - 30.
            # Basically R = ang - 30 (mod 60).
            candidate_rotations.add((ang - math.pi/6) % (math.pi/3))
            candidate_rotations.add(ang % (math.pi/3))
            
        # Also check 0, pi/6, pi/3
        candidate_rotations.add(0.0)
        candidate_rotations.add(math.pi/6)
        candidate_rotations.add(math.pi/3)
        
        # Add some random samples to catch other local minima
        for _ in range(5):
            candidate_rotations.add(np.random.uniform(0, math.pi/3))

        for R in candidate_rotations:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
                
            # Normals of the container hexagon
            # Container is regular hexagon.
            # If we define container by inequalities:
            # |p . n_k| <= S/2 + Apothem_inner? No.
            # The container is the region { x | |x . n_k| <= S/2 + A_inner? }
            # Wait, the "side length" S of the container.
            # The distance from center to side is S * sqrt(3) / 2.
            # Let D = S * sqrt(3) / 2.
            # Then the container is { x | |x . n_k| <= D } for k=0..5.
            # The inner hexagons must be inside this region.
            # An inner hexagon is contained in the container if all its vertices are inside.
            # So for each vertex v of each inner hexagon:
            # | (v - C) . n_k | <= D  for all k.
            
            # This is a system of linear inequalities in C and D.
            # Minimize D.
            
            # Variables: Cx, Cy, D
            # Minimize D
            # Constraints:
            # For each inner hexagon i, each vertex v_j:
            # -D <= (v_j - C) . n_k <= D
            # -D <= (Cx - v_j.x)*n_k.x + (Cy - v_j.y)*n_k.y <= D
            
            # Let's construct LP.
            # c = [0, 0, 1]
            # A_ub, b_ub
            
            A_ub = []
            b_ub = []
            
            normals = []
            for k in range(6):
                theta = R + math.pi/6 + k * math.pi/3 # Assuming flat-top container?
                # Let's be consistent. 
                # If container side length is S, apothem is S*sqrt(3)/2.
                # Let's use normals pointing outwards.
                nx = math.cos(theta)
                ny = math.sin(theta)
                normals.append((nx, ny))
            
            for i in range(len(positions)):
                cx_i, cy_i = positions[i]
                ang_i = angles[i]
                verts = get_hexagon_vertices((cx_i, cy_i), ang_i)
                
                for v in verts:
                    vx, vy = v
                    for nx, ny in normals:
                        # (vx - Cx)*nx + (vy - Cy)*ny <= D
                        # -nx*Cx - ny*Cy - D <= -vx*nx - vy*ny
                        A_ub.append([-nx, -ny, -1.0])
                        b_ub.append(-vx*nx - vy*ny)
                        
                        # -(vx - Cx)*nx - (vy - Cy)*ny <= D
                        # nx*Cx + ny*Cy - D <= vx*nx + vy*ny
                        A_ub.append([nx, ny, -1.0])
                        b_ub.append(vx*nx + vy*ny)
                        
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            bounds = [(None, None), (None, None), (0, None)]
            
            try:
                res = linprog(np.array([0.0, 0.0, 1.0]), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.status == 0:
                    D = res.x[2]
                    S = D / (math.sqrt(3) / 2.0)
                    if S < best_S:
                        best_S = S
                        best_C = (res.x[0], res.x[1])
                        best_R = R
            except Exception:
                pass

        return best_S, best_C, best_R

    def objective_function(params):
        """
        Objective for scipy.optimize.minimize
        params: flattened array of [x0, y0, a0, x1, y1, a1, ...]
        """
        n = len(params) // 3
        positions = []
        angles = []
        for i in range(n):
            positions.append((params[3*i], params[3*i+1]))
            angles.append(params[3*i+2])
        
        # Check overlaps
        # If overlap, return large penalty
        # We can use a simple pairwise check
        # To speed up, we can skip detailed SAT if bounding boxes don't overlap
        for i in range(n):
            for j in range(i+1, n):
                p1 = positions[i]
                p2 = positions[j]
                # Bounding box check
                # Hexagon diameter is 2.
                dx = abs(p1[0] - p2[0])
                dy = abs(p1[1] - p2[1])
                if dx > 2.0 or dy > 2.0:
                    continue
                
                v1 = get_hexagon_vertices(p1, angles[i])
                v2 = get_hexagon_vertices(p2, angles[j])
                
                if check_overlap(v1, v2):
                    # Penalty proportional to overlap depth?
                    # For now, just a large constant.
                    # To make it smoother, we could estimate penetration depth, but that's complex.
                    return 1e6 + calculate_enclosing_hexagon_size(positions, angles)[0]
        
        S, C, R = calculate_enclosing_hexagon_size(positions, angles)
        return S

    def run_local_search(initial_positions, initial_angles, max_iter=1000):
        """
        Run Nelder-Mead to optimize positions and angles.
        """
        n = len(initial_positions)
        params0 = []
        for i in range(n):
            params0.extend([initial_positions[i][0], initial_positions[i][1], initial_angles[i]])
        
        try:
            res = minimize(objective_function, params0, method='Nelder-Mead', 
                           options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
            return res.x, res.fun
        except Exception:
            return None, float('inf')

    best_global_S = float('inf')
    best_global_params = None
    
    # Multi-start strategy
    # 1. Axial lattice starts (from original code, converted to cartesian)
    # 2. Random clustered starts
    
    # Generate a few good axial starts
    axial_starts = [
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (0, 2), (0, -2)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (1, 3), (2, 3)]
    ]
    
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return pts

    starts = []
    
    # Add axial starts with 0 angle
    for ax in axial_starts:
        pts = axial_to_cartesian(ax)
        angs = [0.0] * len(pts)
        starts.append((pts, angs))
    
    # Add random starts
    # Cluster around center
    np.random.seed(42)
    for _ in range(5):
        pts = []
        angs = []
        # Center at 0,0
        pts.append((0.0, 0.0))
        angs.append(0.0)
        # 6 around it
        for k in range(6):
            ang = k * math.pi / 3
            r = math.sqrt(3) # distance to touch center hexagon?
            # Distance between centers of touching hexagons is sqrt(3).
            x = r * math.cos(ang)
            y = r * math.sin(ang)
            pts.append((x, y))
            angs.append(0.0)
        # 4 more, pushed out
        for k in range(4):
            ang = (k + 0.5) * math.pi / 3
            r = 2 * math.sqrt(3) # further out
            x = r * math.cos(ang)
            y = r * math.sin(ang)
            pts.append((x, y))
            angs.append(0.0)
            
        # Add small random perturbations
        pts_perturbed = []
        angs_perturbed = []
        for i in range(11):
            dx = np.random.uniform(-0.1, 0.1)
            dy = np.random.uniform(-0.1, 0.1)
            da = np.random.uniform(-0.1, 0.1)
            pts_perturbed.append((pts[i][0] + dx, pts[i][1] + dy))
            angs_perturbed.append(angs[i] + da)
            
        starts.append((pts_perturbed, angs_perturbed))

    # Sort starts by initial size to prioritize better ones
    def get_initial_size(pts, angs):
        S, _, _ = calculate_enclosing_hexagon_size(pts, angs)
        return S

    starts_with_size = [(get_initial_size(p, a), p, a) for p, a in starts]
    starts_with_size.sort(key=lambda x: x[0])
    
    # Run optimization on top starts
    for size, pts, angs in starts_with_size:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        if size >= best_global_S:
            continue # Skip if already worse than best found
            
        res_params, res_val = run_local_search(pts, angs, max_iter=2000)
        
        if res_params is not None and res_val < best_global_S:
            best_global_S = res_val
            best_global_params = res_params

    # If no optimization succeeded or result is bad, fallback to best axial
    if best_global_params is None or best_global_S > 100:
        # Use the first axial start
        pts, angs = starts[0]
        S, C, R = calculate_enclosing_hexagon_size(pts, angs)
        best_global_S = S
        # Construct params
        params = []
        for i in range(11):
            params.extend([pts[i][0], pts[i][1], angs[i]])
        best_global_params = params

    # Extract final solution
    n = 11
    positions = []
    angles = []
    for i in range(n):
        positions.append((best_global_params[3*i], best_global_params[3*i+1]))
        angles.append(best_global_params[3*i+2])
        
    S, C, R = calculate_enclosing_hexagon_size(positions, angles)
    
    # Final check: ensure no overlaps. If overlaps exist, we might need to scale up or shift.
    # But the optimizer penalized overlaps, so it should be fine.
    # However, numerical errors in SAT might cause false positives or negatives.
    # Let's do a final verification.
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            v1 = get_hexagon_vertices(positions[i], angles[i])
            v2 = get_hexagon_vertices(positions[j], angles[j])
            if check_overlap(v1, v2, tol=1e-6):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # If invalid, scale the configuration slightly to remove overlaps
        # Find center of mass
        cx = sum(p[0] for p in positions) / n
        cy = sum(p[1] for p in positions) / n
        
        scale = 1.0
        while not valid and scale < 2.0:
            scale += 0.01
            positions_scaled = []
            for p in positions:
                positions_scaled.append((cx + (p[0]-cx)*scale, cy + (p[1]-cy)*scale))
            # Re-check
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    v1 = get_hexagon_vertices(positions_scaled[i], angles[i])
                    v2 = get_hexagon_vertices(positions_scaled[j], angles[j])
                    if check_overlap(v1, v2, tol=1e-6):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                positions = positions_scaled
                # Recalculate S
                S, C, R = calculate_enclosing_hexagon_size(positions, angles)
                break
                
    inner = []
    for i in range(n):
        inner.append([float(positions[i][0]), float(positions[i][1]), float(math.degrees(angles[i]))])
        
    center = [float(C[0]), float(C[1])]
    side = float(S)
    rotation = float(math.degrees(R))
    
    return inner, center, side, rotation
