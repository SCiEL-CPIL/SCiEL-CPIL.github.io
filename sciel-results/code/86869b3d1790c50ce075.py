import math
import numpy as np
import time
from scipy.optimize import linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Hexagon vertex generator for unit hexagon
    def hex_vertices(angle_deg=0.0):
        """Return 6 vertices of a unit regular hexagon centered at origin, rotated by angle_deg."""
        angle = angle_deg * math.pi / 180.0
        # Vertices at angles 30, 90, 150, 210, 270, 330 degrees (pointy-top)
        angles = [math.pi / 6 + k * math.pi / 3 for k in range(6)]
        verts = np.array([[math.cos(a + angle), math.sin(a + angle)] for a in angles])
        return verts

    # Compute support function: max dot product of vertices with direction
    def support(verts, direction):
        return np.max(verts @ direction)

    # For two unit hexagons with given rotations, compute the minimum distance
    # between centers in a given direction for them to be non-overlapping
    def min_distance(r1, r2, direction):
        """Minimum center distance in direction 'direction' for non-overlap."""
        v1 = hex_vertices(r1)
        v2 = hex_vertices(r2)
        # The Minkowski sum approach: distance >= support(v1, dir) + support(v2, -dir)
        # But we need the actual minimum, so:
        # Two hexagons don't overlap iff for all directions, the gap is positive
        # The critical direction is where the gap is smallest
        # For two convex polygons, they overlap iff there exists a separating axis
        # Non-overlap: for all 6+6=12 edge normals, the projections don't overlap
        # But the minimum distance in a specific direction is:
        return support(v1, direction) + support(v2, -direction)

    # Check if two hexagons at given positions and rotations overlap
    def hexagons_overlap(pos1, rot1, pos2, rot2):
        v1 = hex_vertices(rot1) + pos1
        v2 = hex_vertices(rot2) + pos2
        # Check all edge normals of both hexagons
        all_normals = []
        for v in [v1, v2]:
            for i in range(6):
                p1 = v[i]
                p2 = v[(i + 1) % 6]
                edge = p2 - p1
                # Normal perpendicular to edge, pointing outward
                normal = np.array([edge[1], -edge[0]])
                # Ensure it points outward
                centroid = np.mean(v, axis=0)
                if np.dot(normal, centroid - p1) < 0:
                    normal = -normal
                normal = normal / np.linalg.norm(normal)
                all_normals.append(normal)
        
        for n in all_normals:
            proj1 = v1 @ n
            proj2 = v2 @ n
            if np.max(proj1) > np.min(proj2) + EPS and np.max(proj2) > np.min(proj1) + EPS:
                # Check if they actually overlap in this direction
                # They overlap if the intervals overlap
                overlap = min(np.max(proj1), np.max(proj2)) - max(np.min(proj1), np.min(proj2))
                if overlap > EPS:
                    # But we need to check all directions; if ANY direction shows separation, no overlap
                    # Actually, if ALL directions show overlap, they overlap
                    # This is getting complex; let's use a different approach
                    pass
        
        # Simpler: check if center of one is inside the other, or if any edge intersects
        # For now, use the separating axis theorem properly
        for n in all_normals:
            min1, max1 = np.min(v1 @ n), np.max(v1 @ n)
            min2, max2 = np.min(v2 @ n), np.max(v2 @ n)
            if max1 < min2 - EPS or max2 < min1 - EPS:
                return False  # Separated, no overlap
        
        # All axes show overlap; check if actually overlapping
        # For convex polygons, if no separating axis, they overlap
        return True

    # Compute the minimum enclosing hexagon for a set of inner hexagons
    def compute_enclosing_hexagon(positions, rotations, outer_rot_deg):
        """Compute minimal enclosing regular hexagon with given rotation."""
        outer_angle = outer_rot_deg * math.pi / 180.0
        # Outer hexagon edge normals
        outer_normals = np.array([[math.cos(outer_angle + math.pi / 6 + k * math.pi / 3),
                                   math.sin(outer_angle + math.pi / 6 + k * math.pi / 3)]
                                  for k in range(6)])
        
        # For each inner hexagon, compute its support in each outer normal direction
        # and in the opposite direction
        A_ub = []
        b_ub = []
        
        for i in range(N):
            rot_i = rotations[i] * math.pi / 180.0
            v_i = hex_vertices(rotations[i]) + positions[i]
            
            for k in range(6):
                n_k = outer_normals[k]
                # Constraint: max(v_i . n_k) <= cx * n_k[0] + cy * n_k[1] + S * APOTHEM
                h_pos = np.max(v_i @ n_k)
                A_ub.append([-n_k[0], -n_k[1], -APOTHEM])
                b_ub.append(-h_pos)
                
                # Constraint: max(v_i . (-n_k)) <= -cx * n_k[0] - cy * n_k[1] + S * APOTHEM
                h_neg = np.max(v_i @ (-n_k))
                A_ub.append([n_k[0], n_k[1], -APOTHEM])
                b_ub.append(h_neg)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                cx, cy, S = res.x
                return S, (cx, cy)
            else:
                return float('inf'), None
        except Exception:
            return float('inf'), None

    # Contact graph enumeration approach
    # For 11 unit hexagons, we enumerate possible adjacency structures
    # Each hexagon can be adjacent to at most 6 others
    # We use a canonical form to avoid duplicates
    
    # Start with a known good configuration: 1 center + 6 ring + 4 more
    # Axial coordinates for a compact arrangement
    
    # Generate candidate configurations using different adjacency patterns
    def generate_candidates():
        candidates = []
        
        # Configuration 1: Compact 3-4-4 arrangement
        # Center hexagon at (0,0), surrounded by 6, then 4 more
        axial_1 = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2)
        ]
        candidates.append((axial_1, 0.0))  # All same rotation
        
        # Configuration 2: 2-4-3-2 arrangement
        axial_2 = [
            (0, 0), (1, 0),
            (0, 1), (1, 1), (2, 1), (-1, 1),
            (0, 2), (1, 2), (2, 2),
            (0, 3), (1, 3)
        ]
        candidates.append((axial_2, 0.0))
        
        # Configuration 3: More spread out
        axial_3 = [
            (0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1), (2, 1),
            (0, 2), (1, 2), (2, 2),
            (1, 3), (2, 3)
        ]
        candidates.append((axial_3, 0.0))
        
        # Configuration 4: Diamond-like
        axial_4 = [
            (0, 0),
            (1, 0), (-1, 0),
            (0, 1), (1, 1), (-1, 1),
            (0, 2), (1, 2), (-1, 2),
            (0, 3), (1, 3)
        ]
        candidates.append((axial_4, 0.0))
        
        # Configuration 5: With some rotated hexagons for tighter packing
        axial_5 = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 1), (1, 2), (-1, 2), (0, 3)
        ]
        candidates.append((axial_5, 0.0))
        
        # Configuration 6: Linear arrangement with offsets
        axial_6 = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2),
            (1, 3)
        ]
        candidates.append((axial_6, 0.0))
        
        # Configuration 7: Cross pattern
        axial_7 = [
            (0, 0),
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, 1),
            (1, -1), (-1, -1),
            (2, 0), (0, 2)
        ]
        candidates.append((axial_7, 0.0))
        
        # Configuration 8: Tight 3x4 grid with one offset
        axial_8 = [
            (0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2),
            (1, 3), (2, 3)
        ]
        candidates.append((axial_8, 0.0))
        
        # Configuration 9: With 30-degree rotated outer ring
        axial_9 = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (-2, 0), (0, 2), (1, 2)
        ]
        candidates.append((axial_9, 30.0))  # Ring hexagons rotated 30 degrees
        
        # Configuration 10: Dense cluster
        axial_10 = [
            (0, 0), (1, 0), (2, 0),
            (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2),
            (1, 3), (2, 3)
        ]
        candidates.append((axial_10, 0.0))
        
        return candidates

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def check_valid_packing(positions, rotations, eps=1e-9):
        """Check if the packing is valid (no overlaps)."""
        for i in range(N):
            for j in range(i + 1, N):
                if hexagons_overlap(positions[i], rotations[i], positions[j], rotations[j]):
                    return False
        return True

    def optimize_configuration(axial_coords, base_rotation, time_budget):
        """Optimize a single configuration by adjusting positions and outer hexagon rotation."""
        start_time = time.time()
        positions = axial_to_cartesian(axial_coords)
        rotations = [base_rotation] * N
        
        # Try different outer hexagon rotations
        best_S = float('inf')
        best_center = None
        best_positions = positions.copy()
        best_rotations = rotations.copy()
        best_outer_rot = 0.0
        
        # Search over outer hexagon rotation angles
        n_angles = 20
        for i in range(n_angles):
            outer_rot = i * (30.0 / n_angles)  # 0 to 30 degrees
            
            if time.time() - start_time > time_budget:
                break
            
            S, center = compute_enclosing_hexagon(positions, rotations, outer_rot)
            if S < best_S:
                best_S = S
                best_center = center
                best_positions = positions.copy()
                best_rotations = rotations.copy()
                best_outer_rot = outer_rot
        
        # Local optimization: adjust positions slightly to reduce S
        if best_S < float('inf') and time.time() - start_time < time_budget:
            # Try small perturbations
            np.random.seed(42)
            for _ in range(50):
                if time.time() - start_time > time_budget:
                    break
                
                perturbed = best_positions.copy()
                for i in range(N):
                    dx = np.random.normal(0, 0.05)
                    dy = np.random.normal(0, 0.05)
                    perturbed[i] += [dx, dy]
                
                # Check validity
                if check_valid_packing(perturbed, best_rotations):
                    S, center = compute_enclosing_hexagon(perturbed, best_rotations, best_outer_rot)
                    if S < best_S:
                        best_S = S
                        best_center = center
                        best_positions = perturbed
        
        return best_S, best_center, best_positions, best_rotations, best_outer_rot

    def final_polish(positions, rotations, outer_rot, time_budget):
        """Final polish: refine positions to minimize S."""
        start_time = time.time()
        
        # Use gradient-free optimization on positions
        from scipy.optimize import minimize
        
        def objective(params):
            pts = params[:2 * N].reshape(N, 2)
            # Check overlap penalty
            penalty = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    if hexagons_overlap(pts[i], rotations[i], pts[j], rotations[j]):
                        penalty += 1000.0
            if penalty > 0:
                return 1e6 + penalty
            
            S, center = compute_enclosing_hexagon(pts, rotations, outer_rot)
            return S if S < float('inf') else 1e6
        
        initial = positions.ravel()
        try:
            res = minimize(objective, initial, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < 1e5:  # Valid solution
                pts = res.x[:2 * N].reshape(N, 2)
                S, center = compute_enclosing_hexagon(pts, rotations, outer_rot)
                if S < float('inf') and check_valid_packing(pts, rotations):
                    return pts, S, center
        except Exception:
            pass
        
        return positions, float('inf'), None

    # Main search
    candidates = generate_candidates()
    time_per_candidate = (TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)) / len(candidates)
    
    best_global_S = float('inf')
    best_global_positions = None
    best_global_rotations = None
    best_global_center = None
    best_global_outer_rot = 0.0

    for axial_coords, base_rot in candidates:
        elapsed = time.time() - START_TIME
        if elapsed > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        remaining = TIME_LIMIT - SAFETY_MARGIN - elapsed
        budget = min(time_per_candidate, remaining / 2)  # Leave time for polish
        
        S, center, positions, rotations, outer_rot = optimize_configuration(
            axial_coords, base_rot, budget
        )
        
        if S < best_global_S:
            best_global_S = S
            best_global_positions = positions
            best_global_rotations = rotations
            best_global_center = center
            best_global_outer_rot = outer_rot
    
    # Final polish on the best configuration
    if best_global_positions is not None and best_global_S < float('inf'):
        elapsed = time.time() - START_TIME
        if elapsed < TIME_LIMIT - SAFETY_MARGIN:
            polish_budget = TIME_LIMIT - SAFETY_MARGIN - elapsed
            positions, S, center = final_polish(
                best_global_positions, best_global_rotations, best_global_outer_rot, polish_budget
            )
            if S < best_global_S:
                best_global_S = S
                best_global_positions = positions
                best_global_center = center

    # Fallback if no valid configuration found
    if best_global_positions is None or best_global_S == float('inf'):
        # Use the first candidate as fallback
        axial_0 = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2)
        ]
        best_global_positions = axial_to_cartesian(axial_0)
        best_global_rotations = [0.0] * N
        best_global_outer_rot = 0.0
        S, center = compute_enclosing_hexagon(best_global_positions, best_global_rotations, best_global_outer_rot)
        best_global_S = S
        best_global_center = center if center else (0.0, 0.0)

    # Build the result
    cx, cy = best_global_center
    inner = []
    for i in range(N):
        x, y = best_global_positions[i]
        angle_deg = best_global_rotations[i]
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(best_global_outer_rot)

    return inner, center, side, rotation
