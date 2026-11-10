# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize, differential_evolution

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = 600.0  # seconds
    
    # --- Geometry Helpers ---
    
    def get_hex_vertices(cx, cy, s, a_deg):
        """
        Returns vertices of a regular hexagon.
        cx, cy: center
        s: side length
        a_deg: angle in degrees (rotation of the hexagon)
        Vertices are at angles a + 60*i for i=0..5.
        """
        a_rad = np.deg2rad(a_deg)
        angles = a_rad + np.deg2rad(60 * np.arange(6))
        xs = cx + s * np.cos(angles)
        ys = cy + s * np.sin(angles)
        return np.column_stack((xs, ys))
    
    def edge_normals(hex_verts):
        """
        Returns the 6 outward normal vectors of the hexagon edges.
        hex_verts: (6, 2) array of vertices in CCW order.
        """
        normals = []
        for i in range(6):
            p1 = hex_verts[i]
            p2 = hex_verts[(i + 1) % 6]
            edge_vec = p2 - p1
            # Normal is perpendicular to edge, pointing outward
            # For CCW polygon, outward normal is (dy, -dx) normalized?
            # Let's check: edge (1,0) -> normal (0,-1). Edge (0,1) -> normal (1,0).
            # Wait, for a square (0,0)->(1,0)->(1,1)->(0,1), CCW.
            # Edge (0,0)->(1,0): vec (1,0). Outward normal should be (0,-1).
            # (dy, -dx) = (0, -1). Correct.
            n = np.array([edge_vec[1], -edge_vec[0]])
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-10:
                n = n / n_norm
            normals.append(n)
        return np.array(normals)
    
    def hex_separation(h1, h2):
        """
        Checks if two hexagons h1, h2 are separated.
        h1, h2: (6,2) arrays of vertices.
        Returns the minimum gap between the two hexagons.
        If gap > 1e-6, they are separated.
        If gap <= 1e-6, they are touching or intersecting.
        
        We check the 12 edge normals (6 from h1, 6 from h2).
        For each normal, we project both hexagons onto the normal line.
        The gap is the distance between the max projection of one and min projection of the other.
        """
        normals1 = edge_normals(h1)
        normals2 = edge_normals(h2)
        
        min_gap = np.inf
        
        # Check normals from h1
        for n in normals1:
            proj1 = h1 @ n
            proj2 = h2 @ n
            # If n points from h1 to h2 roughly, we want max(proj2) - min(proj1)
            # But we don't know direction.
            # Separation if max(proj1) < min(proj2) OR max(proj2) < min(proj1)
            gap1 = min(proj2) - max(proj1)
            gap2 = min(proj1) - max(proj2)
            # The gap along this normal is the positive one if they are separated
            # If both are negative, they overlap in this projection.
            # The actual distance is the max of the positive gaps?
            # Actually, if they are separated, one of gap1, gap2 will be positive.
            # If they overlap, both are negative (or zero if touching).
            # We want the minimum distance, which corresponds to the smallest positive gap
            # among all normals? No, the minimum distance is the minimum over all normals
            # of the separation distance IF they are separated by that normal.
            # If they are not separated by any normal, they intersect.
            
            # Let's define the separation distance along normal n as:
            # d = max(0, min(proj2) - max(proj1)) + max(0, min(proj1) - max(proj2))
            # This is 0 if they overlap in this projection.
            # If they are separated, one term is positive.
            # The actual Euclidean distance is NOT simply the max of these.
            # However, the condition for "non-intersection" in the task is:
            # "on at least one of their 12 edge normals the projected intervals are apart by more than 1e-6"
            # This implies that if they are separated, there must be at least one normal
            # where the gap is > 1e-6.
            # And if they are intersecting, for ALL normals, the gap is <= 1e-6 (actually <= 0).
            
            # So to check validity, we just need to see if ANY normal has a gap > 1e-6.
            # To find the "tightness" for optimization, we want to maximize the minimum gap?
            # No, we want to ensure ALL pairs are separated.
            # For a pair, if they are separated, the "separation margin" is the minimum
            # over all normals? No.
            # If they are separated by normal n1 with gap 1.0 and normal n2 with gap 0.5,
            # they are separated. The "closest" approach is determined by the normal
            # that gives the smallest positive gap?
            # Actually, if two convex polygons are separated, the minimum distance
            # is the maximum over all separating normals of the gap?
            # No, the minimum distance is the minimum over all pairs of points.
            # For convex polygons, the minimum distance is attained either at a vertex-vertex,
            # vertex-edge, or edge-edge pair.
            # The edge-edge case corresponds to parallel edges.
            # The vertex-edge case corresponds to a normal of one edge passing through a vertex of the other.
            
            # The task's validity check is specific:
            # "on at least one of their 12 edge normals the projected intervals are apart by more than 1e-6"
            # This is a sufficient condition for separation.
            # If this condition holds, they are valid.
            # If it fails for all 12 normals, they are invalid (intersecting or touching).
            
            # So for optimization, we can define a penalty for a pair as:
            # penalty = max(0, 1e-6 - max_gap)
            # where max_gap = max over all 12 normals of (positive gap)
            # If max_gap > 1e-6, penalty is 0.
            # If max_gap <= 1e-6, penalty is positive.
            
            pos_gap1 = max(0, gap1)
            pos_gap2 = max(0, gap2)
            max_gap = max(pos_gap1, pos_gap2)
            min_gap = min(min_gap, max_gap) # We want the maximum possible gap for this pair?
            # No, we want to know if THERE EXISTS a normal with gap > 1e-6.
            # So we care about the MAX gap over all normals.
            # If max_gap > 1e-6, it's valid.
            # So we should track the max gap.
            
        # Let's redo the logic.
        max_gap_pair = 0.0
        for n in list(normals1) + list(normals2):
            proj1 = h1 @ n
            proj2 = h2 @ n
            gap1 = min(proj2) - max(proj1)
            gap2 = min(proj1) - max(proj2)
            max_gap = max(0, gap1, gap2)
            max_gap_pair = max(max_gap_pair, max_gap)
        
        return max_gap_pair

    def outer_containment_penalty(inner_hexes, outer_center, outer_side, outer_angle):
        """
        Calculates the maximum violation of containment.
        Returns 0 if all inner hexagons are inside the outer hexagon.
        """
        outer_verts = get_hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
        max_violation = 0.0
        
        for h in inner_hexes:
            for p in h:
                # Check if p is inside outer_verts
                # Using cross product method for convex polygon
                inside = True
                for i in range(6):
                    p1 = outer_verts[i]
                    p2 = outer_verts[(i + 1) % 6]
                    cross = np.cross(p2 - p1, p - p1)
                    if cross < -1e-6:
                        inside = False
                        # Estimate violation depth
                        # Distance from p to edge p1-p2
                        edge_vec = p2 - p1
                        edge_len = np.linalg.norm(edge_vec)
                        if edge_len > 1e-10:
                            dist = abs(np.cross(edge_vec, p - p1)) / edge_len
                        else:
                            dist = 0
                        max_violation = max(max_violation, dist)
                        break
                if not inside:
                    break
                    
        return max_violation

    def objective(params):
        """
        Objective function to minimize.
        params: [x1, y1, a1, ..., x11, y11, a11, cx, cy, ca, s]
        11*3 + 4 = 37 parameters.
        """
        if time.time() > deadline - 10:
            return 1e10
            
        # Extract inner hexagons
        inner_params = params[:33].reshape(11, 3)
        cx, cy, ca, s = params[33:37]
        
        # Ensure side length is reasonable
        if s < np.sqrt(11) - 0.1:
            return 1e10
            
        inner_hexes = []
        for i in range(11):
            x, y, a = inner_params[i]
            verts = get_hex_vertices(x, y, 1.0, a)
            inner_hexes.append(verts)
            
        # Containment penalty
        cont_pen = outer_containment_penalty(inner_hexes, np.array([cx, cy]), s, ca)
        
        # Non-overlap penalty
        overlap_pen = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                gap = hex_separation(inner_hexes[i], inner_hexes[j])
                if gap <= 1e-6:
                    # Penalty proportional to how much they overlap
                    # We want gap > 1e-6.
                    # penalty = max(0, 1e-6 - gap)
                    overlap_pen += (1e-6 - gap)
                    
        # We also want to minimize the side length s.
        # So we add a term for s.
        # Total = cont_pen + overlap_pen + s
        
        # Scale penalties to be significant
        total = cont_pen * 100.0 + overlap_pen * 100.0 + s
        
        return total

    # --- Initial Guesses ---
    
    # Symmetric arrangement: 1 center, 6 around, 4 more?
    # Or a hexagonal lattice.
    # Let's try a hexagonal lattice of 11 hexagons.
    # Center at (0,0).
    # 6 neighbors at distance 2 (side length 1, so distance between centers is 2 for touching).
    # 4 more at distance 4?
    
    # Let's use a random initial guess based on a hexagonal lattice.
    def generate_initial_guess():
        # Center
        centers = [(0, 0)]
        # 6 neighbors
        for i in range(6):
            angle = np.deg2rad(60 * i)
            centers.append((2 * np.cos(angle), 2 * np.sin(angle)))
        # 4 more, placed at distance 4, angles 0, 60, 120, 180?
        # This might overlap.
        # Let's just scatter them randomly within a large circle.
        
        # Better: use a known packing configuration if possible.
        # For 11 hexagons, a common packing is a hexagon of side 2 (7 hexes) plus 4 more.
        # Or a line?
        
        # Let's try a hexagonal grid.
        # (0,0)
        # (2,0), (-2,0), (1, sqrt(3)), (-1, sqrt(3)), (1, -sqrt(3)), (-1, -sqrt(3))
        # (4,0), (-4,0), (3, sqrt(3)), (-3, sqrt(3)) -> 11 hexes
        
        sqrt3 = np.sqrt(3)
        centers = [
            (0, 0),
            (2, 0), (-2, 0),
            (1, sqrt3), (-1, sqrt3),
            (1, -sqrt3), (-1, -sqrt3),
            (4, 0), (-4, 0),
            (3, sqrt3), (-3, sqrt3)
        ]
        
        params = []
        for c in centers:
            params.append([c[0], c[1], 0])
            
        # Outer hexagon
        # Must contain all.
        # Max x is 4+1=5. Min x is -5.
        # Max y is sqrt(3)+1 ~ 2.73. Min y ~ -2.73.
        # A hexagon with side s centered at (0,0) has width 2s and height sqrt(3)s.
        # If flat top (angle 0), width is 2s, height sqrt(3)s.
        # If pointy top (angle 30), width sqrt(3)s, height 2s.
        
        # Let's try flat top (angle 0).
        # Need 2s >= 10 -> s >= 5.
        # Need sqrt(3)s >= 2*2.73 = 5.46 -> s >= 5.46/1.732 = 3.15.
        # So s=5 is enough.
        
        params.extend([0, 0, 0, 5.0])
        
        return np.array(params)
    
    x0 = generate_initial_guess()
    
    # Bounds
    # Inner hexagons: x, y in [-10, 10], angle in [0, 60]
    # Outer: cx, cy in [-5, 5], ca in [0, 60], s in [3.3, 10]
    
    bounds = []
    for i in range(11):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((0, 60))
    bounds.append((-5, 5))
    bounds.append((-5, 5))
    bounds.append((0, 60))
    bounds.append((np.sqrt(11), 10))
    
    # Optimization
    best_params = x0
    best_score = objective(x0)
    
    # Use differential evolution for global search
    # Then local refinement with L-BFGS-B
    
    if time.time() < deadline - 60:
        try:
            result_de = differential_evolution(objective, bounds, maxiter=50, popsize=10, tol=1e-4, seed=42, polish=False)
            if result_de.fun < best_score:
                best_score = result_de.fun
                best_params = result_de.x
        except Exception as e:
            pass
    
    if time.time() < deadline - 30:
        try:
            result_lbfgs = minimize(objective, best_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-8})
            if result_lbfgs.fun < best_score:
                best_score = result_lbfgs.fun
                best_params = result_lbfgs.x
        except Exception as e:
            pass
            
    # Extract result
    inner_params = best_params[:33].reshape(11, 3)
    cx, cy, ca, s = best_params[33:37]
    
    inner_hex_data = inner_params
    outer_hex_data = np.array([cx, cy, ca])
    outer_hex_side_length = float(s)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
