# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from itertools import combinations
from scipy.optimize import minimize

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    time_limit = 600.0
    n = 11

    # 1. Precompute geometry
    # Inner hexagons: Unit side, angle 0. Vertices at 0, 60, ...
    inner_angles_rad = np.deg2rad([0, 60, 120, 180, 240, 300])
    inner_verts_rel = np.array([[np.cos(a), np.sin(a)] for a in inner_angles_rad])
    
    # Outer hexagon: Side S, angle 0. Vertices at 0, 60, ...
    # Normals to edges are at 30, 90, ...
    outer_normals_angles_rad = np.deg2rad([30, 90, 150, 210, 270, 330])
    outer_normals = np.array([[np.cos(a), np.sin(a)] for a in outer_normals_angles_rad])
    
    # Minkowski sum of two unit hexagons (angle 0) is a regular hexagon with side 2.
    # Vertices of Minkowski sum (relative to center of one hex, offset by center of other):
    # Actually, for non-overlap of two convex polygons A and B, the vector (cB - cA) must NOT be in interior(A + B).
    # For two identical regular hexagons with side 1, the Minkowski sum is a regular hexagon with side 2.
    # The condition for non-overlap (allowing touch) is that the distance between centers is >= distance to boundary of Minkowski sum.
    # Since the Minkowski sum is a regular hexagon with side 2, its apothem is 2 * sqrt(3)/2 = sqrt(3).
    # Wait. The Minkowski sum of two hexagons with side s has side 2s.
    # Apothem of a hexagon with side 2 is 2 * sqrt(3)/2 = sqrt(3).
    # So if the distance between centers is >= sqrt(3), they don't overlap?
    # Let's check.
    # Hex 1 at (0,0), Hex 2 at (sqrt(3), 0).
    # Hex 1 rightmost vertex: (1, 0). Hex 2 leftmost vertex: (sqrt(3)-1, 0) approx (0.732, 0).
    # They overlap! 0.732 < 1.
    # Why?
    # The Minkowski sum A+B contains all vectors v such that A and v-B intersect.
    # If c2 - c1 is in the interior of A+B, they overlap.
    # The Minkowski sum of two unit hexagons (flat-to-flat width sqrt(3), point-to-point width 2)
    # is a hexagon with vertices at distance 2 from center?
    # Vertices of unit hex: (1,0), (0.5, 0.866), ...
    # Vertices of Minkowski sum are sums of vertices.
    # (1,0) + (1,0) = (2,0).
    # (1,0) + (0.5, 0.866) = (1.5, 0.866).
    # (0.5, 0.866) + (0.5, 0.866) = (1, 1.732).
    # So the Minkowski sum has vertices at distance 2, 2, 2...
    # It is a regular hexagon with side length 2?
    # Distance between (2,0) and (1.5, 0.866) is sqrt(0.25 + 0.75) = 1.
    # So the Minkowski sum is a regular hexagon with side 1?
    # No, vertices are (2,0), (1.5, 0.866), (1, 1.732), (-1, 1.732), (-1.5, 0.866), (-2, 0).
    # Side length is 1.
    # Apothem of this Minkowski hexagon: distance from center to midpoint of edge (2,0)-(1.5, 0.866).
    # Midpoint: (1.75, 0.433). Distance: sqrt(1.75^2 + 0.433^2) = sqrt(3.0625 + 0.1875) = sqrt(3.25) approx 1.802.
    # Wait, the side length of the Minkowski sum of two hexagons of side 1 is 2?
    # Let's re-evaluate.
    # Hex A: vertices v_i. Hex B: vertices w_j.
    # Minkowski sum vertices are v_i + w_j.
    # If A and B are regular hexagons of side 1, the Minkowski sum is a regular hexagon of side 2.
    # Vertices at distance 2 from center.
    # Let's check the vertices again.
    # (1,0) + (1,0) = (2,0).
    # (1,0) + (0.5, 0.866) = (1.5, 0.866).
    # Distance between (2,0) and (1.5, 0.866) is 1.
    # So the side length is 1?
    # If the side length is 1, the apothem is sqrt(3)/2 approx 0.866.
    # But the distance to vertex (2,0) is 2.
    # This is not a regular hexagon.
    # Ah, the Minkowski sum of two regular hexagons is a regular hexagon.
    # Let's check the vertices of a regular hexagon with side 2.
    # Vertices at distance 2.
    # (2,0), (1, 1.732), (-1, 1.732), (-2, 0), (-1, -1.732), (1, -1.732).
    # Side length: distance (2,0) to (1, 1.732) is sqrt(1 + 3) = 2.
    # So the Minkowski sum has side 2.
    # My previous calculation of (1.5, 0.866) was a vertex of the sum?
    # (1,0) + (0.5, 0.866) = (1.5, 0.866).
    # Is this on the boundary?
    # The boundary of the Minkowski sum is the convex hull of all sums.
    # (1.5, 0.866) is inside the segment from (2,0) to (1, 1.732)?
    # Midpoint of (2,0) and (1, 1.732) is (1.5, 0.866).
    # Yes! So (1.5, 0.866) is the midpoint of the edge, not a vertex.
    # So the Minkowski sum is indeed a regular hexagon with side 2.
    # Its apothem is 2 * sqrt(3)/2 = sqrt(3).
    # So if the distance between centers is >= sqrt(3), they do NOT overlap?
    # Let's check the overlap case again.
    # Hex 1 at (0,0), Hex 2 at (1.5, 0).
    # Distance 1.5.
    # Hex 1 right vertex (1,0). Hex 2 left vertex (0.5, 0).
    # They overlap.
    # Distance 1.5 < sqrt(3) (1.732).
    # So if distance < sqrt(3), they overlap.
    # What if distance = 1.8?
    # Hex 1 right (1,0). Hex 2 left (0.8, 0). Overlap.
    # So the condition distance >= sqrt(3) is NOT sufficient for non-overlap.
    # Why?
    # The Minkowski sum condition is: c2 - c1 is NOT in the interior of A+B.
    # If c2 - c1 is on the boundary, they touch.
    # If c2 - c1 is outside, they are separated.
    # The Minkowski sum is a hexagon with apothem sqrt(3).
    # The distance from center to boundary is sqrt(3) in the direction of the normals (30, 90, ...).
    # In the direction of the vertices (0, 60, ...), the distance is 2.
    # So if the centers are separated along the x-axis (0 deg), the minimum distance is 2?
    # No.
    # If separated along x-axis, the vector is (d, 0).
    # The Minkowski sum boundary at angle 0 is at distance 2.
    # So if d >= 2, they are separated.
    # If d < 2, they overlap?
    # Let's check d=1.8.
    # Hex 1 right (1,0). Hex 2 left (0.8, 0). Overlap.
    # Yes.
    # So the "exclusion zone" is the Minkowski sum itself.
    # For two hexagons to be non-overlapping, the vector between their centers must lie OUTSIDE the Minkowski sum (or on boundary).
    # The Minkowski sum is a regular hexagon with side 2, apothem sqrt(3).
    # The condition is:
    # max over 6 normals n of dot(c2-c1, n) <= apothem = sqrt(3).
    # Wait.
    # If dot(c2-c1, n) <= apothem for all normals, then c2-c1 is INSIDE the Minkowski sum.
    # We want c2-c1 to be OUTSIDE.
    # So we want: max over 6 normals n of dot(c2-c1, n) > apothem?
    # No.
    # If c2-c1 is inside, they overlap.
    # So we need c2-c1 to be OUTSIDE.
    # Condition for OUTSIDE: max over 6 normals n of dot(c2-c1, n) > apothem?
    # Let's check.
    # If c2-c1 = (2, 0).
    # Normals at 30, 90, ...
    # dot((2,0), (cos30, sin30)) = 2 * 0.866 = 1.732 = sqrt(3).
    # dot((2,0), (cos90, sin90)) = 0.
    # Max is sqrt(3).
    # So it is on the boundary. They touch.
    # If c2-c1 = (1.5, 0).
    # dot((1.5,0), (cos30, sin30)) = 1.5 * 0.866 = 1.299.
    # This is < sqrt(3).
    # So it is INSIDE. They overlap.
    # So the condition for NON-overlap is:
    # max over 6 normals n of dot(c2-c1, n) >= sqrt(3).
    # Wait.
    # If I move them further apart, say (3,0).
    # dot((3,0), (cos30, sin30)) = 3 * 0.866 = 2.598.
    # This is > sqrt(3).
    # So the condition is max_proj >= sqrt(3).
    # This makes sense. The "width" of the exclusion zone in the normal direction is 2*apothem = 2*sqrt(3).
    # No, the apothem is the distance from center to edge.
    # So if the projection is larger than the apothem, it's outside.
    
    # So the constraint is:
    # For all pairs i, j:
    # max_k dot(c_j - c_i, n_k) >= sqrt(3)
    # where n_k are the 6 normals of the Minkowski sum (which are the same as the outer hexagon normals).
    
    # Let's verify this constraint.
    # If this holds, do they not overlap?
    # Yes, because the Minkowski sum is the set of all vectors that cause overlap.
    # If the center difference is outside the Minkowski sum, they don't overlap.
    
    # This is a convex constraint?
    # max_k dot(x, n_k) >= C is a non-convex constraint (complement of a convex set).
    # This makes optimization hard.
    # However, we can use a penalty method or just check validity at the end.
    # Or, we can use a simpler constraint: distance >= 2.0.
    # Distance >= 2.0 implies max_proj >= 2 * cos(30) = sqrt(3).
    # So distance >= 2.0 is a sufficient condition for non-overlap.
    # It is not necessary, but it's easy to handle.
    # Let's use distance >= 2.0 as the constraint.
    
    # Objective: Minimize outer side length S.
    # S = max over all inner vertices v of ( max_k dot(v - outer_center, n_k) ) / (sqrt(3)/2)
    # S = (2/sqrt(3)) * max_proj_outer
    
    # We want to minimize S.
    
    def get_outer_side(centers, outer_center):
        # centers: (11, 2)
        # outer_center: (2,)
        max_proj = 0.0
        for i in range(n):
            verts = inner_verts_rel + centers[i]
            verts_shifted = verts - outer_center
            projs = verts_shifted @ outer_normals.T
            max_proj = max(max_proj, np.max(projs))
        return max_proj * 2 / np.sqrt(3)
    
    def check_non_overlap(centers):
        # Check if all pairs satisfy distance >= 2.0
        # This is a sufficient condition.
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < 2.0:
                    return False
        return True
    
    def objective(params):
        centers = params[:2*n].reshape(n, 2)
        outer_center = params[2*n:2*n+2]
        
        side = get_outer_side(centers, outer_center)
        
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < 2.0:
                    penalty += (2.0 - d)**2
        
        return side + 100 * penalty
    
    # Initial guess: Compact cluster
    # 7 hexes in a ring around center, plus 4 more.
    # Let's use a random start or a grid start.
    np.random.seed(42)
    centers0 = np.random.rand(n, 2) * 4 - 2
    outer_center0 = np.array([0.0, 0.0])
    params0 = np.concatenate([centers0.flatten(), outer_center0])
    
    best_params = params0
    best_score = objective(params0)
    
    # Optimization
    # Nelder-Mead
    try:
        res = minimize(objective, params0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6})
        if res.fun < best_score:
            best_params = res.x
            best_score = res.fun
    except Exception:
        pass
    
    if time.time() - start_time > time_limit - 30:
        return _format_result(best_params)
    
    # L-BFGS-B
    try:
        res2 = minimize(objective, best_params, method='L-BFGS-B', options={'maxiter': 500})
        if res2.fun < best_score:
            best_params = res2.x
            best_score = res2.fun
    except Exception:
        pass
    
    # Local refinement: try to reduce side by moving outer center or inner centers slightly
    # We can also try to relax the distance constraint if we use the exact Minkowski check.
    # For now, stick with distance >= 2.0.
    
    return _format_result(best_params)

def _format_result(params):
    n = 11
    centers = params[:2*n].reshape(n, 2)
    outer_center = params[2*n:2*n+2]
    
    # Calculate side length
    inner_angles_rad = np.deg2rad([0, 60, 120, 180, 240, 300])
    inner_verts_rel = np.array([[np.cos(a), np.sin(a)] for a in inner_angles_rad])
    outer_normals_angles_rad = np.deg2rad([30, 90, 150, 210, 270, 330])
    outer_normals = np.array([[np.cos(a), np.sin(a)] for a in outer_normals_angles_rad])
    
    max_proj = 0.0
    for i in range(n):
        verts = inner_verts_rel + centers[i]
        verts_shifted = verts - outer_center
        projs = verts_shifted @ outer_normals.T
        max_proj = max(max_proj, np.max(projs))
        
    side = max_proj * 2 / np.sqrt(3)
    side += 1e-5
    
    inner_hex_data = np.zeros((n, 3))
    inner_hex_data[:, 0] = centers[:, 0]
    inner_hex_data[:, 1] = centers[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([outer_center[0], outer_center[1], 0.0])
    
    return inner_hex_data, outer_hex_data, float(side)

# EVOLVE-BLOCK-END
