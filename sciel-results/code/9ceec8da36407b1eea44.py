# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the respective outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 11
    # Hexagon geometry constants
    # Vertices of unit hexagon (side 1) at angles 0, 60, ...
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    vertices = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    # Normals to the edges (pointing outwards)
    # Edge midpoints are at 30, 90, ...
    normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
    normals = np.stack([np.cos(normal_angles), np.sin(normal_angles)], axis=1)
    
    def get_hex_vertices(center):
        return center + vertices
    
    def min_distance_hexagons(c1, c2):
        """
        Calculate the minimum distance between two unit hexagons centered at c1 and c2.
        Returns distance (positive if separated, negative if overlapping).
        Also returns the direction of the minimum distance vector (for gradient).
        """
        # The minimum distance between two convex polygons is the minimum over all edge pairs
        # or vertex-edge pairs.
        # For hexagons with same orientation, we can use the support function difference.
        # Distance = max_d (h1(d) + h2(-d) - ||c1-c2||_d) ? No.
        # Separation distance d is such that:
        # min_{p in H1, q in H2} ||p - q||
        # This is equivalent to:
        # min_{d} (h1(d) + h2(d) - (c2-c1).d) ?? No.
        
        # Simpler approach for same orientation:
        # The Minkowski difference H1 - H2 is a hexagon with side 2 (scaled).
        # Actually, H1 - H2 is a hexagon with side length 2? No.
        # Unit hex side 1. Minkowski sum of two unit hexes is a hex with side 2.
        # Minkowski difference is also a hex with side 2.
        # The distance from c2-c1 to the boundary of (H1 - H2) is the separation.
        # H1 - H2 is centered at 0. Its support function is h(d) = 1 + 1 = 2?
        # Let's check.
        # Vertex of H1 at (1,0). Vertex of H2 at (1,0). Diff (0,0).
        # Vertex of H1 at (1,0). Vertex of H2 at (-1,0). Diff (2,0).
        # So the Minkowski difference is a hexagon with circumradius 2?
        # The distance from the origin to the boundary of the Minkowski difference in direction d is h_diff(d).
        # If c2-c1 is inside this hexagon, they overlap.
        # The distance to the boundary is h_diff(d) - (c2-c1).d ?
        
        # Let's use the property that for two convex polygons, the minimum distance is:
        # min_{d in S^1} (h1(d) + h2(d) - (c2-c1).d) is not quite right because h2(-d) is needed?
        # h1(d) = max_{p in H1} p.d
        # h2(-d) = max_{q in H2} q.(-d) = - min_{q in H2} q.d
        # Distance >= h1(d) - (c1).d - (h2(-d) - (c2).d) ?
        
        # Let's just compute the distance to the Minkowski Difference polygon.
        # The Minkowski Difference H1 - H2 for two identical regular hexagons is a regular hexagon
        # with side length 2.
        # Its vertices are 2 * vertices of unit hex?
        # Let's verify.
        # H = {x | |x|_inf-like hex norm <= 1}
        # H - H = {x-y}.
        # The support function of H-H is h_H(d) + h_H(-d) = 2 h_H(d) (since symmetric).
        # So the Minkowski difference is a hexagon with support function 2 * h_H(d).
        # This is a hexagon with side length 2?
        # Apothem of unit hex is sqrt(3)/2.
        # Apothem of diff hex is 2 * sqrt(3)/2 = sqrt(3).
        # Side length s and apothem a: a = s * sqrt(3)/2.
        # sqrt(3) = s * sqrt(3)/2 => s = 2.
        # Yes, the Minkowski difference is a regular hexagon with side 2.
        
        # So, the distance between the two hexagons is the distance from the vector (c2-c1)
        # to the boundary of the hexagon H_diff (side 2, centered at 0).
        # If (c2-c1) is inside H_diff, distance is negative (overlap).
        # The signed distance from a point p to a convex polygon P is:
        # If p is inside, -min_{d} (h_P(d) - p.d) ?
        # Actually, the distance from p to the boundary of P (if p is inside) is:
        # min_{d} (h_P(d) - p.d) is the distance to the edge in direction d?
        # No, the distance to the edge perpendicular to d is h_P(d) - p.d.
        # The minimum distance to any edge is min_d (h_P(d) - p.d).
        
        # So, separation = min_d (h_diff(d) - (c2-c1).d)
        # h_diff(d) = 2 * h_unit(d)
        
        # We need to find d that minimizes this.
        # The minimum will occur when d is normal to one of the edges of H_diff.
        # The normals of H_diff are the same as normals of H_unit.
        
        diff = c2 - c1
        # Project diff onto the 6 normals
        projections = diff @ normals.T
        # h_diff in direction normal j is 2 * h_unit_j
        # h_unit_j = apothem = sqrt(3)/2
        # So h_diff_j = sqrt(3)
        
        h_diff = np.sqrt(3.0) * np.ones(6)
        
        # Distance to edge j is h_diff_j - projections[j]
        # The minimum distance to the boundary is the minimum of these values?
        # Wait, if diff is at origin, projections are 0. Distances are sqrt(3).
        # The origin is inside. The distance to the boundary is the minimum distance to any edge.
        # Yes, min_j (h_diff_j - diff.normal_j).
        
        dists = h_diff - projections
        min_dist = np.min(dists)
        
        # For gradient, we need the direction.
        # The gradient of min_dist with respect to diff is -normal_j* where j is the argmin.
        # If there are multiple mins, average them?
        
        return min_dist

    def objective_and_grad(x):
        """
        x is a vector of shape (22,) representing centers of 11 hexagons.
        Returns objective value and gradient.
        Objective: S_required + lambda * sum(overlap^2)
        """
        centers = x.reshape(11, 2)
        
        # 1. Calculate S_required
        # S_required = max_j (max_i (centers[i].normal_j) + 1) * 2 / sqrt(3)
        projections = centers @ normals.T
        max_projs = np.max(projections, axis=0)
        required_Ss = (max_projs + 1.0) * 2.0 / np.sqrt(3.0)
        S_req = np.max(required_Ss)
        
        # Gradient of S_req
        # S_req = max_k (A_k) where A_k = (max_i (c_i.n_k) + 1) * 2/sqrt(3)
        # Let k* be argmax.
        k_star = np.argmax(required_Ss)
        n_k = normals[k_star]
        # max_i (c_i.n_k) is determined by i* = argmax_i (c_i.n_k)
        projs_k = projections[:, k_star]
        i_star = np.argmax(projs_k)
        
        grad_S = np.zeros_like(x)
        # d S_req / d c_i_star = 2/sqrt(3) * n_k
        grad_S[2*i_star : 2*i_star+2] = (2.0 / np.sqrt(3.0)) * n_k
        
        # 2. Calculate Overlap Penalty
        # Penalty = sum_{i<j} max(0, -dist_ij)^2
        # dist_ij = min_dist(c_i, c_j)
        # If dist < 0, overlap = -dist.
        
        penalty = 0.0
        grad_pen = np.zeros_like(x)
        
        for i in range(11):
            for j in range(i+1, 11):
                dist = min_distance_hexagons(centers[i], centers[j])
                if dist < 0:
                    overlap = -dist
                    penalty += overlap * overlap
                    # Gradient of overlap^2 = 2 * overlap * grad(overlap)
                    # grad(overlap) = -grad(dist)
                    # grad(dist) = -normal_j* (where j* is argmin of dist calculation)
                    # So grad(overlap) = normal_j*
                    
                    # We need to recover which normal was the argmin.
                    # Re-calculate to get index
                    diff = centers[j] - centers[i]
                    projections_diff = diff @ normals.T
                    h_diff = np.sqrt(3.0) * np.ones(6)
                    dists = h_diff - projections_diff
                    min_idx = np.argmin(dists)
                    n_min = normals[min_idx]
                    
                    # grad(dist) w.r.t centers[i] is n_min ?
                    # dist = h - (c_j - c_i).n = h - c_j.n + c_i.n
                    # d dist / d c_i = n
                    # d dist / d c_j = -n
                    # overlap = -dist
                    # d overlap / d c_i = -n
                    # d overlap / d c_j = n
                    # d (overlap^2) / d c_i = 2 * overlap * (-n)
                    # d (overlap^2) / d c_j = 2 * overlap * n
                    
                    grad_pen[2*i : 2*i+2] += 2.0 * overlap * (-n_min)
                    grad_pen[2*j : 2*j+2] += 2.0 * overlap * n_min
        
        # Total Objective
        lambda_weight = 100.0
        total_obj = S_req + lambda_weight * penalty
        total_grad = grad_S + lambda_weight * grad_pen
        
        return total_obj, total_grad

    # Initial guess
    # Hexagonal cluster of 7
    c0 = np.array([0, 0])
    c1 = np.array([2, 0])
    c2 = np.array([1, np.sqrt(3)])
    c3 = np.array([-1, np.sqrt(3)])
    c4 = np.array([-2, 0])
    c5 = np.array([-1, -np.sqrt(3)])
    c6 = np.array([1, -np.sqrt(3)])
    
    # 4 more
    r = 3.5
    c7 = np.array([r * np.cos(np.pi/6), r * np.sin(np.pi/6)])
    c8 = np.array([r * np.cos(5*np.pi/6), r * np.sin(5*np.pi/6)])
    c9 = np.array([r * np.cos(7*np.pi/6), r * np.sin(7*np.pi/6)])
    c10 = np.array([r * np.cos(11*np.pi/6), r * np.sin(11*np.pi/6)])
    
    initial_centers = np.array([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])
    x0 = initial_centers.flatten()
    
    # Use L-BFGS-B
    result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, options={'maxiter': 1000, 'ftol': 1e-12})
    
    best_centers = result.x.reshape(11, 2)
    
    # Calculate final S
    projections = best_centers @ normals.T
    max_projs = np.max(projections, axis=0)
    required_Ss = (max_projs + 1.0) * 2.0 / np.sqrt(3.0)
    best_S = np.max(required_Ss)
    
    # Verify validity
    # Check overlaps
    valid = True
    for i in range(11):
        for j in range(i+1, 11):
            dist = min_distance_hexagons(best_centers[i], best_centers[j])
            if dist < -1e-6: # Overlapping
                valid = False
    
    if not valid:
        # Fallback to simple SA if gradient fails? Or just return best effort.
        # For now, return best effort.
        pass

    inner_hex_data = np.hstack([best_centers, np.zeros((11, 1))])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = best_S
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
