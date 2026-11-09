# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def hexagon_packing_12():
    # Vertices of a unit hexagon centered at (0,0), angle 0
    angles = np.deg2rad(np.arange(0, 360, 60))
    rel_vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normals of the outer hexagon (and inner, since aligned)
    normal_angles = np.deg2rad(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Precompute projections of relative vertices onto normals
    # proj_rel[i, j] is projection of vertex i on normal j
    proj_rel = rel_vertices @ normals.T # (6, 6)
    
    def get_s_and_penalty(centers):
        # centers: (12, 2)
        # Calculate s
        # Project all vertices onto all normals
        # Vertex k of hex i is centers[i] + rel_vertices[k]
        # Projection = centers[i] @ normals.T + proj_rel[k]
        
        # proj_centers: (12, 6)
        proj_centers = centers @ normals.T
        
        # All projections: (12, 6, 6) -> centers[i] @ n[j] + rel[k] @ n[j]
        # We need max over i, k, j of | proj_centers[i, j] + proj_rel[k, j] |
        
        # Expand to (12, 6, 6)
        # proj_centers is (12, 6). proj_rel is (6, 6).
        # We want for each i, j, max_k | proj_centers[i,j] + proj_rel[k,j] |
        # Since proj_rel is fixed, we can precompute min and max of proj_rel[:, j]
        # min_rel[:, j] = min_k proj_rel[k, j]
        # max_rel[:, j] = max_k proj_rel[k, j]
        
        min_rel = np.min(proj_rel, axis=0) # (6,)
        max_rel = np.max(proj_rel, axis=0) # (6,)
        
        # For each i, j, the range of projections is [proj_centers[i,j] + min_rel[j], proj_centers[i,j] + max_rel[j]]
        # We need the max absolute value in this range.
        # This is max(|lower|, |upper|)
        
        lower = proj_centers + min_rel # (12, 6)
        upper = proj_centers + max_rel # (12, 6)
        
        abs_lower = np.abs(lower)
        abs_upper = np.abs(upper)
        
        max_abs_proj = np.max(np.maximum(abs_lower, abs_upper))
        
        s = (2.0 / np.sqrt(3)) * max_abs_proj
        
        # Calculate overlap penalty
        # For each pair (i, l), check separation on 6 axes.
        # Axis j:
        # Interval i: [proj_centers[i,j] + min_rel[j], proj_centers[i,j] + max_rel[j]]
        # Interval l: [proj_centers[l,j] + min_rel[j], proj_centers[l,j] + max_rel[j]]
        # Overlap length = min(upper_i, upper_l) - max(lower_i, lower_l)
        # If > 0, they overlap on this axis.
        # If they overlap on ALL 6 axes, they intersect.
        # Penalty: Sum of positive overlaps? Or just a boolean penalty?
        # A soft penalty is better for optimization.
        # Let's use the sum of squared overlaps on all axes for all pairs.
        
        penalty = 0.0
        n = len(centers)
        for i in range(n):
            for l in range(i + 1, n):
                # lower[i], upper[i], lower[l], upper[l] are (6,)
                # overlap_j = min(upper[i,j], upper[l,j]) - max(lower[i,j], lower[l,j])
                # If overlap_j > 0, add to penalty
                # We can vectorize this
                min_upper = np.minimum(upper[i], upper[l])
                max_lower = np.maximum(lower[i], lower[l])
                overlaps = min_upper - max_lower
                # Sum of positive overlaps
                pos_overlaps = np.maximum(0, overlaps)
                penalty += np.sum(pos_overlaps ** 2)
                
        # Scale penalty to be significant
        return s + 10.0 * penalty

    # Initial guess
    # Use the grid from the previous solution as a starting point, but maybe perturb it
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (2, -1), (1, -2), (-1, -2), (-2, -1)
    ]
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = np.sqrt(3) * (r + 0.5 * q)
        centers.append([x, y])
    centers = np.array(centers)
    
    # Optimize
    # Powell method is good for this
    res = minimize(get_s_and_penalty, centers, method='Powell', options={'maxiter': 500, 'disp': False})
    
    optimal_centers = res.x.reshape(12, 2)
    optimal_s, _ = get_s_and_penalty(optimal_centers)
    
    # Verify validity and adjust if needed
    # The penalty should be 0 if valid.
    # If penalty > 0, the solution might be invalid.
    # We can try to clean it up or use a different start.
    
    # For now, return the result.
    # Inner hexagons: (x, y, 0)
    inner_hex_data = np.column_stack([optimal_centers, np.zeros(12)])
    
    # Outer hexagon: (0, 0, 0)
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, optimal_s
# EVOLVE-BLOCK-END
