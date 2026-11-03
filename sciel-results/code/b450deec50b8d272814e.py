"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    """
    Pack 12 unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    
    Strategy:
    1. Use a hexagonal lattice arrangement for the inner hexagons (all rotated 0 degrees relative to each other).
    2. Select the 12 lattice points that form the most compact cluster (minimizing the radius of the convex hull).
    3. Optimize the rotation angle of the entire cluster and the outer hexagon's rotation to minimize the required side length.
    """
    
    # --- Step 1: Generate candidate lattice points ---
    # A regular hexagon with side length 1 has a "radius" (center to vertex) of 1.
    # The distance between centers of two adjacent non-overlapping hexagons is sqrt(3).
    # Let d = sqrt(3). The lattice vectors are d * (1, 0) and d * (1/2, sqrt(3)/2).
    d = math.sqrt(3.0)
    
    # Generate a grid of potential center positions in axial coordinates (q, r)
    # Cartesian coordinates for axial (q, r):
    # x = d * (q + r/2)
    # y = d * (sqrt(3)/2) * r
    # Wait, standard axial to cartesian for pointy-top or flat-top?
    # Let's use flat-top hexagons aligned with x-axis for simplicity first, but the problem says "regular hexagon".
    # The inner hexagons can rotate. If we keep them all aligned with the outer hexagon, the packing is densest on a triangular lattice.
    # The minimum distance between centers of two non-overlapping unit hexagons is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
    # So the lattice spacing is sqrt(3).
    
    # Let's generate points in a square grid large enough and filter.
    # Axial coordinates (q, r) where x = d * (q + 0.5*r), y = d * (0.866 * r) ?
    # Actually, simpler: generate points in Cartesian directly from the triangular lattice.
    # Basis vectors: v1 = (d, 0), v2 = (d/2, d*sqrt(3)/2) = (d/2, 1.5)
    
    points = []
    # Search range -5 to 5 in axial coordinates to be safe
    max_ax = 6
    for q in range(-max_ax, max_ax + 1):
        for r in range(-max_ax, max_ax + 1):
            x = d * (q + 0.5 * r)
            y = d * (math.sqrt(3.0) / 2.0) * r
            points.append((x, y))
    
    points = np.array(points)
    
    # --- Step 2: Select the 12 most compact points ---
    # We want to choose 12 points such that the "radius" of the cluster is minimized.
    # A good proxy is the distance from the centroid. We want the set of 12 points closest to the origin (after centering).
    # However, simply taking the 12 closest to (0,0) might not be optimal if the centroid isn't (0,0).
    # Let's try all combinations? No, too slow.
    # Heuristic: The most compact cluster of 12 hexagons on a triangular lattice is known.
    # It is a "hexagonal" shape but with 12 hexes, it's not a perfect hexagon (1, 7, 19...).
    # 12 hexes form a 3x4 rectangle in axial coords? Or a distorted hexagon.
    # Let's use a greedy approach or just pick the 12 points with the smallest distance to the origin,
    # assuming the lattice is centered at the origin.
    
    # Calculate distances from origin
    dists = np.linalg.norm(points, axis=1)
    # Get indices of the 12 smallest distances
    idx = np.argsort(dists)[:N]
    centers = points[idx]
    
    # Center the cluster
    center_offset = np.mean(centers, axis=0)
    centers = centers - center_offset
    
    # --- Step 3: Determine the minimum enclosing hexagon ---
    # The inner hexagons have side length 1.
    # The outer hexagon has side length S.
    # Containment condition: The Minkowski sum of the inner hexagon (rotated by theta) and the set of centers must be contained in the outer hexagon (rotated by phi).
    # Equivalently, the set of centers must be contained in (Outer Hexagon (rotated by phi)) MINUS (Inner Hexagon (rotated by theta)).
    # The "Minus" operation (Minkowski difference) of two regular hexagons is another regular hexagon (if aligned) or a 12-gon (if misaligned).
    # To minimize S, it is best to align the inner and outer hexagons (theta = phi).
    # If theta = phi = 0, the effective containment region for the centers is a regular hexagon with side length S_inner_effective.
    # The apothem of the inner hexagon is a_i = sqrt(3)/2.
    # The apothem of the outer hexagon is a_o = S * sqrt(3)/2.
    # The centers must lie within a hexagon of apothem a_o - a_i.
    # So, required S = (Apothem_required + a_i) / (sqrt(3)/2).
    # Where Apothem_required is the maximum distance from the center to the boundary of the convex hull of the centers in the hexagonal metric.
    # Actually, for aligned hexagons, the constraint is:
    # For each direction k in the 6 normal directions, the projection of all centers onto that direction must be <= a_o - a_i.
    # So, a_o = max_projection + a_i.
    # S = (max_projection + a_i) * 2 / sqrt(3).
    
    a_i = math.sqrt(3.0) / 2.0
    
    def calc_side_length(centers, rotation_outer):
        """
        Calculate the minimum side length S of the outer hexagon (rotated by rotation_outer)
        that contains the given centers, assuming inner hexagons are aligned with the outer one.
        """
        # The normal directions for a regular hexagon rotated by rotation_outer
        # The normals are at angles: rotation_outer + k * 60 degrees
        angles = rotation_outer + np.arange(6) * np.pi / 3.0
        normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        
        # Project centers onto these normals
        # projections[i, j] = dot(centers[i], normals[j])
        projections = centers @ normals.T
        
        # The maximum extent in each normal direction
        max_extents = np.max(projections, axis=0)
        
        # The required apothem of the outer hexagon
        required_apothem = np.max(max_extents) + a_i
        
        # Convert apothem to side length
        # apothem = S * sqrt(3) / 2  => S = apothem * 2 / sqrt(3)
        S = required_apothem * 2.0 / math.sqrt(3.0)
        
        return S

    # --- Step 4: Optimize the rotation of the cluster ---
    # We can rotate the entire set of centers by an angle alpha to minimize S.
    # Note: Rotating the centers by alpha is equivalent to rotating the outer hexagon by -alpha relative to the fixed inner hexagons.
    # Since we assume inner and outer are aligned, we just need to find the optimal orientation of the cluster.
    
    def objective(alpha):
        # Rotate centers by alpha
        c, s = np.cos(alpha), np.sin(alpha)
        R = np.array([[c, -s], [s, c]])
        rotated_centers = centers @ R.T
        # The outer hexagon is fixed at angle 0 for this calculation, so we pass rotation_outer=0
        return calc_side_length(rotated_centers, 0.0)
    
    # Optimize alpha in [0, pi/3] due to hexagonal symmetry
    alpha_opt = 0.0
    best_S = float('inf')
    # Simple grid search + local optimization
    alphas = np.linspace(0, np.pi/3, 100)
    for a in alphas:
        val = objective(a)
        if val < best_S:
            best_S = val
            alpha_opt = a
    
    # Refine with scipy minimize
    res = minimize(objective, alpha_opt, method='Nelder-Mead', bounds=[(0, np.pi/3)], options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 1000})
    if res.success and res.fun < best_S:
        alpha_opt = res.x[0]
        best_S = res.fun
    
    # --- Step 5: Construct the final output ---
    # Rotate the centers by alpha_opt
    c, s = np.cos(alpha_opt), np.sin(alpha_opt)
    R = np.array([[c, -s], [s, c]])
    final_centers = centers @ R.T
    
    # The inner hexagons are aligned with the outer hexagon.
    # We set the outer hexagon rotation to 0.0 for simplicity, and the inner hexagons to 0.0.
    # This is valid because the problem allows arbitrary rotation for each inner hexagon and the outer hexagon.
    # By rotating the whole cluster, we've effectively aligned the "tightest" direction with one of the hexagon normals.
    
    inner = []
    for x, y in final_centers:
        inner.append([float(x), float(y), 0.0])
    
    outer_center = [0.0, 0.0]
    side_length = float(best_S)
    outer_angle = 0.0
    
    return inner, outer_center, side_length, outer_angle

# EVOLVE-BLOCK-END
