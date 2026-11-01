"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices_origin():
    """Vertices of unit hexagon at origin, angle 0."""
    angles = np.radians(60 * np.arange(6))
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _objective(params):
    # params: 2N array [x1, y1, x2, y2, ..., xN, yN]
    centers = params.reshape(N, 2)
    
    # 1. Check disjointness
    # Distance between centers must be >= sqrt(3)
    # Using broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Exclude self
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    if min_dist < np.sqrt(3):
        # Penalty proportional to violation
        return 1e6 + (np.sqrt(3) - min_dist) * 1000

    # 2. Compute min enclosing hexagon side
    # Precompute support functions for unit hexagon at angle 0
    # We need max/min projection onto normals for various angles
    # Since the inner hexagons are all at angle 0, their support function is fixed.
    # h(theta) for unit hexagon.
    
    best_side = float('inf')
    
    # Scan outer rotation
    thetas = np.linspace(0, 60, 60)
    sqrt3 = np.sqrt(3)
    
    for theta_deg in thetas:
        theta = np.radians(theta_deg)
        # Normals for the outer hexagon edges
        # Edge normals are at theta + 30 + k*60
        normals_angles = np.radians(30 + 60 * np.arange(6)) + theta
        normals = np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])
        
        # Support of unit hexagon (angle 0) onto these normals
        # verts0 is 6x2
        # projections: 6x6
        # We need max and min for each normal
        # Actually, we can just compute the width of the unit hexagon along each normal
        # Width of unit hexagon along direction n is max(v.n) - min(v.n)
        # But we need max and min separately to combine with centers.
        
        # Let's compute h_max and h_min for the unit hexagon for all 6 normals at once
        # verts0 @ normals.T -> 6x6
        # max along axis 0 -> 6 values
        # min along axis 0 -> 6 values
        
        # To avoid recomputing verts0 every time, we can pass it or make it global.
        # Let's make it global in the block.
        
        pass # Placeholder

# We need to restructure to use global verts0
verts0_global = _hex_vertices_origin()

def _objective_optimized(params):
    centers = params.reshape(N, 2)
    
    # 1. Check disjointness
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    if min_dist < np.sqrt(3):
        return 1e6 + (np.sqrt(3) - min_dist) * 1000

    best_side = float('inf')
    thetas = np.linspace(0, 60, 60)
    sqrt3 = np.sqrt(3)
    
    for theta_deg in thetas:
        theta = np.radians(theta_deg)
        normals_angles = np.radians(30 + 60 * np.arange(6)) + theta
        normals = np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])
        
        # Support of unit hexagon
        # verts0_global is 6x2
        # normals is 6x2
        # We want for each normal j, max_i (v_i . n_j) and min_i (v_i . n_j)
        # proj = verts0_global @ normals.T  (6x6)
        h_max = np.max(verts0_global @ normals.T, axis=0) # 6 values
        h_min = np.min(verts0_global @ normals.T, axis=0) # 6 values
        
        # Projections of centers
        # centers is 12x2, normals is 6x2
        # center_projs = centers @ normals.T (12x6)
        center_projs = centers @ normals.T
        
        max_center_proj = np.max(center_projs, axis=0) # 6 values
        min_center_proj = np.min(center_projs, axis=0) # 6 values
        
        # Total max/min for each direction
        total_max = max_center_proj + h_max
        total_min = min_center_proj + h_min
        
        widths = total_max - total_min
        side = np.max(widths) / sqrt3
        
        if side < best_side:
            best_side = side
            
    return best_side

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Initial guess: Honeycomb
    s3 = np.sqrt(3)
    centers = []
    for r in range(3):
        for c in range(4):
            x = c * s3
            y = r * 1.5
            if r % 2 == 1:
                x += s3 / 2.0
            centers.append([x, y])
            
    centers = np.array(centers)
    # Scale slightly to ensure initial validity
    centers = centers * 1.01
    
    x0 = centers.flatten()
    
    result = minimize(_objective_optimized, x0, method='Nelder-Mead', 
                      options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
    
    if result.success or result.fun < 1e5:
        best_centers = result.x.reshape(N, 2)
        # Re-calculate the exact best outer hexagon for the final centers
        # We need the center and angle of the outer hexagon too.
        
        # Find best angle and center
        # The objective function found the min side, but not the center/angle.
        # We need to recover them.
        
        best_side = float('inf')
        best_angle = 0.0
        best_center = np.mean(best_centers, axis=0)
        
        thetas = np.linspace(0, 60, 600) # Finer scan for final result
        sqrt3 = np.sqrt(3)
        
        for theta_deg in thetas:
            theta = np.radians(theta_deg)
            normals_angles = np.radians(30 + 60 * np.arange(6)) + theta
            normals = np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])
            
            h_max = np.max(verts0_global @ normals.T, axis=0)
            h_min = np.min(verts0_global @ normals.T, axis=0)
            
            center_projs = best_centers @ normals.T
            max_center_proj = np.max(center_projs, axis=0)
            min_center_proj = np.min(center_projs, axis=0)
            
            total_max = max_center_proj + h_max
            total_min = min_center_proj + h_min
            
            widths = total_max - total_min
            side = np.max(widths) / sqrt3
            
            if side < best_side:
                best_side = side
                best_angle = theta_deg
                
        # The center of the enclosing hexagon is the midpoint of the tightest constraints?
        # Actually, the center calculated by `np.mean(best_centers, axis=0)` might not be the optimal center for the enclosing hexagon.
        # However, for a symmetric-ish packing, it's close.
        # To be precise, the optimal center minimizes the max width.
        # For a fixed angle, the optimal center is the center of the minimum enclosing hexagon.
        # We can approximate it by the centroid of the "extreme" points.
        # Or just use the centroid. The evaluator checks if the hexagon contains the points.
        # If we use the centroid, and the side length is calculated based on projections from the centroid, it is valid.
        # Wait, my objective function calculated `side` based on `widths`, which are differences between max and min projections.
        # The width is independent of the center of the enclosing hexagon.
        # The side length is $W / \sqrt{3}$.
        # The center of the enclosing hexagon should be such that the projections are centered.
        # i.e., $Center_{proj} = (total\_max + total\_min) / 2$.
        # We should calculate the optimal center for the best angle.
        
        # Let's re-calculate the optimal center for best_angle
        theta = np.radians(best_angle)
        normals_angles = np.radians(30 + 60 * np.arange(6)) + theta
        normals = np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])
        
        h_max = np.max(verts0_global @ normals.T, axis=0)
        h_min = np.min(verts0_global @ normals.T, axis=0)
        
        center_projs = best_centers @ normals.T
        max_center_proj = np.max(center_projs, axis=0)
        min_center_proj = np.min(center_projs, axis=0)
        
        total_max = max_center_proj + h_max
        total_min = min_center_proj + h_min
        
        # The center of the enclosing hexagon in projection space for each normal k
        # is (total_max[k] + total_min[k]) / 2
        # But the center is a single 2D point.
        # This is only consistent if the hexagon is perfectly centered.
        # A better estimate: The center is the intersection of the midlines.
        # For simplicity, use the average of the midpoints in projection space? No, that's not a point.
        # Use the centroid of the best_centers as the outer center.
        # Is it valid?
        # If we place the outer hexagon at `centroid` with side `best_side` and angle `best_angle`,
        # does it contain all inner hexagons?
        # The condition is that for all normals k, the projection of the outer hexagon contains the projection of the inner set.
        # Outer hexagon projection range at centroid C:
        # $C \cdot n_k \pm (side \cdot \sqrt{3}/2)$
        # Inner set projection range:
        # $[min\_proj\_k, max\_proj\_k]$
        # We need $C \cdot n_k - s\sqrt{3}/2 \le min\_proj\_k$ and $C \cdot n_k + s\sqrt{3}/2 \ge max\_proj\_k$.
        # This implies $C \cdot n_k \le min\_proj\_k + s\sqrt{3}/2$ and $C \cdot n_k \ge max\_proj\_k - s\sqrt{3}/2$.
        # So $C \cdot n_k$ must be in $[max\_proj\_k - s\sqrt{3}/2, min\_proj\_k + s\sqrt{3}/2]$.
        # The length of this interval is $min\_proj\_k + max\_proj\_k - 2(s\sqrt{3}/2) + ...$
        # Actually, $s = (max\_proj - min\_proj) / \sqrt{3}$.
        # So $s\sqrt{3}/2 = (max\_proj - min\_proj)/2$.
        # Interval: $[max\_proj - (max\_proj - min\_proj)/2, min\_proj + (max\_proj - min\_proj)/2] = [ (max\_proj+min\_proj)/2, (max\_proj+min\_proj)/2 ]$.
        # So $C \cdot n_k$ MUST be exactly $(max\_proj\_k + min\_proj\_k)/2$.
        # This must hold for all k=0..5.
        # This determines C uniquely (if consistent).
        
        # Let's solve for C.
        # We have 6 equations: $C \cdot n_k = M_k$, where $M_k = (total\_max[k] + total\_min[k])/2$.
        # We only need 2 independent normals to find C.
        # Let's use normals 0 and 1.
        
        n0 = normals[0]
        n1 = normals[1]
        M0 = (total_max[0] + total_min[0]) / 2
        M1 = (total_max[1] + total_min[1]) / 2
        
        # Solve [n0; n1] @ C = [M0; M1]
        A = np.array([n0, n1])
        b = np.array([M0, M1])
        
        try:
            C = np.linalg.solve(A, b)
        except:
            C = np.mean(best_centers, axis=0)
            
        return [[float(best_centers[i][0]), float(best_centers[i][1]), 0.0] for i in range(N)], (float(C[0]), float(C[1])), float(best_side), float(best_angle)
    else:
        # Fallback to initial guess
        centers = x0.reshape(N, 2)
        return [[float(centers[i][0]), float(centers[i][1]), 0.0] for i in range(N)], (0.0, 0.0), 10.0, 0.0
# EVOLVE-BLOCK-END
