# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses Gradient Descent (L-BFGS-B) on a penalty function to minimize the outer hexagon size.
    Variables: centers (12, 2) and angles (12,).
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit
    
    sqrt3 = math.sqrt(3)
    N = 12
    
    # Helper: Get vertices of a hexagon given center, side, angle
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + side * math.cos(a)
            y = cy + side * math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    # Helper: Compute the minimum side length s required to contain a set of hexagons
    # Outer hexagon is centered at origin, angle 0.
    # Normals of outer hexagon edges are at 30, 90, 150, 210, 270, 330 degrees.
    normal_angles = [math.radians(30 + i*60) for i in range(6)]
    normals = np.array([[math.cos(a), math.sin(a)] for a in normal_angles])
    
    def calc_min_s(centers, angles):
        """
        Returns the minimum side length s such that all hexagons fit in the outer hexagon.
        """
        max_proj = 0.0
        for i in range(N):
            cx, cy = centers[i]
            ang = angles[i]
            # Vertices of the i-th hexagon
            verts = get_hex_vertices(cx, cy, 1.0, ang)
            # Project onto the 6 normals of the outer hexagon
            projs = verts @ normals.T
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        # The apothem of the outer hexagon is s * sqrt(3)/2
        # So s * sqrt(3)/2 >= max_proj  =>  s >= 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / sqrt3

    def check_overlap(c1, a1, c2, a2):
        """
        Returns the penetration depth (positive if overlapping, negative if separated).
        Uses SAT (Separating Axis Theorem) with the 12 normals of the two hexagons.
        For efficiency in gradient calculation, we approximate the penalty using the distance between centers
        projected onto the normals, but for robustness in the objective, we use a smooth approximation.
        
        Actually, for L-BFGS-B, we need a differentiable objective.
        The exact SAT separation is non-differentiable at boundaries.
        We will use a penalty based on the minimum distance between edges or centers.
        A simpler differentiable proxy: 
        If we assume the hexagons are small enough or close enough, we can use the distance between centers.
        But they can rotate.
        
        Let's use a robust penalty: 
        For each pair, calculate the minimum separation distance using SAT.
        If separated, penalty = 0.
        If overlapping, penalty = overlap_depth^2.
        This is non-smooth. To make it smooth, we can use a soft-min or just rely on the optimizer's robustness.
        However, L-BFGS-B assumes smoothness.
        
        Alternative: Use the distance between centers as a primary constraint if they are far, 
        but this fails for rotated hexagons.
        
        Let's stick to a penalty that is differentiable almost everywhere.
        We will compute the projection intervals for all 12 axes (6 from hex1, 6 from hex2).
        For each axis, if intervals overlap, we accumulate a penalty.
        """
        v1 = get_hex_vertices(c1[0], c1[1], 1.0, a1)
        v2 = get_hex_vertices(c2[0], c2[1], 1.0, a2)
        
        # Axes from v1
        e1 = np.roll(v1, -1, axis=0) - v1
        n1 = np.column_stack((e1[:, 1], -e1[:, 0]))
        # Axes from v2
        e2 = np.roll(v2, -1, axis=0) - v2
        n2 = np.column_stack((e2[:, 1], -e2[:, 0]))
        
        all_normals = np.vstack((n1, n2))
        
        penalty = 0.0
        for n in all_normals:
            proj1 = v1 @ n
            proj2 = v2 @ n
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # If separated, gap = min2 - max1 or min1 - max2
            # If overlapping, gap is negative
            gap1 = min2 - max1
            gap2 = min1 - max2
            gap = max(gap1, gap2)
            
            if gap < 0:
                # Overlapping. Penalty is square of overlap depth.
                # To make it smoother, we can use a sigmoid or just square.
                penalty += (-gap) ** 2
        
        return penalty

    def objective(params):
        """
        Objective function for the optimizer.
        params: (36,) array = [x1, y1, th1, x2, y2, th2, ..., x12, y12, th12]
        """
        centers = params[0:24].reshape(12, 2)
        angles = params[24:36]
        
        # 1. Calculate s
        s = calc_min_s(centers, angles)
        
        # 2. Calculate overlap penalty
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                c1 = centers[i]
                a1 = angles[i]
                c2 = centers[j]
                a2 = angles[j]
                # Quick distance check to avoid expensive SAT if far apart
                dist = np.linalg.norm(c1 - c2)
                if dist < 3.5: # Max distance for unit hexagons to touch is ~3.46
                    overlap_pen += check_overlap(c1, a1, c2, a2)
        
        # 3. Containment penalty (redundant if s is minimized, but helps guide gradient)
        # We want s to be small. The objective is to minimize s + lambda * penalty.
        # However, s is determined by the max projection.
        # If we just minimize s, the hexagons will try to shrink s, which forces them to move in.
        # But if they overlap, s might be small but invalid.
        # So we minimize s + large_constant * overlap_penalty.
        
        # We also need to ensure the outer hexagon is large enough to contain them?
        # No, s is the variable we are minimizing. The "outer hexagon" is defined by s.
        # The constraint is that the inner hexagons must be INSIDE the outer hexagon of side s.
        # By definition of calc_min_s, if s is the min side length, they are inside.
        # Wait, calc_min_s returns the *required* s.
        # So if we minimize the required s, we are finding the tightest bounding hexagon.
        # The "outer hexagon" in the output will have side length s.
        # The inner hexagons are fixed at side 1.
        # So the only constraint is non-overlap.
        # The containment is automatically satisfied by the definition of s as the bounding size.
        
        # Therefore, the objective is simply:
        # Minimize (required_s + 1000 * overlap_penalty)
        
        # We add a small regularization to keep centers near origin to prevent drift
        reg = 0.01 * np.sum(centers**2)
        
        return s + 100.0 * overlap_pen + reg

    # Initial guess: Hexagonal cluster
    # 12 hexagons in a hexagonal arrangement:
    # Center (0,0)
    # Ring 1: 6 hexagons at distance 2*sqrt(3)/2 * 2? No.
    # Distance between centers of touching hexagons is 2 * apothem = sqrt(3) * 2 / 2 * 2?
    # Side 1 hexagon. Apothem = sqrt(3)/2.
    # Distance between centers of touching hexagons = 2 * apothem = sqrt(3).
    # Wait, if they are flat-topped, horizontal distance is 1.5? No.
    # Flat-topped hexagon:
    # Vertices at (1,0), (0.5, sqrt3/2), (-0.5, sqrt3/2), (-1,0), (-0.5, -sqrt3/2), (0.5, -sqrt3/2).
    # Width = 2, Height = sqrt(3).
    # Horizontal neighbors: distance 1.5? No, 1 + 0.5 = 1.5?
    # If I put one at (0,0) and one at (1.5, 0), do they touch?
    # Right vertex of first: (1,0). Left vertex of second: (1.5-1, 0) = (0.5, 0).
    # Gap = 0.5. They don't touch.
    # To touch, the distance between centers must be 2 * apothem = sqrt(3) ≈ 1.732?
    # No, the distance between parallel edges is 2 * apothem = sqrt(3).
    # So if I shift by (sqrt(3), 0), the left edge of the second is at sqrt(3) - 1 = 0.732.
    # The right edge of the first is at 1.
    # They overlap.
    # Ah, the distance between centers of *touching* hexagons in a grid is:
    # If aligned, the distance is 2 * apothem = sqrt(3) only if they are touching edge-to-edge.
    # Yes, for flat-topped hexagons, the distance between centers of horizontally adjacent hexagons is 1.5 * 2? No.
    # Let's use the known lattice vectors for hexagonal packing of hexagons.
    # a1 = (1.5, sqrt(3)/2) * 2? No.
    # For unit hexagons (side 1):
    # Vector to right neighbor: (1.5, 0) ? No.
    # Let's check the vertices again.
    # Hex 1 at (0,0). Rightmost point (1,0).
    # Hex 2 at (x,0). Leftmost point (x-1, 0).
    # Touching means x-1 = 1 => x=2.
    # So distance is 2?
    # But they would overlap in y?
    # Hex 1 top y = sqrt(3)/2. Hex 2 top y = sqrt(3)/2.
    # They share the edge? No, they share the vertex?
    # If x=2, Hex 1 right vertex (1,0). Hex 2 left vertex (1,0).
    # They touch at a single point.
    # So the lattice vectors are (2, 0) and (1, sqrt(3)).
    
    # Let's generate 12 centers using this lattice.
    # 3x4 grid?
    # Or a hexagonal cluster of 12?
    # 1 (center) + 6 (ring 1) + 5 (partial ring 2)?
    # Let's just use a 3x4 grid of the lattice vectors.
    # v1 = (2, 0)
    # v2 = (1, sqrt(3))
    
    centers_init = []
    for i in range(3):
        for j in range(4):
            x = j * 2.0
            y = i * sqrt(3)
            if i % 2 == 1:
                x += 1.0
            centers_init.append([x, y])
    
    centers_init = np.array(centers_init)
    centers_init -= np.mean(centers_init, axis=0) # Center the cluster
    
    angles_init = np.zeros(12)
    
    params_init = np.concatenate([centers_init.flatten(), angles_init])
    
    # Optimize
    best_params = params_init
    best_val = objective(params_init)
    
    # Run optimization
    try:
        # Use L-BFGS-B
        # We might need multiple restarts or a robust method.
        # Since the landscape is complex, let's try a few iterations or just one good run.
        
        # To improve chances, let's do a few restarts with different initial angles or slight perturbations
        # if time permits.
        
        res = minimize(
            objective, 
            params_init, 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'maxfun': 20000, 'disp': False}
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
            
    except Exception as e:
        pass # Keep initial if optimization fails

    # Extract best results
    best_centers = best_params[0:24].reshape(12, 2)
    best_angles = best_params[24:36]
    best_s = calc_min_s(best_centers, best_angles)
    
    # Verify validity
    # Check if any overlap remains (should be minimal if penalty worked)
    # If there is significant overlap, we might need to adjust s or reject.
    # But the task requires valid packing.
    # If the optimizer didn't fully resolve overlaps, we might have an invalid result.
    # To be safe, we can increase s slightly if there's any residual overlap?
    # No, s is determined by the geometry.
    # If they overlap, the packing is invalid.
    
    # Let's do a final check and if invalid, fall back to the initial lattice (which is valid).
    def is_valid_packing(centers, angles, s):
        # Check containment (should be true by construction of s)
        apothem = s * sqrt3 / 2
        for i in range(N):
            verts = get_hex_vertices(centers[i,0], centers[i,1], 1.0, angles[i])
            projs = verts @ normals.T
            if np.any(np.max(projs, axis=0) > apothem + 1e-6):
                return False
        # Check non-overlap
        for i in range(N):
            for j in range(i+1, N):
                c1 = centers[i]
                a1 = angles[i]
                c2 = centers[j]
                a2 = angles[j]
                dist = np.linalg.norm(c1 - c2)
                if dist < 3.5:
                    pen = check_overlap(c1, a1, c2, a2)
                    if pen > 1e-8: # If penalty > 0, they overlap
                        return False
        return True

    if not is_valid_packing(best_centers, best_angles, best_s):
        # Fall back to initial lattice
        best_centers = centers_init
        best_angles = angles_init
        best_s = calc_min_s(best_centers, best_angles)
        
    # Construct return values
    inner_ret = np.zeros((12, 3))
    for i in range(12):
        inner_ret[i, 0] = best_centers[i, 0]
        inner_ret[i, 1] = best_centers[i, 1]
        inner_ret[i, 2] = best_angles[i] # angle in degrees
        
    outer_ret = np.array([0.0, 0.0, 0.0])
    
    return inner_ret, outer_ret, best_s

# EVOLVE-BLOCK-END
