# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Strategy:
    1. Use a symmetry-reduced parameterization (1 center, 6 ring, 4 outer) to reduce search space.
    2. Use a two-stage optimization: 
       - Stage 1: Coarse random restarts to find a valid, non-overlapping configuration.
       - Stage 2: Fine-tuning to minimize the outer hexagon side length.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    # Pre-compute constants
    SQRT3 = np.sqrt(3)
    APOTHEM = SQRT3 / 2.0
    # Normals for a hexagon with angle 0 (vertices at 0, 60, ...)
    # Edge 0: (1,0) to (0.5, 0.866). Normal angle 30 deg.
    normal_angles = np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles))) # (6, 2)
    
    # Vertices for a unit hexagon with angle 0
    vert_angles = np.radians(60 * np.arange(6))
    unit_verts = np.column_stack((np.cos(vert_angles), np.sin(vert_angles))) # (6, 2)

    def check_separation(centers):
        """
        Checks if all pairs of hexagons are separated.
        Returns True if valid, False otherwise.
        Also returns the minimum gap found (for optimization penalty).
        """
        min_gap = np.inf
        n_hex = len(centers)
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                c1 = centers[i]
                c2 = centers[j]
                diff = c1 - c2
                # Project difference onto all 6 normals
                # Since both are aligned, the 12 normals are just the 6 unique ones
                # Gap on normal k = 2*APOTHEM - |dot(diff, n_k)|
                projs = np.abs(normals @ diff)
                gaps = 2 * APOTHEM - projs
                max_gap = np.max(gaps)
                if max_gap < min_gap:
                    min_gap = max_gap
                # If max_gap <= 1e-6, they are intersecting or touching
                if max_gap <= 1e-6:
                    return False, min_gap
        return True, min_gap

    def check_containment(centers, S):
        """
        Checks if all inner hexagons are inside the outer hexagon of side S.
        Returns True if valid, False otherwise.
        Also returns the minimum distance to boundary (for optimization penalty).
        """
        min_dist = np.inf
        # Outer hexagon centered at (0,0), side S, angle 0.
        # Vertices at S * (cos(60i), sin(60i)).
        # Half-planes defined by normals at 30, 90, ...
        # Inequality: dot(p, n_k) <= S * APOTHEM
        limit = S * APOTHEM
        
        for c in centers:
            # Vertices of this inner hexagon
            verts = c + unit_verts
            # Project vertices onto normals
            projs = verts @ normals.T
            # For each vertex, check all 6 constraints
            # dist = limit - proj. We need dist >= 0 for all.
            # Min dist is min over all vertices and all normals of (limit - proj)
            min_dist_here = np.min(limit - projs)
            if min_dist_here < min_dist:
                min_dist = min_dist_here
            if min_dist_here < -1e-6:
                return False, min_dist
        return True, min_dist

    def objective(params):
        """
        Objective function for optimization.
        params: [r1, theta1, r2, theta2, S]
        r1: radius of inner ring (6 hexes)
        theta1: rotation of inner ring
        r2: radius of outer ring (4 hexes)
        theta2: rotation of outer ring
        S: side length of outer hexagon
        """
        if time.time() > deadline:
            return 1e9

        r1, theta1, r2, theta2, S = params
        if S < np.sqrt(11):
            return 1e9

        # Construct centers
        centers = np.zeros((11, 2))
        # 0: Center
        centers[0] = [0, 0]
        
        # 1-6: Inner ring
        # Angles: 0, 60, 120... + theta1
        angles1 = np.radians(60 * np.arange(6)) + np.radians(theta1)
        for k in range(6):
            centers[k+1] = r1 * np.array([np.cos(angles1[k]), np.sin(angles1[k])])
            
        # 7-10: Outer ring
        # Placed in gaps of the inner ring.
        # If inner ring is at 0, 60..., gaps are at 30, 90, 150, 210, 270, 330.
        # We pick 4. To maintain symmetry relative to the 6-ring, we can pick 
        # 30, 150, 270, 30? No.
        # Let's pick 30, 90, 150, 210? Or 30, 150, 270, 330?
        # Actually, the "flower" usually has the outer ones aligned with the inner ring's vertices or gaps.
        # If we rotate the outer ring by theta2, we can place them at:
        # 30 + theta2, 90 + theta2, 150 + theta2, 210 + theta2?
        # Or maybe 30 + theta2, 150 + theta2, 270 + theta2, 30 + theta2?
        # Let's assume the 4 outer ones are placed at angles:
        # 30 + theta2, 150 + theta2, 270 + theta2, 30 + theta2? No, that's 3 unique.
        # Let's try: 30 + theta2, 90 + theta2, 150 + theta2, 210 + theta2.
        # This breaks 6-fold symmetry but keeps some structure.
        # Alternatively, place them at 0, 120, 240, 300? No, 11 hexes.
        
        # Let's try a specific configuration that is known to be compact:
        # The 4 outer hexes are placed in 4 of the 6 gaps.
        # To maximize symmetry, we might place them at 30, 150, 270, 30? 
        # Actually, let's just parameterize the angles of the 4 outer hexes relative to the ring.
        # But that adds 4 variables.
        
        # Let's stick to: Outer ring angles are 30 + theta2, 90 + theta2, 150 + theta2, 210 + theta2.
        # This corresponds to picking 4 consecutive gaps.
        # Is that optimal? Maybe not.
        # Better: Let's allow the 4 outer hexes to be at angles:
        # 30 + theta2, 150 + theta2, 270 + theta2, 30 + theta2?
        # No, let's just use the 4 gaps: 30, 90, 150, 210 (relative to 0).
        # And rotate the whole set by theta2.
        
        angles2 = np.radians([30, 90, 150, 210]) + np.radians(theta2)
        for k in range(4):
            centers[k+7] = r2 * np.array([np.cos(angles2[k]), np.sin(angles2[k])])

        # Check separation
        sep_ok, sep_gap = check_separation(centers)
        if not sep_ok:
            # Penalty: large value if overlapping
            # We want to minimize S, but we MUST be separated.
            # If not separated, return high penalty.
            # To help optimizer, return S + penalty
            # Penalty should be proportional to violation.
            # Violation is 1e-6 - sep_gap (if sep_gap < 1e-6)
            penalty = 100.0 * (1e-6 - sep_gap)
            return S + penalty

        # Check containment
        cont_ok, cont_dist = check_containment(centers, S)
        if not cont_ok:
            # Penalty for containment violation
            penalty = 100.0 * (-cont_dist)
            return S + penalty

        # If valid, return S
        return S

    # Initial guesses
    # r1 ~ 1.8, r2 ~ 3.5, S ~ 6.0
    # theta1, theta2 ~ 0
    
    best_params = None
    best_S = np.inf
    best_centers = None
    
    # Stage 1: Random restarts to find valid configuration
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Randomize initial guess
        r1 = np.random.uniform(1.5, 2.2)
        theta1 = np.random.uniform(0, 60)
        r2 = np.random.uniform(3.0, 4.5)
        theta2 = np.random.uniform(0, 60)
        S_init = 7.0
        
        x0 = np.array([r1, theta1, r2, theta2, S_init])
        
        try:
            res = scipy.optimize.minimize(objective, x0, method='Nelder-Mead', 
                                          options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_S:
                # Check if it's actually valid (fun == S and no penalty)
                S_val = res.fun
                # Re-verify
                r1, theta1, r2, theta2, S_val = res.x
                centers = np.zeros((11, 2))
                centers[0] = [0, 0]
                angles1 = np.radians(60 * np.arange(6)) + np.radians(theta1)
                for k in range(6):
                    centers[k+1] = r1 * np.array([np.cos(angles1[k]), np.sin(angles1[k])])
                angles2 = np.radians([30, 90, 150, 210]) + np.radians(theta2)
                for k in range(4):
                    centers[k+7] = r2 * np.array([np.cos(angles2[k]), np.sin(angles2[k])])
                
                sep_ok, _ = check_separation(centers)
                cont_ok, _ = check_containment(centers, S_val)
                
                if sep_ok and cont_ok:
                    best_S = S_val
                    best_params = res.x
                    best_centers = centers
        except Exception:
            pass

    # If we found a valid configuration, try to refine S further
    if best_params is not None:
        # Fix positions, optimize S?
        # Actually, the optimizer above minimizes S subject to constraints.
        # So best_S should be close to optimal for this topology.
        
        # Let's do a final fine-tune with more iterations
        try:
            res_final = scipy.optimize.minimize(objective, best_params, method='Nelder-Mead',
                                                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res_final.fun < best_S:
                # Verify
                r1, theta1, r2, theta2, S_val = res_final.x
                centers = np.zeros((11, 2))
                centers[0] = [0, 0]
                angles1 = np.radians(60 * np.arange(6)) + np.radians(theta1)
                for k in range(6):
                    centers[k+1] = r1 * np.array([np.cos(angles1[k]), np.sin(angles1[k])])
                angles2 = np.radians([30, 90, 150, 210]) + np.radians(theta2)
                for k in range(4):
                    centers[k+7] = r2 * np.array([np.cos(angles2[k]), np.sin(angles2[k])])
                
                sep_ok, _ = check_separation(centers)
                cont_ok, _ = check_containment(centers, S_val)
                
                if sep_ok and cont_ok:
                    best_S = S_val
                    best_centers = centers
        except Exception:
            pass

    # Fallback to seed if optimization failed
    if best_centers is None:
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    # Format output
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = best_centers[i, 0]
        inner_hex_data[i, 1] = best_centers[i, 1]
        inner_hex_data[i, 2] = 0.0 # All aligned
        
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, best_S

# EVOLVE-BLOCK-END
