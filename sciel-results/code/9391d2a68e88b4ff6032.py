"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.optimize import linprog

def construct():
    """
    Pack 21 circles in a rectangle of perimeter 4, maximizing sum of radii.
    
    Strategy: Continuous Simulated Annealing
    - Circles are placed in a continuous 2D space.
    - Rectangle aspect ratio is fixed for each run, multiple runs with different ratios.
    - Objective: Sum of radii, where radius of circle i is approximated by 
      min(dist_to_boundary_i, min_j dist(i,j)/2).
    - Moves: Small random displacements of individual circles.
    """
    n = N
    best_global_score = -float('inf')
    best_global_result = None
    
    # Try a range of aspect ratios t where w = 1+t, h = 1-t
    # Perimeter = 2(w+h) = 2(1+t + 1-t) = 4. Correct.
    aspect_ratios = [0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4]
    
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    
    for t in aspect_ratios:
        w = 1.0 + t
        h = 1.0 - t
        if w < 0.1 or h < 0.1:
            continue
            
        # Initialization: Hexagonal packing
        # Estimate density: 21 circles in area w*h.
        # Area per circle ~ w*h/21. Radius ~ sqrt(w*h/21/pi).
        # Let's just place them in a grid-like hex pattern.
        
        # Determine grid size
        # Approx cols = sqrt(21 * w/h)
        cols = int(np.ceil(np.sqrt(n * w / h)))
        rows = int(np.ceil(n / cols))
        
        # Adjust spacing to fit
        # Spacing in x: w / cols
        # Spacing in y: h / rows
        # For hex packing, vertical spacing is spacing_x * sqrt(3)/2
        # We need to scale to fit in w x h.
        
        # Let's try a simple grid initialization first, then let SA fix it.
        # Or better, a hexagonal lattice scaled to fit.
        
        # Generate hexagonal lattice points
        points = []
        # Estimate side length of hexagon
        # Area of hexagon with side s is (3*sqrt(3)/2) * s^2
        # We want total area ~ w*h
        # 21 * (3*sqrt(3)/2) * s^2 ~ w*h  => s ~ sqrt(w*h / (21 * 2.598))
        s = np.sqrt(w * h / (n * 2.598))
        
        # Generate points in a hexagonal grid
        # dx = s, dy = s * sqrt(3)/2
        dx = s
        dy = s * np.sqrt(3) / 2
        
        # We need to generate enough points and pick n, or generate exactly n.
        # Let's generate a grid of points and center it.
        max_cols = int(np.ceil(w / dx)) + 2
        max_rows = int(np.ceil(h / dy)) + 2
        
        hex_points = []
        for r in range(max_rows):
            for c in range(max_cols):
                x = c * dx
                y = r * dy
                if r % 2 == 1:
                    x += dx / 2
                hex_points.append((x, y))
        
        if len(hex_points) < n:
             # Fallback to square grid
             cols_sq = int(np.ceil(np.sqrt(n)))
             rows_sq = int(np.ceil(n / cols_sq))
             dx_sq = w / cols_sq
             dy_sq = h / rows_sq
             for r in range(rows_sq):
                 for c in range(cols_sq):
                     if len(hex_points) >= n: break
                     x = (c + 0.5) * dx_sq
                     y = (r + 0.5) * dy_sq
                     hex_points.append((x, y))
                 if len(hex_points) >= n: break
        
        # Select n points closest to the center or just the first n?
        # Better: Select a subset that fits best.
        # For now, take the first n points, then shift them to be centered.
        # Actually, random initialization with SA is often more robust than a bad hex init.
        # Let's use random initialization within the box, slightly shrunk to avoid initial overlaps.
        
        # Shrink factor to ensure initial non-overlap? 
        # If we start with overlaps, SA has to push them apart.
        # Let's start with a random cloud.
        
        positions = np.random.default_rng(100 + int(t*1000)).uniform(0, 1, size=(n, 2))
        positions[:, 0] *= w
        positions[:, 1] *= h
        
        # SA Parameters
        T = 0.05
        T_min = 0.0001
        cooling = 0.9995
        max_iters = 50000
        step_size = 0.05
        
        current_score = -float('inf')
        # Calculate initial score
        def compute_score(pos, w, h):
            # pos is (n, 2)
            # r_i = min(x_i, w-x_i, y_i, h-y_i, min_j dist_ij/2)
            # Vectorized distance
            # dist matrix (n, n)
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, float('inf'))
            
            min_dists = np.min(dists, axis=1)
            boundary_limits = np.minimum(np.minimum(pos[:, 0], w - pos[:, 0]), np.minimum(pos[:, 1], h - pos[:, 1]))
            
            radii = np.minimum(boundary_limits, min_dists / 2.0)
            
            # Ensure positive radii
            radii = np.maximum(radii, 0.0)
            
            return np.sum(radii), radii
        
        current_score, _ = compute_score(positions, w, h)
        best_local_score = current_score
        best_local_pos = positions.copy()
        
        for it in range(max_iters):
            # Pick a random circle to move
            i = np_rng.integers(n)
            
            # Generate move: random displacement
            # Step size decays with temperature? Or fixed?
            # Let's use a step size proportional to T or fixed.
            # Fixed step is simpler.
            delta = np_rng.normal(0, step_size, size=2)
            
            new_pos = positions.copy()
            new_pos[i] += delta
            
            # Check bounds: If it goes out of bounds, the radius will be 0 or negative.
            # We should clamp or reject if it goes significantly out.
            # The score function handles out-of-bounds by setting radius to 0 (or negative, clamped to 0).
            # But we want circles to stay inside.
            # Let's clamp positions to [0, w] x [0, h] for the calculation, 
            # but this distorts the geometry.
            # Better: Reject move if it goes out of bounds?
            # Or allow it, score drops, SA rejects it.
            # Allowing it is fine, it acts as a penalty.
            
            new_score, _ = compute_score(new_pos, w, h)
            
            delta_score = new_score - current_score
            
            # Acceptance
            if delta_score > 0 or np_rng.random() < np.exp(delta_score / T):
                positions = new_pos
                current_score = new_score
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_pos = positions.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        if best_local_score > best_global_score:
            best_global_score = best_local_score
            # Store the positions and dimensions
            # We need to reconstruct the final radii for the best position
            _, final_radii = compute_score(best_local_pos, w, h)
            
            # Build result array
            result = np.zeros((n, 3))
            result[:, 0] = best_local_pos[:, 0]
            result[:, 1] = best_local_pos[:, 1]
            result[:, 2] = final_radii
            
            # Validation and correction
            # The heuristic radii might lead to overlaps if not careful?
            # r_i = min(bound, min_j d_ij/2).
            # r_i + r_j <= d_ij?
            # If r_i is limited by bound, and r_j is limited by d_ij/2, 
            # then r_i + r_j <= bound_i + d_ij/2. This is NOT necessarily <= d_ij.
            # Example: d_ij = 1. r_i = 0.4 (bound), r_j = 0.5 (d_ij/2). Sum = 0.9 <= 1. OK.
            # Example: d_ij = 1. r_i = 0.6 (bound? impossible if d_ij=1 and they are close).
            # Wait, if r_i is limited by boundary, it means the circle is close to the wall.
            # The constraint is r_i + r_j <= d_ij.
            # My heuristic: r_i = min(B_i, min_k d_ik/2).
            # So r_i <= d_ij/2.
            # And r_j <= d_ij/2.
            # So r_i + r_j <= d_ij.
            # This heuristic ALWAYS produces a feasible packing (no overlaps).
            # It might not be the MAXIMAL sum of radii, but it is valid.
            
            best_global_result = result
    
    if best_global_result is None:
        # Fallback
        result = np.zeros((n, 3))
        for i in range(n):
            result[i] = [0.5, 0.5, 0.01]
        return result
    
    # Final safety check: Ensure no overlaps and within bounds
    # The heuristic guarantees no overlaps, but let's double check due to floating point.
    res = best_global_result
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((res[i, 0] - res[j, 0])**2 + (res[i, 1] - res[j, 1])**2)
            sum_r = res[i, 2] + res[j, 2]
            if dist < sum_r - 1e-9:
                # Scale down radii slightly to fix
                scale = (dist / sum_r) * 0.999
                res[i, 2] *= scale
                res[j, 2] *= scale
    
    return res

# EVOLVE-BLOCK-END
