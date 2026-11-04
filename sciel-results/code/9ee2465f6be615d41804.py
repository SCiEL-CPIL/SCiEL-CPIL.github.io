"""
Circle packing in unit square using Radial Sweep strategy.
"""

N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import differential_evolution

def _eval_packing(centers, radii):
    """
    Evaluate the packing. Returns (sum_radii, is_valid).
    """
    n = len(radii)
    total_r = np.sum(radii)
    
    # Check boundaries
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x < r or x > 1.0 - r or y < r or y > 1.0 - r:
            return -np.inf, False
            
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[i] + radii[j]
            if dist_sq < min_dist * min_dist:
                return -np.inf, False
                
    return total_r, True

def _radial_sweep_objective(params, n_circles=26):
    """
    Objective function for the radial sweep representation.
    
    params structure:
    - params[0]: center_x (fixed at 0.5 usually, but let's optimize slightly)
    - params[1]: center_y
    - params[2]: center_radius
    - Remaining params: For each of the (n_circles - 1) outer circles:
        - ring_index (0 for inner ring, 1 for outer ring) -> but we need continuous optimization.
        - Let's define a fixed structure: 
          1 central circle.
          6 in the first ring (hexagonal-ish).
          12 in the second ring.
          7 in the third ring? 
          Total 1+6+12+7 = 26.
          
    Actually, a more flexible radial sweep allows varying the number per ring or just 
    optimizing angles and radii for a fixed set of "slots".
    
    Let's define slots:
    Slot 0: Center.
    Slots 1-6: Ring 1.
    Slots 7-18: Ring 2.
    Slots 19-25: Ring 3.
    
    Variables per outer slot i:
    - angle_i
    - radius_i (distance from center)
    - size_i (radius of the circle itself)
    
    However, to keep it simple and robust, let's fix the ring radii (distances from center) 
    to be optimized globally or fixed, and optimize angles and sizes.
    
    Better approach: 
    Variables:
    1. center_x, center_y, center_r
    2. For each of the 25 outer circles:
       - angle
       - dist_from_center
       - circle_radius
       
    This is 3 + 25*3 = 78 variables.
    
    Objective:
    - Sum of all radii.
    - Penalty for overlaps (pairwise).
    - Penalty for boundary violations.
    
    To make it a "Radial Sweep", we can constrain the outer circles to lie on specific 
    radial bands to reduce dimensionality? 
    The prompt suggests: "optimizing the angular positions and radial distances... relative to a fixed central anchor".
    
    Let's stick to the 78 variable DE optimization but structure the initialization 
    to be radial. The "Radial Sweep" here is the representation and initialization strategy.
    
    Actually, to be more specific to the "1D angular packing" idea mentioned in the directive 
    (though 78 vars is not 1D), we can try to fix the radial distances to a few discrete 
    bands and only optimize angles and sizes? 
    No, continuous optimization of dist is better for finding the optimum.
    
    Let's implement the 78-variable DE with a radial initialization.
    """
    cx, cy, cr = params[0], params[1], params[2]
    
    centers = np.empty((n_circles, 2))
    radii = np.empty(n_circles)
    
    centers[0] = [cx, cy]
    radii[0] = cr
    
    idx = 3
    for i in range(1, n_circles):
        angle = params[idx]
        dist = params[idx+1]
        r = params[idx+2]
        idx += 3
        
        x = cx + dist * np.cos(angle)
        y = cy + dist * np.sin(angle)
        
        centers[i] = [x, y]
        radii[i] = r
        
    # Calculate penalties and score
    score = np.sum(radii)
    
    # Boundary penalty
    for i in range(n_circles):
        x, y = centers[i]
        r = radii[i]
        if x < r: score -= (r - x)**2 * 100
        if x > 1.0 - r: score -= (x - (1.0 - r))**2 * 100
        if y < r: score -= (r - y)**2 * 100
        if y > 1.0 - r: score -= (y - (1.0 - r))**2 * 100
        
    # Overlap penalty
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                overlap = min_dist - dist
                score -= overlap**2 * 100
                
    return -score # Minimize negative score

def _radial_init(n_circles=26):
    """
    Initialize parameters in a radial layout.
    """
    # Center
    cx, cy, cr = 0.5, 0.5, 0.1
    
    params = [cx, cy, cr]
    
    # Define rings
    # Ring 1: 6 circles
    # Ring 2: 12 circles
    # Ring 3: 7 circles
    rings = [6, 12, 7]
    ring_dists = [0.2, 0.4, 0.6] # Initial guesses
    
    ring_idx = 0
    circle_idx = 1
    
    for num_in_ring, dist in zip(rings, ring_dists):
        # Add small noise to angles to avoid symmetry lock
        base_angle = 0
        for k in range(num_in_ring):
            angle = base_angle + 2 * np.pi * k / num_in_ring + random.uniform(-0.1, 0.1)
            # Initial radius guess: small
            r = 0.05 + random.uniform(0, 0.02)
            
            params.append(angle)
            params.append(dist + random.uniform(-0.02, 0.02))
            params.append(r)
            
            circle_idx += 1
            
    return params

def _bounds_for_radial(n_circles=26):
    """
    Define bounds for Differential Evolution.
    """
    bounds = []
    # Center x, y
    bounds.append((0.1, 0.9))
    bounds.append((0.1, 0.9))
    # Center radius
    bounds.append((0.01, 0.3))
    
    for i in range(1, n_circles):
        # Angle: 0 to 2*pi
        bounds.append((0, 2 * np.pi))
        # Dist from center: 0 to 0.8 (max distance to corner is ~0.707, but circle needs space)
        bounds.append((0.05, 0.7))
        # Radius: 0.01 to 0.3
        bounds.append((0.01, 0.3))
        
    return bounds

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s safety margin for overhead
    
    best_params = None
    best_score = -np.inf
    best_centers = None
    best_radii = None
    
    # Use Differential Evolution for the radial sweep parameters
    # DE is robust for global optimization and handles the non-convex landscape well.
    
    # We will run DE for a fixed amount of time, then polish.
    
    bounds = _bounds_for_radial(N)
    init_params = _radial_init(N)
    
    # Set random seed for reproducibility in this run? No, let it vary or use fixed for stability.
    # Using a fixed seed helps with debugging, but random is better for exploration.
    # Let's use a few restarts if time permits.
    
    # DE requires maxiter and popsize. We'll estimate based on time.
    # 78 variables. Popsize 30. 
    # Each function evaluation is O(N^2) ~ 26^2 = 676 ops. Very fast.
    # We can afford many iterations.
    
    # Let's run DE for 10 seconds first.
    t_start = time.time()
    
    try:
        # DE with time budget
        # We can't directly pass time limit to DE, so we estimate maxiter.
        # Let's do 200 iterations with popsize 20.
        result = differential_evolution(
            _radial_sweep_objective, 
            bounds, 
            seed=42, 
            maxiter=300, 
            popsize=20,
            tol=1e-4,
            polish=False # We will do our own polish
        )
        
        if result.success or result.fun < best_score:
            best_params = result.x
            best_score = -result.fun
            
            # Reconstruct centers and radii
            cx, cy, cr = best_params[0], best_params[1], best_params[2]
            centers = np.empty((N, 2))
            radii = np.empty(N)
            centers[0] = [cx, cy]
            radii[0] = cr
            idx = 3
            for i in range(1, N):
                angle = best_params[idx]
                dist = best_params[idx+1]
                r = best_params[idx+2]
                idx += 3
                x = cx + dist * np.cos(angle)
                y = cy + dist * np.sin(angle)
                centers[i] = [x, y]
                radii[i] = r
                
            # Check validity and score
            s, valid = _eval_packing(centers, radii)
            if valid:
                best_score = s
                best_centers = centers
                best_radii = radii
            else:
                # If invalid, we need to fix it.
                # Apply a simple repair: shrink circles that violate boundaries or overlap.
                # For now, store it and try to repair in a local step.
                best_centers = centers
                best_radii = radii
                best_score = s # Keep the score even if invalid? No, we want valid.
                # Let's force a valid state by shrinking.
                _repair_packing(centers, radii)
                s, valid = _eval_packing(centers, radii)
                if valid:
                    best_score = s
                    best_centers = centers
                    best_radii = radii

    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to grid if DE fails completely
        best_centers, best_radii = _grid_init(N)
        best_score, _ = _eval_packing(best_centers, best_radii)

    # Local Polish: Gradient-based or Simple Local Search to maximize sum of radii
    # Since DE might leave overlaps or suboptimal sizes, we do a local search.
    
    if time.time() < deadline - 10:
        best_centers, best_radii = _local_polish_radial(best_centers, best_radii, N, time_limit=20.0)
        s, valid = _eval_packing(best_centers, best_radii)
        if valid:
            best_score = s
        else:
            # If polish made it invalid, revert? Or repair.
            _repair_packing(best_centers, best_radii)
            s, valid = _eval_packing(best_centers, best_radii)
            if valid:
                best_score = s
            else:
                # Last resort: use the pre-polish result if it was valid
                # We didn't store it explicitly, so let's just ensure validity now.
                pass

    # Final Validation and Cleanup
    if best_centers is None:
        best_centers, best_radii = _grid_init(N)
        
    result = np.column_stack([best_centers, best_radii])
    
    # Ensure strict validity: clip radii to fit boundaries
    for i in range(N):
        x, y, ri = result[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if ri > max_r:
            ri = max_r
        result[i, 2] = ri
        # Adjust position if it was pushed out by clipping? 
        # Clipping radius doesn't move center. If center is outside new bounds, move center.
        if x < ri: result[i, 0] = ri
        if x > 1.0 - ri: result[i, 0] = 1.0 - ri
        if y < ri: result[i, 1] = ri
        if y > 1.0 - ri: result[i, 1] = 1.0 - ri
        
    # Check for overlaps one last time and shrink if necessary
    for _ in range(20):
        modified = False
        for i in range(N):
            for j in range(i+1, N):
                dx = result[i, 0] - result[j, 0]
                dy = result[i, 1] - result[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = result[i, 2] + result[j, 2]
                if dist < min_dist:
                    overlap = min_dist - dist
                    if result[i, 2] < result[j, 2]:
                        result[i, 2] = max(0.001, result[i, 2] - overlap/2)
                    else:
                        result[j, 2] = max(0.001, result[j, 2] - overlap/2)
                    modified = True
        if not modified:
            break
            
    return result

def _repair_packing(centers, radii):
    """
    Simple repair to make packing valid by shrinking radii.
    """
    n = len(radii)
    for _ in range(50):
        modified = False
        # Boundary check
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r:
                radii[i] = max_r * 0.99
                modified = True
        # Overlap check
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    # Shrink both slightly
                    shrink = overlap / 2.0 + 0.001
                    radii[i] = max(0.001, radii[i] - shrink)
                    radii[j] = max(0.001, radii[j] - shrink)
                    modified = True
        if not modified:
            break

def _local_polish_radial(centers, radii, n, time_limit=20.0):
    """
    Local search to maximize sum of radii.
    Tries to increase radius of each circle, moving it if necessary.
    """
    current_score, valid = _eval_packing(centers, radii)
    if not valid:
        _repair_packing(centers, radii)
        current_score, valid = _eval_packing(centers, radii)
        
    start_time = time.time()
    
    for _ in range(10000):
        if time.time() - start_time > time_limit:
            break
            
        i = np.random.randint(n)
        # Try to increase radius
        delta_r = 0.005
        new_r = radii[i] + delta_r
        
        # Check boundary
        x, y = centers[i]
        if x < new_r or x > 1.0 - new_r or y < new_r or y > 1.0 - new_r:
            # Try to move center to allow larger radius
            # Move towards center of available space?
            # Simple: try random small moves
            moved = False
            for _ in range(10):
                dx = np.random.normal(0, 0.02)
                dy = np.random.normal(0, 0.02)
                nx = x + dx
                ny = y + dy
                if nx < new_r or nx > 1.0 - new_r or ny < new_r or ny > 1.0 - new_r:
                    continue
                
                # Check overlaps with new position and new radius
                valid_move = True
                for j in range(n):
                    if i == j: continue
                    ddx = nx - centers[j][0]
                    ddy = ny - centers[j][1]
                    dist = np.sqrt(ddx*ddx + ddy*ddy)
                    if dist < new_r + radii[j]:
                        valid_move = False
                        break
                
                if valid_move:
                    centers[i] = [nx, ny]
                    radii[i] = new_r
                    moved = True
                    break
            
            if not moved:
                continue
        else:
            # Check overlaps at current position with new radius
            valid_move = True
            for j in range(n):
                if i == j: continue
                ddx = x - centers[j][0]
                ddy = y - centers[j][1]
                dist = np.sqrt(ddx*ddx + ddy*ddy)
                if dist < new_r + radii[j]:
                    valid_move = False
                    break
            
            if valid_move:
                radii[i] = new_r
            else:
                # Try to move
                moved = False
                for _ in range(10):
                    dx = np.random.normal(0, 0.02)
                    dy = np.random.normal(0, 0.02)
                    nx = x + dx
                    ny = y + dy
                    if nx < new_r or nx > 1.0 - new_r or ny < new_r or ny > 1.0 - new_r:
                        continue
                    
                    valid_move = True
                    for j in range(n):
                        if i == j: continue
                        ddx = nx - centers[j][0]
                        ddy = ny - centers[j][1]
                        dist = np.sqrt(ddx*ddx + ddy*ddy)
                        if dist < new_r + radii[j]:
                            valid_move = False
                            break
                    
                    if valid_move:
                        centers[i] = [nx, ny]
                        radii[i] = new_r
                        moved = True
                        break

    return centers, radii

def _grid_init(n):
    """
    Initialize with a grid layout.
    """
    side = int(np.ceil(np.sqrt(n)))
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        gx = i % side
        gy = i // side
        centers[i][0] = (gx + 0.5) / side
        centers[i][1] = (gy + 0.5) / side
        radii[i] = 0.5 / side - 0.01
    return centers, radii
# EVOLVE-BLOCK-END
