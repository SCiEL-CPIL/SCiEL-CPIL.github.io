# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _is_valid(centers, sides, tol=1e-7):
    """Check if the packing is valid."""
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    
    # Check inside unit square
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
            
    # Check no overlaps
    for i in range(n):
        for j in range(i+1, n):
            # Overlap if |cx_i - cx_j| < (s_i + s_j)/2 AND |cy_i - cy_j| < (s_i + s_j)/2
            # Valid if NOT overlapping, i.e., |cx_i - cx_j| >= (s_i + s_j)/2 - tol OR |cy_i - cy_j| >= (s_i + s_j)/2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            min_dx = (sides[i] + sides[j])/2 - tol
            min_dy = (sides[i] + sides[j])/2 - tol
            if dx < min_dx and dy < min_dy:
                return False
    return True

def _objective(centers, sides):
    """Objective: sum of side lengths."""
    return np.sum(sides)

def _local_search(centers, sides, rng, max_iters=10000, tol=1e-7):
    """
    Perform local search with simulated annealing.
    Returns best valid configuration found.
    """
    n = len(sides)
    current_centers = centers.copy()
    current_sides = sides.copy()
    current_val = _objective(current_centers, current_sides)
    
    best_centers = current_centers.copy()
    best_sides = current_sides.copy()
    best_val = current_val
    
    if not _is_valid(current_centers, current_sides, tol):
        return best_centers, best_sides, best_val

    T = 0.01 # Initial temperature
    cooling_rate = 0.999
    min_T = 1e-6
    
    for it in range(max_iters):
        if T < min_T:
            break
            
        # Generate a neighbor
        neighbor_centers = current_centers.copy()
        neighbor_sides = current_sides.copy()
        
        # Pick a random square to perturb
        idx = rng.integers(0, n)
        
        # Perturb center
        noise = 0.01
        neighbor_centers[idx, 0] += rng.uniform(-noise, noise)
        neighbor_centers[idx, 1] += rng.uniform(-noise, noise)
        
        # Perturb side
        side_noise = 0.01
        neighbor_sides[idx] += rng.uniform(-side_noise, side_noise)
        
        # Ensure side is positive and within bounds roughly
        neighbor_sides[idx] = max(0.001, min(1.0, neighbor_sides[idx]))
        
        # Ensure center allows square to be inside [0,1]
        s = neighbor_sides[idx]
        min_cx = s/2
        max_cx = 1 - s/2
        min_cy = s/2
        max_cy = 1 - s/2
        
        neighbor_centers[idx, 0] = np.clip(neighbor_centers[idx, 0], min_cx, max_cx)
        neighbor_centers[idx, 1] = np.clip(neighbor_centers[idx, 1], min_cy, max_cy)
        
        # Check validity
        if not _is_valid(neighbor_centers, neighbor_sides, tol):
            # Reject invalid moves
            T *= cooling_rate
            continue
            
        neighbor_val = _objective(neighbor_centers, neighbor_sides)
        
        # Simulated annealing acceptance
        delta = neighbor_val - current_val
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_centers = neighbor_centers
            current_sides = neighbor_sides
            current_val = neighbor_val
            
            if current_val > best_val:
                best_val = current_val
                best_centers = current_centers.copy()
                best_sides = current_sides.copy()
                
        T *= cooling_rate
        
    return best_centers, best_sides, best_val

def _generate_initial_config(n=10, method='grid'):
    """Generate an initial configuration."""
    if method == 'grid':
        # 4x3 grid, but we have 10 squares.
        # Use 4 columns, 3 rows.
        # Fill 10 squares.
        # To maximize sum, maybe make them all 0.25?
        # Or vary sizes?
        # Let's start with uniform 0.25.
        centers = np.zeros((n, 2))
        sides = np.full(n, 0.25)
        
        # Place them in a grid pattern
        # 4 cols, 3 rows
        # Indices 0-3 in row 0, 4-7 in row 1, 8-9 in row 2
        for i in range(n):
            col = i % 4
            row = i // 4
            cx = (col + 0.5) * 0.25
            cy = (row + 0.5) * 0.25
            centers[i, 0] = cx
            centers[i, 1] = cy
        return centers, sides
        
    elif method == 'heuristic':
        # Try a known good packing structure
        # e.g. 1 large square (0.5) and 9 small ones (0.25)?
        # Large at (0.25, 0.25).
        # Remaining space: 
        #   x in [0.5, 1], y in [0, 1] (width 0.5, height 1) -> 4 squares of 0.25
        #   x in [0, 0.5], y in [0.5, 1] (width 0.5, height 0.5) -> 4 squares of 0.25? No, 0.5x0.5 fits 4 of 0.25.
        # Total: 1 large (0.5) + 8 small (0.25) = 9 squares.
        # We need 10.
        # Maybe 2 large (0.5) and 8 small (0.25)?
        # 2 large take 0.5x0.5 each. Total area 0.5.
        # 8 small take 0.25x0.25 each. Total area 8 * 0.0625 = 0.5.
        # Total area 1.0. Perfect tiling?
        # Layout:
        #   Large 1 at (0.25, 0.25)
        #   Large 2 at (0.75, 0.25)
        #   Top half y in [0.5, 1]:
        #     x in [0, 0.5]: 4 small squares of 0.25
        #     x in [0.5, 1]: 4 small squares of 0.25
        # This is 2 + 8 = 10 squares.
        # Sum of sides: 2*0.5 + 8*0.25 = 1.0 + 2.0 = 3.0.
        # This is better than 2.5!
        
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        # 2 Large
        centers[0] = [0.25, 0.25]
        sides[0] = 0.5
        centers[1] = [0.75, 0.25]
        sides[1] = 0.5
        
        # 8 Small
        # Top-left quadrant (x:0-0.5, y:0.5-1.0)
        # 4 squares of 0.25
        for k in range(4):
            col = k % 2
            row = k // 2
            cx = (col + 0.5) * 0.25
            cy = 0.5 + (row + 0.5) * 0.25
            centers[2+k, 0] = cx
            centers[2+k, 1] = cy
            sides[2+k] = 0.25
            
        # Top-right quadrant (x:0.5-1.0, y:0.5-1.0)
        # 4 squares of 0.25
        for k in range(4):
            col = k % 2
            row = k // 2
            cx = 0.5 + (col + 0.5) * 0.25
            cy = 0.5 + (row + 0.5) * 0.25
            centers[6+k, 0] = cx
            centers[6+k, 1] = cy
            sides[6+k] = 0.25
            
        return centers, sides

    else:
        return _generate_initial_config(n, 'grid')

def squares_in_square10() -> tuple:
    """
    Returns:
        centers: np.ndarray of shape (10, 2)
        sides:   np.ndarray of shape (10,)
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = 600 # seconds
    
    best_centers = None
    best_sides = None
    best_val = -1.0
    
    rng = np.random.default_rng(42)
    
    # Try different initial configurations and local search
    # 1. Heuristic (2 large, 8 small)
    c_init, s_init = _generate_initial_config(n, 'heuristic')
    if _is_valid(c_init, s_init, tol):
        bc, bs, bv = _local_search(c_init, s_init, rng, max_iters=50000, tol=tol)
        if bv > best_val:
            best_val = bv
            best_centers = bc
            best_sides = bs
            
    # Check time
    if time.time() - start_time > deadline - 60:
        if best_centers is not None:
            return best_centers, best_sides
        # Fallback to grid
        c_grid, s_grid = _generate_initial_config(n, 'grid')
        return c_grid, s_grid

    # 2. Grid baseline
    c_init, s_init = _generate_initial_config(n, 'grid')
    bc, bs, bv = _local_search(c_init, s_init, rng, max_iters=50000, tol=tol)
    if bv > best_val:
        best_val = bv
        best_centers = bc
        best_sides = bs
        
    # Check time
    if time.time() - start_time > deadline - 60:
        if best_centers is not None:
            return best_centers, best_sides
        c_grid, s_grid = _generate_initial_config(n, 'grid')
        return c_grid, s_grid

    # 3. Random restarts with local search
    # Try to find even better packings
    max_restarts = 20
    for r in range(max_restarts):
        if time.time() - start_time > deadline - 30:
            break
            
        # Random initial config
        c_rand = np.random.uniform(0.1, 0.9, (n, 2))
        s_rand = np.random.uniform(0.1, 0.4, n)
        
        # Quick validity check and repair
        # Just run local search, it will handle invalid starts by rejecting moves
        # But we need a valid start to start the annealing loop properly?
        # My _local_search rejects invalid neighbors, so if start is invalid, it does nothing.
        # So we need a valid start.
        
        # Let's try to construct a valid random start by placing squares greedily
        c_valid = np.zeros((n, 2))
        s_valid = np.zeros(n)
        
        # Place 1 large
        s_valid[0] = 0.5
        c_valid[0] = [0.25, 0.25]
        
        # Place 1 large
        s_valid[1] = 0.5
        c_valid[1] = [0.75, 0.25]
        
        # Place 8 small
        for k in range(2, n):
            s_valid[k] = 0.25
            # Try to place in top half
            # Randomly pick from top-left or top-right quadrant
            if k < 6:
                # Top-left
                col = (k-2) % 2
                row = (k-2) // 2
                c_valid[k, 0] = (col + 0.5) * 0.25
                c_valid[k, 1] = 0.5 + (row + 0.5) * 0.25
            else:
                # Top-right
                col = (k-6) % 2
                row = (k-6) // 2
                c_valid[k, 0] = 0.5 + (col + 0.5) * 0.25
                c_valid[k, 1] = 0.5 + (row + 0.5) * 0.25
                
        # Add some noise
        for i in range(n):
            c_valid[i, 0] += rng.uniform(-0.05, 0.05)
            c_valid[i, 1] += rng.uniform(-0.05, 0.05)
            s_valid[i] += rng.uniform(-0.05, 0.05)
            s_valid[i] = max(0.01, s_valid[i])
            
        # Clip to bounds
        for i in range(n):
            s = s_valid[i]
            c_valid[i, 0] = np.clip(c_valid[i, 0], s/2, 1-s/2)
            c_valid[i, 1] = np.clip(c_valid[i, 1], s/2, 1-s/2)
            
        if _is_valid(c_valid, s_valid, tol):
            bc, bs, bv = _local_search(c_valid, s_valid, rng, max_iters=20000, tol=tol)
            if bv > best_val:
                best_val = bv
                best_centers = bc
                best_sides = bs

    if best_centers is None:
        # Fallback
        c_grid, s_grid = _generate_initial_config(n, 'grid')
        return c_grid, s_grid
        
    return best_centers, best_sides

# EVOLVE-BLOCK-END
