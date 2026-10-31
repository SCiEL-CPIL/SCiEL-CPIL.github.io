"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import random


def construct():
    """
    Pack 21 circles in a rectangle of perimeter 4, maximizing sum of radii.
    
    Strategy: Discrete grid-based simulated annealing (2D Ising-like lattice gas)
    - Circles are placed on a high-resolution spatial grid
    - Each circle occupies one grid cell
    - Radii are dynamically calculated as min(distance to boundaries, distance to neighbors)
    - Simulated annealing swaps/moves circles between cells to escape local minima
    - Rectangle aspect ratio is optimized alongside circle positions
    """
    n = N
    
    # Grid resolution for discrete state space
    grid_res = 40  # 40x40 grid cells
    
    # Rectangle parameter: t in [-0.4, 0.4], w = 1+t, h = 1-t
    # We'll try multiple aspect ratios
    
    best_solution = None
    best_score = -float('inf')
    
    # Try several aspect ratios
    aspect_ratios = [0.0, 0.1, -0.1, 0.2, -0.2, 0.15, -0.15, 0.05, -0.05]
    
    for t in aspect_ratios:
        w = 1.0 + t
        h = 1.0 - t
        
        if w <= 0.05 or h <= 0.05:
            continue
        
        # Grid cell sizes
        cell_w = w / grid_res
        cell_h = h / grid_res
        
        # Initialize: place circles on a subset of grid cells
        # Use a hexagonal-like pattern for initial placement
        occupied = set()
        circles = []  # list of (grid_x, grid_y)
        
        # Initial placement: try to pack densely
        # Use a pattern that avoids corners
        for gy in range(2, grid_res - 2):
            for gx in range(2, grid_res - 2):
                if len(circles) >= n:
                    break
                # Hexagonal offset for odd rows
                if gy % 2 == 1 and gx % 2 == 1:
                    continue
                if (gx, gy) not in occupied:
                    circles.append((gx, gy))
                    occupied.add((gx, gy))
            if len(circles) >= n:
                break
        
        # If we didn't get enough, add more from remaining cells
        if len(circles) < n:
            for gy in range(grid_res):
                for gx in range(grid_res):
                    if len(circles) >= n:
                        break
                    if (gx, gy) not in occupied:
                        circles.append((gx, gy))
                        occupied.add((gx, gy))
                if len(circles) >= n:
                    break
        
        if len(circles) < n:
            continue
        
        # Function to compute radii and score for current configuration
        def compute_radii_and_score(circle_positions):
            """
            Given circle positions on grid, compute max possible radii and total score.
            Radius of each circle = min(dist to boundary, dist to each neighbor / 2)
            Actually: for circle i at (x_i, y_i), max r_i = min(
                x_i, w - x_i, y_i, h - y_i,
                min over j != i of dist(i,j) / 2
            )
            But we need all circles to have consistent radii. The sum of radii is maximized
            when each circle takes its max possible radius given the others.
            
            Actually, this is a coupled problem. Let's use a simpler approach:
            For a given set of positions, the max sum of radii is achieved when
            each circle's radius is limited by its nearest neighbor and boundaries.
            
            We'll compute: for each circle i, r_i_max = min(
                x_i, w - x_i, y_i, h - y_i,
                min_{j != i} dist(i,j) / 2
            )
            Then sum r_i_max. This is an upper bound but achievable if we set r_i = r_i_max.
            Wait, that's not quite right either because if r_i = dist(i,j)/2 and r_j = dist(i,j)/2,
            they just touch. But if r_i is also limited by boundary, it might be smaller.
            
            Let me think again. The constraint is: for all i,j, dist(i,j) >= r_i + r_j
            and r_i <= min(x_i, w-x_i, y_i, h-y_i)
            
            To maximize sum r_i, we can set r_i as large as possible. The binding constraints
            are the pairwise distances and boundaries.
            
            A simple approximation: r_i = 0.5 * min(
                2*min(x_i, w-x_i, y_i, h-y_i),
                min_{j != i} dist(i,j)
            )
            This ensures r_i + r_j <= dist(i,j) if both use the same formula? No, that's not guaranteed.
            
            Better approach: use a relaxation. Set r_i = alpha * min(
                min(x_i, w-x_i, y_i, h-y_i),
                min_{j != i} dist(i,j) / 2
            )
            for some alpha in [0, 1]. We want to find the max alpha such that all constraints hold.
            
            Actually, the cleanest way: for a fixed set of positions, the max sum of radii
            is a linear program. But for speed, let's use a heuristic:
            r_i = min(boundary_limit_i, min_{j != i} dist(i,j) / 2)
            Then check if this is feasible: for all i,j, r_i + r_j <= dist(i,j)?
            Not necessarily, because r_i might be limited by boundary while r_j is limited by
            distance to i.
            
            Let's use a simpler conservative estimate:
            r_i = 0.5 * min(
                2 * min(x_i, w-x_i, y_i, h-y_i),
                min_{j != i} dist(i,j)
            )
            This guarantees r_i + r_j <= dist(i,j) because:
            r_i <= dist(i,j)/2 and r_j <= dist(i,j)/2, so r_i + r_j <= dist(i,j). 
            And r_i <= boundary limit.
            
            Sum of these r_i is our score.
            """
            total_r = 0.0
            radii = []
            for i in range(len(circle_positions)):
                gx, gy = circle_positions[i]
                x = (gx + 0.5) * cell_w
                y = (gy + 0.5) * cell_h
                
                # Boundary limits
                bound_limit = min(x, w - x, y, h - y)
                
                # Nearest neighbor distance
                min_dist = float('inf')
                for j in range(len(circle_positions)):
                    if i == j:
                        continue
                    gx2, gy2 = circle_positions[j]
                    x2 = (gx2 + 0.5) * cell_w
                    y2 = (gy2 + 0.5) * cell_h
                    d = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if d < min_dist:
                        min_dist = d
                
                if min_dist == float('inf'):
                    r = bound_limit
                else:
                    r = 0.5 * min(2 * bound_limit, min_dist)
                
                radii.append(r)
                total_r += r
            
            return total_r, radii
        
        # Initial score
        current_score, _ = compute_radii_and_score(circles)
        
        # Simulated annealing
        T = 1.0
        T_min = 0.001
        cooling = 0.999
        max_iters = 20000
        rng = random.Random(42)
        
        best_local_score = current_score
        best_local_circles = circles[:]
        
        for it in range(max_iters):
            # Generate a move: swap two circles or move one circle
            move_type = rng.random()
            
            if move_type < 0.5:
                # Swap two circles
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if i == j:
                    continue
                new_circles = circles[:]
                new_circles[i], new_circles[j] = new_circles[j], new_circles[i]
            elif move_type < 0.8:
                # Move one circle to a random nearby cell
                i = rng.randint(0, n - 1)
                gx, gy = circles[i]
                # Try a random offset
                dx = rng.randint(-3, 3)
                dy = rng.randint(-3, 3)
                new_gx = gx + dx
                new_gy = gy + dy
                # Check bounds
                if new_gx < 0 or new_gx >= grid_res or new_gy < 0 or new_gy >= grid_res:
                    continue
                # Check if target is occupied
                if (new_gx, new_gy) in occupied and (new_gx, new_gy) != (gx, gy):
                    continue
                new_circles = circles[:]
                new_circles[i] = (new_gx, new_gy)
                # Update occupied set for scoring (we'll recompute anyway)
            else:
                # Move one circle to a random cell
                i = rng.randint(0, n - 1)
                new_gx = rng.randint(0, grid_res - 1)
                new_gy = rng.randint(0, grid_res - 1)
                if (new_gx, new_gy) in occupied and (new_gx, new_gy) != circles[i]:
                    continue
                new_circles = circles[:]
                new_circles[i] = (new_gx, new_gy)
            
            # Check for duplicate positions
            pos_set = set()
            valid_move = True
            for pos in new_circles:
                if pos in pos_set:
                    valid_move = False
                    break
                pos_set.add(pos)
            
            if not valid_move:
                continue
            
            # Compute new score
            new_score, _ = compute_radii_and_score(new_circles)
            delta = new_score - current_score
            
            # Accept or reject
            if delta > 0 or rng.random() < np.exp(delta / T):
                # Update occupied set
                old_pos = circles[0] if move_type < 0.5 else circles[0]
                # Actually, let's just rebuild occupied from new_circles
                circles = new_circles
                current_score = new_score
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_circles = circles[:]
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        # Track global best
        if best_local_score > best_score:
            best_score = best_local_score
            best_solution = (best_local_circles, w, h, cell_w, cell_h)
    
    # Convert best solution to output format
    if best_solution is None:
        # Fallback: simple grid in 1x1 square
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= n:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.005
                cells.append([x, y, r])
        return np.array(cells[:n])
    
    best_circles, w, h, cell_w, cell_h = best_solution
    
    # Compute radii for the best configuration
    _, radii = compute_radii_and_score(best_circles)
    
    # Build output
    result = []
    for i in range(n):
        gx, gy = best_circles[i]
        x = (gx + 0.5) * cell_w
        y = (gy + 0.5) * cell_h
        r = radii[i]
        # Ensure within bounds
        r = max(0.001, min(r, min(w, h) / 2 - 0.001))
        x = max(r, min(x, w - r))
        y = max(r, min(y, h - r))
        result.append([x, y, r])
    
    result = np.array(result)
    
    # Final validation
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((result[i, 0] - result[j, 0])**2 + (result[i, 1] - result[j, 1])**2)
            if dist < result[i, 2] + result[j, 2] - 1e-6:
                valid = False
        if result[i, 0] < result[i, 2] or result[i, 0] > w - result[i, 2]:
            valid = False
        if result[i, 1] < result[i, 2] or result[i, 1] > h - result[i, 2]:
            valid = False
    
    if not valid:
        # Scale down radii slightly
        scale = 0.95
        for i in range(n):
            result[i, 2] *= scale
            result[i, 0] = max(result[i, 2], min(result[i, 0], w - result[i, 2]))
            result[i, 1] = max(result[i, 2], min(result[i, 1], h - result[i, 2]))
    
    return result
# EVOLVE-BLOCK-END
