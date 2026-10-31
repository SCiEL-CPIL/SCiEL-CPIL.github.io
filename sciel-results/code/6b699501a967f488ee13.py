"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import random


def construct():
    """
    Pack 21 circles in a rectangle of perimeter 4, maximizing sum of radii.
    
    Strategy: 2D Ising-like lattice gas with simulated annealing
    - Circles are placed on a high-resolution spatial grid
    - Each circle occupies one grid cell
    - Radii are dynamically calculated as min(distance to boundaries, distance to neighbors)
    - Simulated annealing swaps/moves circles between cells to escape local minima
    - Rectangle aspect ratio is optimized alongside circle positions
    """
    n = N
    
    # Grid resolution for discrete state space
    grid_res = 30
    
    # Rectangle parameter: t in [-0.3, 0.3], w = 1+t, h = 1-t
    best_solution = None
    best_score = -float('inf')
    
    # Try several aspect ratios
    aspect_ratios = [0.0, 0.1, -0.1, 0.2, -0.2, 0.15, -0.15, 0.05, -0.05, 0.25, -0.25]
    
    for t in aspect_ratios:
        w = 1.0 + t
        h = 1.0 - t
        
        if w <= 0.1 or h <= 0.1:
            continue
        
        # Grid cell sizes
        cell_w = w / grid_res
        cell_h = h / grid_res
        
        # Initialize: place circles on a subset of grid cells
        # Use a hexagonal-like pattern for initial placement
        occupied = set()
        circles = []  # list of (grid_x, grid_y)
        
        # Initial placement: try to pack densely
        for gy in range(1, grid_res - 1):
            for gx in range(1, grid_res - 1):
                if len(circles) >= n:
                    break
                # Hexagonal offset
                if gy % 2 == 1 and gx % 2 == 0:
                    continue
                if (gx, gy) not in occupied:
                    circles.append((gx, gy))
                    occupied.add((gx, gy))
            if len(circles) >= n:
                break
        
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
            total_r = 0.0
            radii = []
            for i in range(len(circle_positions)):
                gx, gy = circle_positions[i]
                x = (gx + 0.5) * cell_w
                y = (gy + 0.5) * cell_h
                
                bound_limit = min(x, w - x, y, h - y)
                
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
        T = 0.5
        T_min = 0.001
        cooling = 0.9995
        max_iters = 15000
        rng = random.Random(42)
        
        best_local_score = current_score
        best_local_circles = circles[:]
        
        for it in range(max_iters):
            move_type = rng.random()
            
            if move_type < 0.4:
                # Swap two circles
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if i == j:
                    continue
                new_circles = circles[:]
                new_circles[i], new_circles[j] = new_circles[j], new_circles[i]
            elif move_type < 0.75:
                # Move one circle to a random nearby cell
                i = rng.randint(0, n - 1)
                gx, gy = circles[i]
                dx = rng.randint(-2, 2)
                dy = rng.randint(-2, 2)
                new_gx = gx + dx
                new_gy = gy + dy
                if new_gx < 0 or new_gx >= grid_res or new_gy < 0 or new_gy >= grid_res:
                    continue
                if (new_gx, new_gy) in occupied and (new_gx, new_gy) != (gx, gy):
                    continue
                new_circles = circles[:]
                new_circles[i] = (new_gx, new_gy)
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
            
            new_score, _ = compute_radii_and_score(new_circles)
            delta = new_score - current_score
            
            if delta > 0 or rng.random() < np.exp(delta / T):
                circles = new_circles
                current_score = new_score
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_circles = circles[:]
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        if best_local_score > best_score:
            best_score = best_local_score
            best_solution = (best_local_circles, w, h, cell_w, cell_h)
    
    # Convert best solution to output format
    if best_solution is None:
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
    
    _, radii = compute_radii_and_score(best_circles)
    
    result = []
    for i in range(n):
        gx, gy = best_circles[i]
        x = (gx + 0.5) * cell_w
        y = (gy + 0.5) * cell_h
        r = radii[i]
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
        scale = 0.95
        for i in range(n):
            result[i, 2] *= scale
            result[i, 0] = max(result[i, 2], min(result[i, 0], w - result[i, 2]))
            result[i, 1] = max(result[i, 2], min(result[i, 1], h - result[i, 2]))
    
    return result
# EVOLVE-BLOCK-END
