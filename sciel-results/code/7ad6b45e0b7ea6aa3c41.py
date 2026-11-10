"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np


def _score(centers):
    """Compute the packing radius for a set of centers."""
    # Distance to boundary
    boundary = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    return min(min_boundary, min_pair)


def _hex_init(n):
    """Create a hexagonal grid initialization."""
    # For 50 circles, use a hexagonal arrangement
    # Hex grid: rows alternate offset
    # Try different row counts
    best_centers = None
    best_score = -1
    
    for rows in range(5, 12):
        cols = int(np.ceil(n / rows))
        # Hex spacing
        dx = 1.0 / (cols + 0.5)  # approximate
        dy = dx * np.sqrt(3) / 2
        
        # Center the grid
        x_start = -0.5 + dx / 2
        y_start = -0.5 + dy / 2
        
        centers = []
        for i in range(rows):
            for j in range(cols):
                if len(centers) >= n:
                    break
                x = x_start + j * dx
                y = y_start + i * dy
                # Offset odd rows
                if i % 2 == 1:
                    x += dx / 2
                centers.append([x, y])
        
        if len(centers) >= n:
            centers = np.array(centers[:n])
            s = _score(centers)
            if s > best_score:
                best_score = s
                best_centers = centers
    
    if best_centers is None:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:n]
    
    return best_centers


def _simulated_annealing(centers, max_time=100.0):
    """Run simulated annealing to improve the packing."""
    start_time = time.time()
    deadline = start_time + max_time
    
    current = centers.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    # Temperature schedule
    T_start = 0.01
    T_end = 1e-8
    cooling_rate = 0.999
    
    T = T_start
    step_size = 0.005
    
    while T > T_end:
        if time.time() > deadline - 5:
            break
        
        # Generate a perturbation
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            
            idx = np.random.randint(len(current))
            dx = np.random.uniform(-step_size, step_size)
            dy = np.random.uniform(-step_size, step_size)
            
            new_center = current.copy()
            new_center[idx] += [dx, dy]
            
            # Keep within bounds
            new_center[idx] = np.clip(new_center[idx], -0.5, 0.5)
            
            new_score = _score(new_center)
            
            # Acceptance criterion
            if new_score >= current_score or np.random.random() < np.exp((new_score - current_score) / T):
                current = new_center
                current_score = new_score
                
                if current_score > best_score:
                    best = current.copy()
                    best_score = current_score
        
        T *= cooling_rate
        # Adapt step size
        if T < 0.001:
            step_size = 0.001
        elif T < 0.01:
            step_size = 0.003
    
    return best, best_score


def construct():
    start_time = time.time()
    total_time_limit = 110.0  # Leave margin for safety
    
    # Initialize with hexagonal grid
    centers = _hex_init(N)
    score = _score(centers)
    
    # Run simulated annealing with multiple restarts
    best_centers = centers
    best_score = score
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    if remaining > 10:
        centers, score = _simulated_annealing(centers, max_time=remaining - 5)
        if score > best_score:
            best_centers = centers
            best_score = score
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    if remaining > 10:
        # Try another restart from a slightly different initial position
        np.random.seed(42)
        centers2 = _hex_init(N) + np.random.uniform(-0.001, 0.001, (N, 2))
        centers2 = np.clip(centers2, -0.5, 0.5)
        score2 = _score(centers2)
        
        if score2 > best_score:
            best_centers = centers2
            best_score = score2
        
        centers2, score2 = _simulated_annealing(centers2, max_time=remaining - 5)
        if score2 > best_score:
            best_centers = centers2
            best_score = score2
    
    # Final local refinement
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    if remaining > 5:
        # Simple gradient-free local search
        centers = best_centers.copy()
        step = 0.0005
        for _ in range(100):
            if time.time() > start_time + total_time_limit - 2:
                break
            improved = False
            for i in range(N):
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        new_c = centers.copy()
                        new_c[i] += [dx, dy]
                        if np.any(np.abs(new_c[i]) > 0.5):
                            continue
                        s = _score(new_c)
                        if s > _score(centers) + 1e-12:
                            centers = new_c
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break
        
        final_score = _score(centers)
        if final_score > best_score:
            best_centers = centers
            best_score = final_score
    
    return best_centers
# EVOLVE-BLOCK-END
