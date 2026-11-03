import numpy as np
import time

N = 50

def _score(centers):
    """Compute the radius for a given configuration."""
    # Boundary constraint
    boundary = 0.5 - np.abs(centers).max(axis=0)
    # Actually need min over all points and all coordinates
    boundary = (0.5 - np.abs(centers)).min()
    
    # Pairwise distances
    # For 50 points, 50*49/2 = 1225 pairs
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    
    r = min(boundary, min_dist / 2.0)
    return r

def _hexagonal_init(n_rows, n_cols, offset=0.0):
    """Create a hexagonal-like initial configuration."""
    centers = []
    for i in range(n_rows):
        for j in range(n_cols):
            x = (j + 0.5 + (0.5 if i % 2 == 1 else 0.0)) / n_cols - 0.5
            y = (i + 0.5) / n_rows - 0.5
            centers.append([x, y])
    centers = np.array(centers)
    if len(centers) > N:
        centers = centers[:N]
    elif len(centers) < N:
        # Pad with random points
        pad = np.random.randn(N - len(centers), 2) * 0.01
        pad[:, 0] += centers[-1, 0]
        pad[:, 1] += centers[-1, 1]
        centers = np.vstack([centers, pad])
    return centers

def _random_init():
    """Random initial configuration inside the square."""
    return (np.random.rand(N, 2) - 0.5) * 0.9

def _optimize(centers, time_budget=100.0):
    """Simulated annealing optimization."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    current_centers = centers.copy()
    current_score = _score(current_centers)
    
    T = 0.01  # Initial temperature
    T_min = 1e-8
    alpha = 0.999  # Cooling rate
    
    step_size = 0.01
    
    while time.time() - start_time < time_budget:
        # Generate a move: pick a random point and nudge it
        idx = np.random.randint(N)
        move = np.random.randn(2) * step_size
        
        new_centers = current_centers.copy()
        new_centers[idx] += move
        
        # Check if still in bounds (with margin)
        if np.any(np.abs(new_centers) > 0.5):
            new_centers[idx] = current_centers[idx]
            continue
        
        new_score = _score(new_centers)
        
        if new_score >= current_score or np.random.random() < np.exp((new_score - current_score) / max(T, 1e-10)):
            current_centers = new_centers
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_centers = current_centers.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min
        
        # Occasionally reheat
        if np.random.random() < 0.01:
            T = max(T, 0.005)
        
        # Adapt step size
        if current_score > best_score:
            step_size = max(step_size * 0.9, 0.001)
        elif np.random.random() < 0.1:
            step_size = min(step_size * 1.1, 0.05)
    
    return best_centers, best_score

def _local_refine(centers, time_budget=30.0):
    """Local refinement using gradient-like moves."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _score(best_centers)
    current_centers = centers.copy()
    current_score = _score(current_centers)
    
    step_size = 0.005
    
    while time.time() - start_time < time_budget:
        improved = False
        for idx in range(N):
            # Try small moves in 8 directions
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_centers = current_centers.copy()
                    new_centers[idx, 0] += dx
                    new_centers[idx, 1] += dy
                    
                    if np.any(np.abs(new_centers) > 0.5):
                        continue
                    
                    new_score = _score(new_centers)
                    if new_score > current_score:
                        current_centers = new_centers
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
        
        if current_score > best_score:
            best_score = current_score
            best_centers = current_centers.copy()
        
        if not improved:
            step_size *= 0.7
            if step_size < 1e-6:
                break
        
        # Check time
        if time.time() - start_time > time_budget - 5:
            break
    
    return best_centers, best_score

def construct():
    """Main entry point: find optimal packing of 50 circles in unit square."""
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit
    
    np.random.seed(42)
    
    best_centers = None
    best_score = -1
    
    # Generate multiple initial configurations
    inits = []
    
    # Hexagonal-like grids with different parameters
    for rows in range(5, 9):
        for cols in range(5, 9):
            if rows * cols >= N:
                c = _hexagonal_init(rows, cols)
                inits.append(c)
    
    # Random inits
    for _ in range(5):
        inits.append(_random_init())
    
    # Process each init with time budget
    n_inits = len(inits)
    time_per_init = (DEADLINE - (time.time() - start_time)) / max(n_inits, 1)
    time_per_init = min(time_per_init, 15.0)  # Cap per init
    
    for i, init in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 5:
            break
        
        # Annealing phase
        t_anneal = min(time_per_init * 0.6, remaining * 0.5)
        if t_anneal > 1:
            centers_anneal, score_anneal = _optimize(init, time_budget=t_anneal)
            if score_anneal > best_score:
                best_score = score_anneal
                best_centers = centers_anneal.copy()
        
        # Local refinement phase
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 5 and best_centers is not None:
            t_refine = min(time_per_init * 0.4, remaining * 0.4)
            if t_refine > 1:
                centers_refine, score_refine = _local_refine(centers_anneal, time_budget=t_refine)
                if score_refine > best_score:
                    best_score = score_refine
                    best_centers = centers_refine.copy()
    
    # Final refinement on best result
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 10 and best_centers is not None:
        centers_final, score_final = _local_refine(best_centers, time_budget=min(remaining * 0.8, 30.0))
        if score_final > best_score:
            best_score = score_final
            best_centers = centers_final.copy()
    
    # Ensure validity: clip to square
    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    else:
        best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
