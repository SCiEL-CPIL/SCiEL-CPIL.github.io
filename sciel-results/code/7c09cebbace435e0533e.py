import numpy as np
import time

N = 26

def construct():
    """
    Packs 26 circles in a unit square maximizing the sum of radii.
    Uses Simulated Annealing to escape local minima.
    """
    # Initial valid configuration: Grid layout
    side = int(np.ceil(np.sqrt(N)))
    # We might have an incomplete last row/col, so we adjust grid size
    # For N=26, side=6 (36 cells) or side=5 (25 cells). 
    # Let's use a 5x5 grid + 1 extra? No, 5x5 is 25. We need 26.
    # Let's use a 6x6 grid but only place 26? Or just random valid start.
    # A simple valid start: Place circles in a grid with small radius.
    # Let's try a dense grid initialization to start with a good sum.
    
    # Better initialization: 
    # Sort circles by some heuristic? 
    # Let's just use a random valid packing.
    # Or, start with a grid of 5x5 = 25 circles + 1 more.
    # Actually, let's just use random positions and small radii, then expand.
    
    def random_valid_start():
        # Place circles with small radius r=0.02
        # Check if valid, if not, retry or adjust.
        # For simplicity, let's place them on a grid with spacing 0.2
        # 5x5 grid = 25 circles. Add 26th at center?
        # Let's just use a grid of 26 points.
        # Arrange 26 points in 5 rows: 5,5,5,5,6?
        # Or just use np.random and ensure validity by shrinking radii.
        
        # Simplest robust start:
        # Grid 5x5 = 25 circles at (0.1, 0.3, 0.5, 0.7, 0.9)
        # 26th circle at (0.5, 0.5) with smaller radius?
        # Let's just generate random positions and set r=0.01, then grow.
        
        x = np.random.uniform(0.05, 0.95, N)
        y = np.random.uniform(0.05, 0.95, N)
        r = np.full(N, 0.01)
        return x, y, r

    x, y, r = random_valid_start()
    
    # Ensure initial validity (shrink radii if needed)
    def check_validity(x, y, r):
        # Boundary
        if np.any(x < r) or np.any(x > 1 - r) or np.any(y < r) or np.any(y > 1 - r):
            return False
        # Overlap
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        np.fill_diagonal(dist, np.inf)
        if np.any(dist < r_sum):
            return False
        return True

    # If not valid, shrink all radii until valid
    while not check_validity(x, y, r):
        r *= 0.9
        if np.max(r) < 1e-4:
            # Fallback to grid
            grid_x, grid_y = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))
            x = np.concatenate([grid_x.flatten(), [0.5]])
            y = np.concatenate([grid_y.flatten(), [0.5]])
            r = np.full(N, 0.05)
            # This might still overlap, so we loop again or just accept low score
            break

    best_sum = np.sum(r)
    best_state = (x.copy(), y.copy(), r.copy())

    # Simulated Annealing Parameters
    T = 1.0
    cooling_rate = 0.9995
    min_T = 1e-6
    max_iter = 20000
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer

    current_sum = np.sum(r)
    current_penalty = 0.0
    
    def calc_energy(x, y, r):
        # Energy = -Sum(R) + Penalty
        # Penalty is sum of squared violations
        pen = 0.0
        # Boundary
        pen += np.sum(np.maximum(0, r - x)**2)
        pen += np.sum(np.maximum(0, x - (1 - r))**2)
        pen += np.sum(np.maximum(0, r - y)**2)
        pen += np.sum(np.maximum(0, y - (1 - r))**2)
        
        # Overlap
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        np.fill_diagonal(dist, np.inf)
        violations = np.maximum(0, r_sum - dist)
        pen += np.sum(violations**2)
        
        return -np.sum(r) + 100.0 * pen

    current_energy = calc_energy(x, y, r)

    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Choose a random circle to perturb
        idx = np.random.randint(N)
        
        # Perturb type: position or radius
        if np.random.rand() < 0.5:
            # Perturb position
            dx = np.random.normal(0, 0.05)
            dy = np.random.normal(0, 0.05)
            new_x = x.copy()
            new_y = y.copy()
            new_r = r.copy()
            new_x[idx] += dx
            new_y[idx] += dy
            # Clamp to [0,1] roughly, penalty will handle it
            new_x[idx] = np.clip(new_x[idx], 0, 1)
            new_y[idx] = np.clip(new_y[idx], 0, 1)
        else:
            # Perturb radius
            dr = np.random.normal(0, 0.02)
            new_x = x.copy()
            new_y = y.copy()
            new_r = r.copy()
            new_r[idx] += dr
            # Clamp radius
            new_r[idx] = np.clip(new_r[idx], 0.001, 0.5)
            
        new_energy = calc_energy(new_x, new_y, new_r)
        
        delta_E = new_energy - current_energy
        
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            x, y, r = new_x, new_y, new_r
            current_energy = new_energy
            
            # Track best valid solution
            # Check if current state is valid
            # Optimization: Only check validity if energy is low (small penalty)
            # Or just check every few iterations
            if i % 10 == 0:
                if check_validity(x, y, r):
                    current_sum = np.sum(r)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_state = (x.copy(), y.copy(), r.copy())
        
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T

    # Final verification of best_state
    bx, by, br = best_state
    if not check_validity(bx, by, br):
        # Fallback if best state somehow invalid (shouldn't happen if check is strict)
        # Try to repair by shrinking
        while not check_validity(bx, by, br) and np.max(br) > 1e-4:
            br *= 0.95

    return np.column_stack((bx, by, br))
