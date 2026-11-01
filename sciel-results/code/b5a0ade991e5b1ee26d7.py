import numpy as np

def construct():
    N = 32
    # Initial positions: Grid with slight jitter
    side = int(np.ceil(np.sqrt(N)))
    centers = []
    for i in range(side):
        for j in range(side):
            if len(centers) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                centers.append([x, y])
    centers = np.array(centers)
    
    # Initial radii: Small value to allow movement
    # Estimate max possible radius roughly
    # For N circles in unit square, avg area per circle ~ 1/N
    # Area of circle ~ pi r^2. r ~ sqrt(1/(N*pi)) ~ 0.1
    radii = np.full(N, 0.05)
    
    def get_energy(centers, radii):
        # Objective: Minimize -sum(radii) + penalty
        # Penalty for wall violation
        pen = 0.0
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            # Wall constraints
            d_left = x
            d_right = 1 - x
            d_bottom = y
            d_top = 1 - y
            min_wall = min(d_left, d_right, d_bottom, d_top)
            if r > min_wall:
                pen += (r - min_wall) ** 2
            
            # Neighbor constraints
            for j in range(i + 1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = dist
                if radii[i] + radii[j] > min_dist:
                    pen += (radii[i] + radii[j] - min_dist) ** 2
        
        return -np.sum(radii) + 100.0 * pen

    best_centers = centers.copy()
    best_radii = radii.copy()
    best_energy = get_energy(best_centers, best_radii)
    
    T = 0.05
    T_min = 0.0001
    cooling_rate = 0.999
    steps = 20000
    
    current_centers = centers.copy()
    current_radii = radii.copy()
    current_energy = get_energy(current_centers, current_radii)
    
    for step in range(steps):
        T *= cooling_rate
        if T < T_min:
            break
            
        # Randomly select a circle to move
        i = np.random.randint(N)
        
        # Decide whether to move position or radius
        if np.random.random() < 0.7:
            # Move position
            delta = np.random.normal(0, 0.05)
            new_x = current_centers[i][0] + delta[0]
            new_y = current_centers[i][1] + delta[1]
            
            # Keep within bounds for stability, but allow penalty to handle it
            # Actually, keeping them strictly inside helps convergence
            new_x = np.clip(new_x, 0.0, 1.0)
            new_y = np.clip(new_y, 0.0, 1.0)
            
            old_x, old_y = current_centers[i]
            current_centers[i][0] = new_x
            current_centers[i][1] = new_y
            
            new_energy = get_energy(current_centers, current_radii)
            
            if new_energy < current_energy or np.random.random() < np.exp((current_energy - new_energy) / T):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_centers = current_centers.copy()
                    best_radii = current_radii.copy()
            else:
                current_centers[i][0] = old_x
                current_centers[i][1] = old_y
        else:
            # Adjust radius
            # Try to increase radius
            delta_r = np.random.uniform(0, 0.05)
            old_r = current_radii[i]
            current_radii[i] = old_r + delta_r
            
            new_energy = get_energy(current_centers, current_radii)
            
            if new_energy < current_energy or np.random.random() < np.exp((current_energy - new_energy) / T):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_centers = current_centers.copy()
                    best_radii = current_radii.copy()
            else:
                current_radii[i] = old_r
        
        # Occasionally allow radius decrease to escape local minima? 
        # Usually increasing is the goal, but if stuck, decreasing might allow re-packing.
        # For now, stick to increasing radii as the primary move, 
        # but the penalty term discourages invalid states.
        
    # Final cleanup: Ensure radii are valid given the best centers
    # We can do a quick local relaxation on the best solution to maximize radii
    final_radii = np.zeros(N)
    for _ in range(50):
        for i in range(N):
            r_max = min(best_centers[i][0], 1-best_centers[i][0], best_centers[i][1], 1-best_centers[i][1])
            for j in range(N):
                if i == j: continue
                d = np.linalg.norm(best_centers[i] - best_centers[j])
                r_max = min(r_max, d - final_radii[j])
            final_radii[i] = max(0, r_max)
            
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [best_centers[i][0], best_centers[i][1], final_radii[i]]
        
    return result
