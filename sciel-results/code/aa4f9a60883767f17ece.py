import numpy as np
import time
from scipy.spatial.distance import cdist

N = 26

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    best_score = -np.inf
    best_solution = None
    
    def max_radius_for_circle(x, y, other_xs, other_ys, other_rs):
        """Calculate the maximum radius a circle at (x,y) can have without overlapping others or boundaries."""
        # Boundary constraints
        r_bound = min(x, 1.0 - x, y, 1.0 - y)
        
        if len(other_xs) == 0:
            return max(0.0, r_bound)
        
        # Distance to other circles
        dx = other_xs - x
        dy = other_ys - y
        dists = np.sqrt(dx**2 + dy**2)
        
        # Radius limited by distance to other circles minus their radii
        r_others = dists - other_rs
        r_max = np.min(r_others)
        
        return max(0.0, min(r_bound, r_max))

    def eval_packing(z):
        """Evaluate the sum of radii for a valid packing. Returns -inf if invalid."""
        x, y, r = z[:, 0], z[:, 1], z[:, 2]
        
        # Check boundaries
        if np.any(r > x) or np.any(r > 1.0 - x) or np.any(r > y) or np.any(r > 1.0 - y):
            return -np.inf
            
        # Check overlaps
        dist = cdist(np.column_stack((x, y)), np.column_stack((x, y)))
        iu = np.triu_indices(N, k=1)
        d_ij = dist[iu]
        r_i, r_j = r[iu[0]], r[iu[1]]
        
        if np.any(d_ij < r_i + r_j - 1e-9):
            return -np.inf
            
        return np.sum(r)

    def best_fit_local_search(z_init, max_iters=2000):
        """
        Discrete best-fit local search.
        1. Start with a valid packing (radii set to max possible for initial positions).
        2. Iteratively try to move one circle slightly and update its radius to the max possible.
        3. Accept if the sum of radii increases.
        """
        z = z_init.copy()
        
        # Initial pass: set radii to max possible for the given centers
        for i in range(N):
            other_idx = [j for j in range(N) if j != i]
            if other_idx:
                z[i, 2] = max_radius_for_circle(z[i, 0], z[i, 1], z[other_idx, 0], z[other_idx, 1], z[other_idx, 2])
            else:
                z[i, 2] = 0.5
        # Ensure validity (might need slight reduction due to numerical issues)
        current_sum = eval_packing(z)
        if current_sum < 0:
             # Try to fix by reducing radii slightly
             z[:, 2] *= 0.99
             current_sum = eval_packing(z)
             if current_sum < 0:
                 return z, -np.inf
        
        best_z = z.copy()
        best_sum = current_sum
        
        # Local search
        for it in range(max_iters):
            if time.time() - start_time > timeout:
                break
            
            improved = False
            # Random order or cyclic order? Cyclic is often faster for convergence, random avoids cycles.
            # Let's use a random order to escape local minima more easily.
            indices = np.random.permutation(N)
            
            for i in indices:
                # Try small perturbations to position
                # We only change position, then recalculate radius.
                # If the new radius + other changes lead to higher sum, accept.
                
                # Current radius
                r_old = z[i, 2]
                
                # Try a few random small moves
                for _ in range(5):
                    dx = np.random.randn() * 0.01
                    dy = np.random.randn() * 0.01
                    
                    new_x = np.clip(z[i, 0] + dx, 0.0, 1.0)
                    new_y = np.clip(z[i, 1] + dy, 0.0, 1.0)
                    
                    # Calculate max radius at new position
                    other_idx = [j for j in range(N) if j != i]
                    if other_idx:
                        new_r = max_radius_for_circle(new_x, new_y, z[other_idx, 0], z[other_idx, 1], z[other_idx, 2])
                    else:
                        new_r = 0.5
                    
                    # If the new radius is significantly larger, update
                    # We don't strictly need to check the whole packing validity if we use max_radius,
                    # but we must ensure it doesn't violate constraints with the *current* other circles.
                    # max_radius_for_circle ensures this.
                    
                    if new_r > r_old + 1e-6:
                        z[i, 0] = new_x
                        z[i, 1] = new_y
                        z[i, 2] = new_r
                        improved = True
                        
                        # Update best if this specific move improved the total sum
                        # We can approximate the new sum without full eval for speed, or just do full eval occasionally
                        # Since we only changed one circle's radius, and it increased, the sum increases by (new_r - r_old)
                        # However, moving the center might have affected the *potential* radius of other circles?
                        # No, we only update the radius of the moved circle. The others remain fixed.
                        # So the sum definitely increases.
                        
                        # Check global best
                        # To save time, we can update best_sum incrementally
                        # But eval_packing is cheap enough (N=26)
                        # Let's just track it
                        pass

            # Periodically do a full evaluation to update best_solution
            if it % 10 == 0 or improved:
                current_sum = eval_packing(z)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_z = z.copy()
            
            if not improved and it > 100:
                # Stagnation: try a larger jump or restart?
                # For now, just continue. Randomness in index selection helps.
                pass
                
        return best_z, best_sum

    def generate_grid_init(rows, cols, offset_x=0.0, offset_y=0.0):
        """Generate a grid initialization."""
        pts = []
        for i in range(rows):
            for j in range(cols):
                x = (j + 0.5) / cols + offset_x
                y = (i + 0.5) / rows + offset_y
                if 0 <= x <= 1 and 0 <= y <= 1:
                    pts.append((x, y))
        
        if len(pts) >= N:
            np.random.shuffle(pts)
            selected = pts[:N]
            init = np.zeros((N, 3))
            for i, (x, y) in enumerate(selected):
                init[i, 0] = x
                init[i, 1] = y
                init[i, 2] = 0.0 # Will be calculated
            return init
        return None

    def generate_hex_init(scale, offset_x, offset_y):
        """Generate a hexagonal initialization."""
        pts = []
        row = 0
        while True:
            y = row * scale * np.sqrt(3) / 2 + offset_y
            if y > 1.0:
                break
            col = 0
            while True:
                x = col * scale + offset_x + (row % 2) * scale / 2
                if x > 1.0:
                    break
                pts.append((x, y))
                col += 1
            row += 1
        
        if len(pts) >= N:
            np.random.shuffle(pts)
            selected = pts[:N]
            init = np.zeros((N, 3))
            for i, (x, y) in enumerate(selected):
                init[i, 0] = x
                init[i, 1] = y
                init[i, 2] = 0.0
            return init
        return None

    # Generate diverse initial configurations
    inits = []
    
    # Grids
    for rows in range(3, 7):
        for cols in range(3, 7):
            if rows * cols >= N:
                init = generate_grid_init(rows, cols)
                if init is not None:
                    inits.append(init)
    
    # Hexagons
    for scale in [0.15, 0.18, 0.22, 0.25]:
        for ox in [0.0, 0.05, 0.1]:
            for oy in [0.0, 0.05, 0.1]:
                init = generate_hex_init(scale, ox, oy)
                if init is not None:
                    inits.append(init)
    
    # Random
    for _ in range(10):
        rx = np.random.rand(N)
        ry = np.random.rand(N)
        inits.append(np.column_stack((rx, ry, np.zeros(N))))
        
    # Run local search from each init
    for init in inits:
        if time.time() - start_time > timeout:
            break
        
        try:
            z_final, score = best_fit_local_search(init, max_iters=500)
            if score > best_score:
                best_score = score
                best_solution = z_final
        except Exception:
            pass
    
    # If no valid solution found, try a simple fallback
    if best_solution is None or best_score < 0:
        # Fallback: place circles in a grid with small radii
        side = int(np.ceil(np.sqrt(N)))
        best_solution = np.zeros((N, 3))
        count = 0
        for i in range(side):
            for j in range(side):
                if count < N:
                    best_solution[count, 0] = (i + 0.5) / side
                    best_solution[count, 1] = (j + 0.5) / side
                    best_solution[count, 2] = 0.5 / side - 0.01
                    count += 1
        # Ensure validity of fallback
        # This grid might have overlaps if side is small?
        # 26 circles. 5x5 grid. dist = 0.2. r = 0.09. 2r = 0.18 < 0.2. Valid.
        
    return best_solution
