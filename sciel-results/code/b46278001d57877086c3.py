import numpy as np

def construct():
    N = 32
    
    def calc_score(centers):
        """Vectorized score calculation: r_i = min(d_wall_i, 0.5 * min_j d_ij)"""
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Set diagonal to inf so it doesn't affect min
        np.fill_diagonal(dists, np.inf)
        
        # Minimum distance to any other circle
        min_nn_dist = np.min(dists, axis=1)
        
        # Distance to walls
        dx = np.minimum(centers[:, 0], 1.0 - centers[:, 0])
        dy = np.minimum(centers[:, 1], 1.0 - centers[:, 1])
        d_wall = np.minimum(dx, dy)
        
        # Radius for each circle
        radii = np.minimum(d_wall, 0.5 * min_nn_dist)
        
        return np.sum(radii), radii

    # Initialization: Grid with small jitter for 32 circles (8x4 or 4x8)
    rng = np.random.default_rng(42)
    
    # Try multiple initial configurations and pick the best
    best_score = -np.inf
    best_centers = None
    best_radii = None
    
    for trial in range(5):
        # Different grid layouts: 8x4, 4x8, or jittered 6x6 (take 32)
        if trial == 0:
            # 8 columns x 4 rows
            cols, rows = 8, 4
            centers = np.zeros((N, 2))
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if idx < N:
                        centers[idx, 0] = (c + 0.5) / cols
                        centers[idx, 1] = (r + 0.5) / rows
                        idx += 1
        elif trial == 1:
            # 4 columns x 8 rows
            cols, rows = 4, 8
            centers = np.zeros((N, 2))
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if idx < N:
                        centers[idx, 0] = (c + 0.5) / cols
                        centers[idx, 1] = (r + 0.5) / rows
                        idx += 1
        elif trial == 2:
            # 6x6 grid, take first 32
            cols, rows = 6, 6
            centers = np.zeros((N, 2))
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if idx < N:
                        centers[idx, 0] = (c + 0.5) / cols
                        centers[idx, 1] = (r + 0.5) / rows
                        idx += 1
        elif trial == 3:
            # Hexagonal-ish: offset rows
            cols, rows = 6, 6
            centers = np.zeros((N, 2))
            idx = 0
            for r in range(rows):
                offset = 0.5 / cols if r % 2 == 1 else 0.0
                for c in range(cols):
                    if idx < N:
                        centers[idx, 0] = (c + 0.5 + offset) / cols
                        centers[idx, 1] = (r + 0.5) / rows
                        idx += 1
        else:
            # Random initialization
            centers = rng.uniform(0.05, 0.95, size=(N, 2))
        
        # Add small jitter to break symmetry
        centers = centers + rng.uniform(-0.01, 0.01, size=(N, 2))
        centers = np.clip(centers, 0.02, 0.98)
        
        score, radii = calc_score(centers)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            best_radii = radii.copy()
    
    # Simulated Annealing
    current_score = best_score
    current_centers = best_centers.copy()
    current_radii = best_radii.copy()
    
    T_start = 0.005
    T_end = 0.0001
    steps = 15000
    move_size_start = 0.08
    move_size_end = 0.002
    
    for step in range(steps):
        # Pick a random circle
        i = rng.integers(0, N)
        
        # Move size decreases over time
        frac = step / steps
        move_size = move_size_start * (1 - frac) + move_size_end * frac
        
        # Generate move
        dx = rng.uniform(-move_size, move_size)
        dy = rng.uniform(-move_size, move_size)
        
        old_x = current_centers[i, 0]
        old_y = current_centers[i, 1]
        
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Keep inside box
        new_x = np.clip(new_x, 0.01, 0.99)
        new_y = np.clip(new_y, 0.01, 0.99)
        
        current_centers[i, 0] = new_x
        current_centers[i, 1] = new_y
        
        # Calculate new score
        new_score, new_radii = calc_score(current_centers)
        
        delta = new_score - current_score
        
        # Temperature anneals
        T = T_start * (T_end / T_start) ** (step / steps)
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            # Accept
            current_score = new_score
            current_radii = new_radii
            
            if current_score > best_score:
                best_score = current_score
                best_centers = current_centers.copy()
                best_radii = current_radii.copy()
        else:
            # Reject
            current_centers[i, 0] = old_x
            current_centers[i, 1] = old_y

    # Final result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [best_centers[i, 0], best_centers[i, 1], best_radii[i]]
        
    return result
