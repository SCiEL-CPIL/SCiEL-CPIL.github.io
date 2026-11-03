import numpy as np
import time
import math
import random

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin for 150s limit
    
    def compute_radius(centers):
        """
        Computes the max radius for the given centers.
        Returns 0 if invalid.
        """
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        # Check bounds
        if np.any(np.abs(centers) > 0.5 + 1e-9):
            return 0.0
        
        # Wall distance
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        # Pair distance
        # Use vectorized distance calculation
        # diff[i, j] = c_i - c_j
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        # Set diagonal to inf
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq)
        min_pair = np.sqrt(min_dist_sq)
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def hexagonal_init(rows, cols, jitter=0.0, seed=0):
        """
        Generates a hexagonal lattice packing.
        """
        rng = np.random.RandomState(seed)
        # Estimate d based on cols
        d = 1.0 / cols * 0.99
        
        # Check if it fits vertically
        row_spacing = d * math.sqrt(3) / 2
        total_height = (rows - 1) * row_spacing + d
        if total_height > 1.0:
            # Shrink d to fit
            d = 1.0 / ((rows - 1) * math.sqrt(3) / 2 + 1) * 0.99
            
        pts = []
        for row in range(rows):
            y = -0.5 + d/2 + row * (d * math.sqrt(3) / 2)
            if row % 2 == 0:
                x_start = -0.5 + d/2
                for col in range(cols):
                    x = x_start + col * d
                    if jitter > 0:
                        x += rng.uniform(-jitter*d, jitter*d)
                        y += rng.uniform(-jitter*d, jitter*d)
                    pts.append([x, y])
            else:
                x_start = -0.5 + d
                for col in range(cols - 1):
                    x = x_start + col * d
                    if jitter > 0:
                        x += rng.uniform(-jitter*d, jitter*d)
                        y += rng.uniform(-jitter*d, jitter*d)
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) >= N:
            return pts[:N]
        else:
            return None

    def square_grid_init(m, jitter=0.0, seed=0):
        rng = np.random.RandomState(seed)
        d = 1.0 / m
        pts = []
        for i in range(m):
            for j in range(m):
                x = -0.5 + (i + 0.5) * d
                y = -0.5 + (j + 0.5) * d
                if jitter > 0:
                    x += rng.uniform(-jitter*d, jitter*d)
                    y += rng.uniform(-jitter*d, jitter*d)
                pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            return pts[:N]
        else:
            return None

    def simulated_annealing(centers, time_limit=5.0, initial_temp=0.005, cooling_rate=0.999):
        """
        Simulated Annealing on the exact non-smooth objective.
        """
        current_centers = centers.copy()
        current_radius = compute_radius(current_centers)
        best_centers = current_centers.copy()
        best_radius = current_radius
        
        start_time = time.time()
        temp = initial_temp
        rng = np.random.RandomState(42)
        
        # Pre-allocate noise array for speed
        noise = np.zeros((N, 2))
        
        while time.time() < start_time + time_limit:
            # Generate a new candidate by perturbing a random subset of points
            # or all points with small noise? 
            # Perturbing all points is O(N^2) eval. 
            # Let's try perturbing a random subset of size K to reduce cost?
            # No, compute_radius is O(N^2) anyway. 
            # So perturbing all vs subset doesn't change eval cost significantly if we recompute full radius.
            # However, perturbing all points might jump out of local minima better.
            
            # Strategy: Perturb all points with Gaussian noise scaled by temp
            noise[:] = rng.normal(0, temp, (N, 2))
            candidate = current_centers + noise
            
            # Ensure candidate stays within bounds? 
            # If it goes out, compute_radius returns 0, which is a huge penalty.
            # This effectively rejects out-of-bounds moves.
            
            candidate_radius = compute_radius(candidate)
            
            if candidate_radius > current_radius:
                # Always accept improvement
                current_centers = candidate
                current_radius = candidate_radius
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_centers = candidate.copy()
            else:
                # Accept worse solution with probability exp((new - old) / T)
                # Since we are maximizing, delta = new - old is negative.
                delta = candidate_radius - current_radius
                if delta < 0:
                    # Avoid overflow in exp
                    if delta / temp > -700: 
                        prob = math.exp(delta / temp)
                    else:
                        prob = 0.0
                    if rng.random() < prob:
                        current_centers = candidate
                        current_radius = candidate_radius
            
            # Cool down
            temp *= cooling_rate
            if temp < 1e-8:
                temp = 1e-8 # Floor to prevent complete freeze too early? 
                # Actually, let it run until time limit.
                
        return best_centers, best_radius

    # Main Logic
    best_centers = None
    best_radius = 0.0
    
    # Generate diverse initializations
    inits = []
    
    # 1. Hexagonal grids with different row counts and jitters
    for rows in range(12, 18):
        cols = math.ceil(N / rows)
        pts = hexagonal_init(rows, cols, jitter=0.0, seed=rows)
        if pts is not None:
            inits.append(pts)
        for jitter in [0.02, 0.05, 0.1]:
            pts = hexagonal_init(rows, cols, jitter=jitter, seed=rows*100 + int(jitter*100))
            if pts is not None:
                inits.append(pts)
                
    # 2. Square grids
    m = int(np.ceil(np.sqrt(N)))
    for jitter in [0.0, 0.05]:
        pts = square_grid_init(m, jitter=jitter, seed=42)
        if pts is not None:
            inits.append(pts)
            
    # 3. Random
    for seed in range(3):
        rng = np.random.RandomState(seed)
        pts = rng.uniform(-0.49, 0.49, (N, 2))
        inits.append(pts)

    # Evaluate all inits
    for i, init in enumerate(inits):
        if time.time() > deadline - 10:
            break
        r = compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init.copy()
            
    # Sort inits by radius and run SA on the top ones
    init_scores = [(compute_radius(init), i) for i, init in enumerate(inits)]
    init_scores.sort(reverse=True)
    
    # We have ~130s left. 
    # SA is fast. Let's run SA on the top 10 distinct initializations.
    # Each SA run gets ~10s.
    tried_indices = set()
    sa_runs = 0
    max_sa_runs = 10
    
    for score, idx in init_scores:
        if sa_runs >= max_sa_runs:
            break
        if idx in tried_indices:
            continue
        tried_indices.add(idx)
        
        if time.time() > deadline - 15:
            break
            
        # Run SA on this init
        # Start with a temperature slightly larger than the current radius gap?
        # Radius is ~0.03. A move of 0.005 is reasonable.
        optimized_centers, r_opt = simulated_annealing(inits[idx], time_limit=10.0, initial_temp=0.005)
        
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = optimized_centers.copy()
            
        sa_runs += 1
        
    # Final polish: Run SA on the absolute best result found, with a smaller initial temp
    if best_centers is not None and time.time() < deadline - 5:
        final_centers, r_final = simulated_annealing(best_centers, time_limit=10.0, initial_temp=0.001)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = final_centers.copy()

    if best_centers is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
