# EVOLVE-BLOCK-START
import time
import numpy as np

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    num_tubes = N * N
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit

    # Pre-compute tube indices
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten().astype(np.float64)
    j_flat = j_grid.flatten().astype(np.float64)

    # Define a fast deterministic volume estimator
    # We use a fixed grid of (a, b, c) samples within the bounding box
    # To be fast, we pre-compute c and use a moderate grid for a, b
    # But since the box changes, we must re-normalize or resample.
    # Strategy: Rescale a, b from [0,1] to [X0, X1], [Y0, Y1] on the fly.
    
    # Use a fixed set of random points in [0,1]^3 for the search.
    # This makes the cost function consistent and fast (no rng consumption).
    rng_search = np.random.default_rng(42)
    num_search_pts = 20000
    a_norm = rng_search.uniform(0, 1, num_search_pts)
    b_norm = rng_search.uniform(0, 1, num_search_pts)
    c_samples = rng_search.uniform(0, 1, num_search_pts)
    
    # We will check containment in vectorized batches
    batch_size = 16

    def fast_volume(x, y):
        X0 = np.min(x)
        X1 = np.max(x + (i_flat + 1) / N)
        Y0 = np.min(y)
        Y1 = np.max(y + (j_flat + 1) / N)
        
        w = X1 - X0
        h = Y1 - Y0
        if w <= 1e-9 or h <= 1e-9:
            return np.inf

        # Map normalized samples to the box
        a = X0 + a_norm * w
        b = Y0 + b_norm * h
        
        inside_any = np.zeros(num_search_pts, dtype=bool)
        
        for start_idx in range(0, num_tubes, batch_size):
            end_idx = min(start_idx + batch_size, num_tubes)
            
            bx = x[start_idx:end_idx][:, np.newaxis]
            by = y[start_idx:end_idx][:, np.newaxis]
            bi = i_flat[start_idx:end_idx][:, np.newaxis]
            bj = j_flat[start_idx:end_idx][:, np.newaxis]
            
            pa = a[np.newaxis, :]
            pb = b[np.newaxis, :]
            pc = c_samples[np.newaxis, :]
            
            # Tube boundaries at height c
            left = bx + pc * bi / N
            right = left + 1.0 / N
            bottom = by + pc * bj / N
            top = bottom + 1.0 / N
            
            in_x = (pa >= left) & (pa <= right)
            in_y = (pb >= bottom) & (pb <= top)
            
            inside_any |= np.any(in_x & in_y, axis=0)
            
        return np.mean(inside_any) * w * h

    # 1. Generate a set of diverse initial candidates
    # Candidate A: Keich-like separable (good baseline)
    # Candidate B: Random
    # Candidate C: "Diagonal" structure
    # Candidate D: Anti-diagonal
    # Candidate E: Grid-based with specific offsets
    
    candidates = []
    
    # A: Keich
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)
        
    init_x = keich[i_flat.astype(int)]
    init_y = keich[j_flat.astype(int)]
    candidates.append((init_x.copy(), init_y.copy()))
    
    # B: Random in [0, 0.5]
    rng_init = np.random.default_rng(1)
    candidates.append((rng_init.uniform(0, 0.5, num_tubes), rng_init.uniform(0, 0.5, num_tubes)))
    
    # C: Diagonal (x=i/16, y=j/16)
    diag_x = i_flat / (2*N)
    diag_y = j_flat / (2*N)
    candidates.append((diag_x, diag_y))
    
    # D: Anti-diagonal
    anti_x = (N-1-i_flat) / (2*N)
    anti_y = (N-1-j_flat) / (2*N)
    candidates.append((anti_x, anti_y))
    
    # E: Random permutation of a grid
    # Try a few random configurations
    for seed in range(3):
        rng_init = np.random.default_rng(seed + 10)
        # Random positions in [0, 1]
        candidates.append((rng_init.uniform(0, 1, num_tubes), rng_init.uniform(0, 1, num_tubes)))

    # 2. Evaluate all candidates and pick the best one to start local search
    best_vol = np.inf
    best_x = None
    best_y = None
    for cx, cy in candidates:
        vol = fast_volume(cx, cy)
        if vol < best_vol:
            best_vol = vol
            best_x = cx.copy()
            best_y = cy.copy()
            current_x = cx.copy()
            current_y = cy.copy()
            current_vol = vol

    # 3. Local Search: Coordinate Descent / Simulated Annealing Hybrid
    # We use a finer grid for moves: 1/32
    grid_step = 1.0 / 32.0
    
    T = 0.05
    cooling_rate = 0.995
    min_T = 1e-5
    
    rng_search_move = np.random.default_rng(123)
    
    # Iterate until deadline
    while time.time() < deadline:
        # Check for timeout more frequently
        if rng_search_move.random() < 0.01:
            if time.time() > deadline:
                break
            
        k = rng_search_move.integers(0, num_tubes)
        
        # Choose axis: x or y
        if rng_search_move.random() < 0.5:
            # Move x
            delta = rng_search_move.choice([-1, 1]) * grid_step
            new_x = current_x.copy()
            new_x[k] += delta
            # Keep in reasonable bounds to prevent box expansion
            # If a tube goes too far, the box expands, increasing volume usually.
            # We can clip to the current box range roughly, or just allow it.
            # Let's allow it but check if it creates a huge box.
            new_y = current_y
        else:
            # Move y
            delta = rng_search_move.choice([-1, 1]) * grid_step
            new_y = current_y.copy()
            new_y[k] += delta
            new_x = current_x
            
        new_vol = fast_volume(new_x, new_y)
        
        if new_vol < current_vol:
            current_vol = new_vol
            current_x = new_x
            current_y = new_y
            if new_vol < best_vol:
                best_vol = new_vol
                best_x = current_x.copy()
                best_y = current_y.copy()
        else:
            # Accept with probability exp(-delta / T)
            delta_vol = new_vol - current_vol
            if delta_vol < 0: # Should be positive here
                prob = 1.0
            else:
                prob = np.exp(-delta_vol / T) if T > min_T else 0.0
            
            if rng_search_move.random() < prob:
                current_vol = new_vol
                current_x = new_x
                current_y = new_y
                
        T *= cooling_rate
        if T < min_T:
            T = min_T

    # 4. Final Refinement
    # Do a few more passes with smaller steps
    grid_step_final = 1.0 / 64.0
    T = 0.01
    for _ in range(1000):
        if time.time() > deadline - 10:
            break
        k = rng_search_move.integers(0, num_tubes)
        if rng_search_move.random() < 0.5:
            delta = rng_search_move.choice([-1, 1]) * grid_step_final
            new_x = current_x.copy()
            new_x[k] += delta
            new_y = current_y
        else:
            delta = rng_search_move.choice([-1, 1]) * grid_step_final
            new_y = current_y.copy()
            new_y[k] += delta
            new_x = current_x
            
        new_vol = fast_volume(new_x, new_y)
        if new_vol < current_vol:
            current_vol = new_vol
            current_x = new_x
            current_y = new_y
            if new_vol < best_vol:
                best_vol = new_vol
                best_x = current_x.copy()
                best_y = current_y.copy()
        elif rng_search_move.random() < np.exp(-(new_vol - current_vol) / T):
            current_vol = new_vol
            current_x = new_x
            current_y = new_y
        T *= 0.99

    # Return the best found
    if best_x is None:
        # Fallback to Keich
        positions = np.stack([init_x, init_y], axis=1)
    else:
        positions = np.stack([best_x, best_y], axis=1)
        
    if not np.all(np.isfinite(positions)):
        positions = np.stack([init_x, init_y], axis=1)
        
    return positions

# EVOLVE-BLOCK-END
