"""Force-directed repulsion + simulated annealing for 50-circle packing."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(cents):
    """Compute the packing radius (the score)."""
    n = len(cents)
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
    else:
        pair_min = 0.5
    return min(wall_min, pair_min)

def _construct_hexagonal():
    """Create hexagonal packing initial arrangement."""
    best_cents = None
    best_r = 0.0

    for n_rows in [5, 6, 7, 8]:
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]

        max_rc = max(row_counts)
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)

        if r <= best_r:
            continue

        centers = []
        total_y = (n_rows - 1) * 2 * r * np.sqrt(3.0) / 2.0
        y_start = -0.5 + r + total_y / 2.0

        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * 2 * r * np.sqrt(3.0) / 2.0
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            offset = (2 * r * np.sqrt(3.0) / 2.0) * 0.5 * (row_idx % 2)
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])

        centers = np.array(centers[:N])
        r_actual = _score(centers)

        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()

    if best_cents is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _score(best_cents)

    return best_cents, best_r

def _construct_grid():
    """Create a simple grid initial arrangement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N], _score(grid[:N])

def _construct_random(rng):
    """Create a random initial arrangement."""
    cents = rng.uniform(-0.4, 0.4, size=(N, 2))
    return cents, _score(cents)

def _force_directed_anneal(cents, time_limit, rng, T0=0.01, T_end=1e-5, cooling=0.9995):
    """
    Force-directed repulsion model + simulated annealing.
    
    Forces:
    - Pairwise repulsion: F_ij = k / d_ij^2 (Coulomb-like), pushes apart
    - Wall repulsion: F_wall = k_wall / d_wall^2, pushes away from walls
    
    The system naturally seeks to maximize minimum distances.
    Simulated annealing allows escape from local minima.
    """
    start_time = time.time()
    best_cents = cents.copy()
    best_r = _score(best_cents)
    n = len(cents)
    
    k_pair = 0.001
    k_wall = 0.001
    T = T0
    
    # Precompute pair indices for efficiency
    pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    while time.time() - start_time < time_limit:
        # Compute forces
        forces = np.zeros_like(best_cents)
        
        # Pairwise repulsion
        for i, j in pair_indices:
            diff = best_cents[i] - best_cents[j]
            d2 = np.dot(diff, diff)
            d = np.sqrt(d2) if d2 > 1e-12 else 1e-6
            if d < 0.15:  # Only apply force when circles are relatively close
                f_mag = k_pair / (d2 + 1e-8)
                forces[i] += f_mag * diff / d
                forces[j] -= f_mag * diff / d
        
        # Wall repulsion
        for i in range(n):
            # Left wall
            d_left = best_cents[i, 0] + 0.5
            if d_left < 0.05:
                f = k_wall / (d_left**2 + 1e-8)
                forces[i, 0] += f
            # Right wall
            d_right = 0.5 - best_cents[i, 0]
            if d_right < 0.05:
                f = k_wall / (d_right**2 + 1e-8)
                forces[i, 0] -= f
            # Bottom wall
            d_bottom = best_cents[i, 1] + 0.5
            if d_bottom < 0.05:
                f = k_wall / (d_bottom**2 + 1e-8)
                forces[i, 1] += f
            # Top wall
            d_top = 0.5 - best_cents[i, 1]
            if d_top < 0.05:
                f = k_wall / (d_top**2 + 1e-8)
                forces[i, 1] -= f
        
        # Random perturbation (simulated annealing)
        noise = rng.normal(0, T, size=best_cents.shape)
        
        # Update positions
        new_cents = best_cents + forces * 0.01 + noise
        new_cents[:, 0] = np.clip(new_cents[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
        new_cents[:, 1] = np.clip(new_cents[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
        
        new_r = _score(new_cents)
        
        # Acceptance criterion (simulated annealing)
        if new_r > best_r:
            best_cents = new_cents
            best_r = new_r
        else:
            # Metropolis acceptance: accept worse solutions with probability exp(-delta/T)
            delta = best_r - new_r
            if delta < 1e-10 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                best_cents = new_cents
        
        # Cool down
        T = max(T * cooling, T_end)
        
        # Occasionally reheat to escape deep local minima
        if rng.random() < 0.005:
            T = max(T * 3.0, T0 * 0.3)
    
    return best_cents, best_r

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(12345)
    
    # Generate multiple diverse initial configurations
    inits = []
    
    # Hexagonal
    c_hex, r_hex = _construct_hexagonal()
    inits.append((c_hex, r_hex))
    
    # Grid
    c_grid, r_grid = _construct_grid()
    inits.append((c_grid, r_grid))
    
    # Several random starts
    for seed in [1, 7, 42, 123, 999]:
        rng_sub = np.random.default_rng(seed)
        c_rand, r_rand = _construct_random(rng_sub)
        inits.append((c_rand, r_rand))
    
    # Sort by initial score, best first
    inits.sort(key=lambda x: -x[1])
    
    best_cents = inits[0][0].copy()
    best_r = inits[0][1]
    
    # Allocate time: give more time to better initial configurations
    # Use 60% of time for the best init, 25% for second, 15% for third
    total_time = deadline - (time.time() - start_time)
    
    # Try top 3 initial configurations with allocated time
    time_allocations = [0.55, 0.30, 0.15]
    
    for idx in range(min(3, len(inits))):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5.0:
            break
        
        alloc = time_allocations[idx] * total_time
        alloc = min(alloc, remaining - 2.0)
        if alloc < 3.0:
            break
        
        c_init, r_init = inits[idx]
        
        # Run force-directed annealing
        c_opt, r_opt = _force_directed_anneal(
            c_init, time_limit=alloc, rng=rng,
            T0=0.008, T_end=1e-6, cooling=0.9997
        )
        
        if r_opt > best_r:
            best_r = r_opt
            best_cents = c_opt
    
    return best_cents
# EVOLVE-BLOCK-END
