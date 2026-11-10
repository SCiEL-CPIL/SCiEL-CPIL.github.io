# EVOLVE-BLOCK-START
import numpy as np
import time


def _union_length_1d(pos, N):
    """
    Compute the length of the union of N intervals in 1D.
    Interval k is [pos[k], pos[k] + 1/N].
    Uses a sweep-line approach for exact computation.
    """
    n = len(pos)
    if n == 0:
        return 0.0
    w = 1.0 / N
    # Create events: (position, type) where type=0 is start, type=1 is end
    # Sort by position, then by type (starts before ends at same position)
    starts = pos
    ends = pos + w
    all_points = np.concatenate([starts, ends])
    order = np.argsort(all_points, kind='stable')
    sorted_points = all_points[order]
    
    # Determine which are starts and which are ends
    # First n in sorted order... actually we need to track which are starts
    # Better approach: create events array
    events_pos = np.concatenate([starts, ends])
    events_type = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    order = np.lexsort((events_type, events_pos))
    sorted_pos = events_pos[order]
    sorted_type = events_type[order]
    
    total_len = 0.0
    active = 0
    prev_pos = sorted_pos[0]
    
    # Process events
    for i in range(len(sorted_pos)):
        curr_pos = sorted_pos[i]
        if active > 0:
            total_len += curr_pos - prev_pos
        if sorted_type[i] == 0:
            active += 1
        else:
            active -= 1
        prev_pos = curr_pos
    
    return total_len


def _cost_1d(pos, N):
    """
    Cost for 1D problem: minimize the union length of intervals [pos[k], pos[k]+1/N].
    This is the same as _union_length_1d.
    """
    return _union_length_1d(pos, N)


def _compute_volume(positions, N=8, n_samples=500000):
    """
    Compute the Monte Carlo estimate of the union volume.
    positions: (64, 2) array of base positions.
    Returns the estimated volume V.
    """
    N = 8
    i_flat = np.repeat(np.arange(N), N)
    j_flat = np.tile(np.arange(N), N)
    
    x = positions[:, 0]
    y = positions[:, 1]
    
    X0 = np.min(x)
    X1 = np.max(x + (i_flat + 1) / N)
    Y0 = np.min(y)
    Y1 = np.max(y + (j_flat + 1) / N)
    
    box_area = (X1 - X0) * (Y1 - Y0)
    if box_area <= 1e-9:
        return 0.0
    
    rng = np.random.default_rng(12345)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    # Check if each point is inside at least one tube
    # Tube (i, j) at height c: 
    #   x in [x_k + c*i/N, x_k + c*i/N + 1/N]
    #   y in [y_k + c*j/N, y_k + c*j/N + 1/N]
    
    # Vectorized check: for each sample, check all 64 tubes
    # a: (n_samples,), b: (n_samples,), c: (n_samples,)
    # For tube k: a in [x_k + c*i_k/N, x_k + c*i_k/N + 1/N] and b in [y_k + c*j_k/N, y_k + c*j_k/N + 1/N]
    
    # x_k + c*i_k/N: shape (n_samples, 64)
    x_shift = x[None, :] + c[:, None] * i_flat[None, :] / N
    y_shift = y[None, :] + c[:, None] * j_flat[None, :] / N
    
    in_x = (a[:, None] >= x_shift) & (a[:, None] <= x_shift + 1.0 / N)
    in_y = (b[:, None] >= y_shift) & (b[:, None] <= y_shift + 1.0 / N)
    
    in_any = np.any(in_x & in_y, axis=1)
    
    fraction = np.mean(in_any)
    V = fraction * box_area
    return V


def _optimize_1d(N, max_time=500.0, initial_pos=None):
    """
    Optimize 1D positions to minimize union length of intervals [pos[k], pos[k]+1/N].
    Uses gradient-free optimization with a deadline guard.
    """
    start_time = time.time()
    w = 1.0 / N
    
    if initial_pos is None:
        # Use Keich construction as initial guess
        pos_k = int(np.log2(N))
        j_terms = np.arange(1, pos_k + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)
        pos = keich.copy()
    else:
        pos = initial_pos.copy()
    
    best_pos = pos.copy()
    best_cost = _cost_1d(pos, N)
    
    # Use a local search with decreasing step sizes
    step_size = 0.1
    min_step = 1e-6
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() - start_time < max_time:
        improved = False
        for _ in range(100):
            if time.time() - start_time >= max_time:
                break
            # Try random perturbations
            trial = pos + np.random.randn(N) * step_size
            trial_cost = _cost_1d(trial, N)
            if trial_cost < best_cost:
                best_cost = trial_cost
                best_pos = trial.copy()
                pos = trial.copy()
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        if no_improve_count >= max_no_improve:
            step_size *= 0.5
            no_improve_count = 0
            if step_size < min_step:
                break
        
        # Also try coordinate-wise updates
        for k in range(N):
            if time.time() - start_time >= max_time:
                break
            for delta in [-step_size, -step_size/2, step_size/2, step_size]:
                trial = pos.copy()
                trial[k] += delta
                trial_cost = _cost_1d(trial, N)
                if trial_cost < best_cost:
                    best_cost = trial_cost
                    best_pos = trial.copy()
                    pos = trial.copy()
                    improved = True
                    break
    
    return best_pos, best_cost


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    We build N*N = 64 "generalized tubes" indexed by (i, j), i, j in {0, ..., N-1}.
    Tube (i, j) has a base position (x, y) and, at height c in [0, 1], occupies the
    axis-aligned square
        [x + c*i/N, x + 1/N + c*i/N] x [y + c*j/N, y + 1/N + c*j/N].
    
    The task is to choose the 64 base positions so that the VOLUME of the union
    of all tubes is as SMALL as possible.
    
    Key insight: If we place base_x as a function of i only, and base_y as a function
    of j only, then the union volume factorizes. Specifically, the union in x is the
    union of intervals depending on i, and the union in y is the union of intervals
    depending on j. The 3D union volume is approximately the product of the 1D union
    lengths in x and y (times the height, which is 1).
    
    So we can independently optimize the 1D x-positions and 1D y-positions.
    """
    N = 8
    deadline = time.time() + 600.0
    
    # Optimize 1D x-positions
    x_pos, x_cost = _optimize_1d(N, max_time=250.0)
    
    # Optimize 1D y-positions (can reuse same time budget)
    remaining_time = deadline - time.time() - 10.0  # 10s safety margin
    if remaining_time > 50.0:
        y_pos, y_cost = _optimize_1d(N, max_time=remaining_time)
    else:
        y_pos = x_pos.copy()
        y_cost = x_cost
    
    # Build the 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_pos[i_flat]
    base_y = y_pos[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
