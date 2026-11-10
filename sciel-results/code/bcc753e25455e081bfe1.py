# EVOLVE-BLOCK-START
import numpy as np
import time


def _union_length_1d(base, N):
    """Compute the length of the union of intervals at height c for a 1D problem.
    
    For base positions base[i] and grid index i, the interval at height c is
    [base[i] + c*i/N, base[i] + c*i/N + 1/N].
    
    We integrate over c in [0,1] to get the effective 1D measure.
    We use a fine grid in c for accurate integration.
    """
    # Use a moderate number of c-samples for speed during optimization
    c_samples = np.linspace(0, 1, 64)
    total_length = 0.0
    for c in c_samples:
        starts = base + c * np.arange(N) / N
        ends = starts + 1.0 / N
        # Sort by start
        order = np.argsort(starts)
        s = starts[order]
        e = ends[order]
        # Merge intervals
        union_len = 0.0
        cur_s, cur_e = s[0], e[0]
        for k in range(1, N):
            if s[k] > cur_e:
                union_len += cur_e - cur_s
                cur_s, cur_e = s[k], e[k]
            else:
                cur_e = max(cur_e, e[k])
        union_len += cur_e - cur_s
        total_length += union_len
    return total_length / len(c_samples)


def _cost_1d(base, N):
    """The 1D cost is the union length integrated over c.
    We minimize this to minimize the 2D product area, hence 3D volume."""
    return _union_length_1d(base, N)


def _cost_3d(base_x, base_y, N):
    """Full 3D cost: product of 1D costs (since union is a product)."""
    cx = _union_length_1d(base_x, N)
    cy = _union_length_1d(base_y, N)
    # The actual volume involves the bounding box area times fraction inside
    # But for optimization, minimizing the product of 1D union lengths
    # (normalized by the bounding box) is what matters.
    # Bounding box in x: min(base_x) to max(base_x + (i+1)/N)
    x0 = np.min(base_x)
    x1 = np.max(base_x + (np.arange(N) + 1) / N)
    y0 = np.min(base_y)
    y1 = np.max(base_y + (np.arange(N) + 1) / N)
    box_area = (x1 - x0) * (y1 - y0)
    if box_area <= 1e-9:
        return 1e6
    # The fraction of the box covered by the union
    # For a product union, fraction = (cx / (x1-x0)) * (cy / (y1-y0))
    # Volume = box_area * fraction = cx * cy
    return cx * cy


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Uses simulated annealing to optimize the 1D base positions.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Initial Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    best_x = keich.copy()
    best_y = keich.copy()
    best_cost = _cost_3d(best_x, best_y, N)
    
    # Simulated annealing
    T = 0.01
    T_min = 1e-6
    cooling = 0.995
    step_size = 0.02
    max_steps = 50000
    
    current_x = best_x.copy()
    current_y = best_y.copy()
    current_cost = best_cost
    
    rng = np.random.default_rng(42)
    
    for step in range(max_steps):
        if time.time() > deadline - 30.0:
            break
        
        # Choose which dimension to perturb
        if rng.random() < 0.5:
            # Perturb x
            idx = rng.integers(N)
            delta = rng.normal(0, step_size)
            new_x = current_x.copy()
            new_x[idx] += delta
            new_cost = _cost_3d(new_x, current_y, N)
            if new_cost < current_cost or rng.random() < np.exp(-(new_cost - current_cost) / T):
                current_x = new_x
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = current_x.copy()
        else:
            # Perturb y
            idx = rng.integers(N)
            delta = rng.normal(0, step_size)
            new_y = current_y.copy()
            new_y[idx] += delta
            new_cost = _cost_3d(current_x, new_y, N)
            if new_cost < current_cost or rng.random() < np.exp(-(new_cost - current_cost) / T):
                current_y = new_y
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_y = current_y.copy()
        
        T = max(T_min, T * cooling)
    
    # Build positions array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_x[i_flat]
    base_y = best_y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END
