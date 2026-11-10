import numpy as np
import time


def _compute_area(positions, n=64, n_heights=20000):
    """Compute the union area using quadrature."""
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/n
    y = (np.arange(n_heights) + 0.5) / n_heights  # y_m = (m+0.5)/n_heights
    w = (1 - y) / n  # width at each height
    
    # L_i = x_i + a_i * y for each triangle i and height m
    # Shape: (n, n_heights)
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort L at each height
    L_sorted = np.sort(L, axis=0)  # (n, n_heights)
    
    # Union length = w + sum over consecutive pairs of min(L_{j+1} - L_j, w)
    diffs = np.diff(L_sorted, axis=0)  # (n-1, n_heights)
    overlaps = np.minimum(diffs, w[np.newaxis, :])  # (n-1, n_heights)
    
    union_lengths = w + np.sum(overlaps, axis=0)  # (n_heights,)
    area = np.mean(union_lengths)
    
    return area


def _local_search(positions, n=64, max_iter=5000, step_size=0.01, n_heights=20000):
    """Perform local search to minimize area."""
    current = positions.copy().astype(np.float64)
    current_area = _compute_area(current, n, n_heights)
    best = current.copy()
    best_area = current_area
    
    np.random.seed(42)
    
    for iter_idx in range(max_iter):
        # Try random perturbations
        idx = np.random.randint(n)
        delta = np.random.uniform(-step_size, step_size)
        trial = current.copy()
        trial[idx] += delta
        
        trial_area = _compute_area(trial, n, n_heights)
        
        if trial_area < current_area:
            current = trial
            current_area = trial_area
            if current_area < best_area:
                best = current.copy()
                best_area = current_area
        
        # Gradually reduce step size
        step_size = max(0.001, step_size * 0.999)
    
    return best, best_area


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Returns positions that minimize the union area of the 64 triangles.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    # Start with the "aligned" initialization: x_i = -a_i + 0.5
    # This centers the triangles and should give good overlap
    a = (np.arange(n) + 1) / n
    positions = -a + 0.5
    
    # Also try the Keich construction as an alternative start
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    keich_positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        keich_positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    keich_positions = np.array(keich_positions, dtype=np.float64)
    
    # Evaluate both starting points
    area1 = _compute_area(positions, n, 20000)
    area2 = _compute_area(keich_positions, n, 20000)
    
    if area2 < area1:
        positions = keich_positions
        area1 = area2
    
    # Try a few random initializations
    np.random.seed(123)
    for _ in range(5):
        random_pos = np.random.uniform(-1, 1, n)
        rand_area = _compute_area(random_pos, n, 20000)
        if rand_area < area1:
            positions = random_pos
            area1 = rand_area
    
    # Perform local search
    best_positions, best_area = _local_search(positions, n, max_iter=3000, step_size=0.05, n_heights=20000)
    
    # Check time and do additional rounds if possible
    elapsed = time.time() - start_time
    if elapsed < 550:
        # Try more iterations with smaller step
        best_positions, best_area = _local_search(best_positions, n, max_iter=2000, step_size=0.01, n_heights=20000)
    
    # Final refinement
    elapsed = time.time() - start_time
    if elapsed < 580:
        best_positions, best_area = _local_search(best_positions, n, max_iter=1000, step_size=0.005, n_heights=20000)
    
    # Ensure finite values
    if not np.all(np.isfinite(best_positions)):
        return np.zeros(n, dtype=np.float64)
    
    return best_positions.astype(np.float64)
