import numpy as np
import time
from scipy.optimize import minimize_scalar

def _area_fast(x):
    """
    Compute the area of the union of the 64 triangles.
    x: shape (64,) array of base positions.
    """
    n = 64
    a = (np.arange(n) + 1) / n
    # y_m = (m + 0.5) / 20000
    y = (np.arange(20000) + 0.5) / 20000.0
    
    # L_i = x_i + a_i * y
    # L shape: (64, 20000)
    L = x[:, None] + a[:, None] * y[None, :]
    
    # w = (1 - y) / 64
    w = (1.0 - y) / n
    
    # Sort L along axis 0
    L_sorted = np.sort(L, axis=0)
    
    # Union length = w + sum over consecutive pairs of min(L_{j+1} - L_j, w)
    # Diff shape: (63, 20000)
    diffs = np.diff(L_sorted, axis=0)
    
    # min(diffs, w)
    # w shape: (20000,) -> broadcast to (63, 20000)
    mins = np.minimum(diffs, w[None, :])
    
    # Sum over axis 0 (the 63 pairs)
    union_lengths = w + np.sum(mins, axis=0)
    
    # Area is mean of union lengths
    area = np.mean(union_lengths)
    return area

def _keich_positions():
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)

def _aligned_positions():
    # Try to align the triangles more closely
    # The Keich construction is good, but maybe we can tweak it.
    # Or just return Keich for now as a baseline for local search.
    return _keich_positions()

def _random_positions(seed):
    rng = np.random.RandomState(seed)
    # Initial positions in [0, 1]
    return rng.rand(64)

def _local_search(x0, max_time, seed):
    """
    Coordinate descent with analytic 1D line search (Brent).
    """
    start_time = time.time()
    x = x0.copy()
    current_area = _area_fast(x)
    best_x = x.copy()
    best_area = current_area
    
    n = 64
    rng = np.random.RandomState(seed)
    
    while time.time() - start_time < max_time:
        improved = False
        # Random order of coordinates
        order = rng.permutation(n)
        for i in order:
            if time.time() - start_time > max_time - 5: # Safety margin
                break
            
            # Define 1D function to minimize: f(t) = area(x with x_i = t)
            def f(t):
                x_new = x.copy()
                x_new[i] = t
                return _area_fast(x_new)
            
            # Bounds for t: The triangles should probably stay within a reasonable range.
            # The x_i are base positions. The apex is at x_i + a_i.
            # To minimize area, we want them to overlap.
            # Let's bound t to [-1, 2] or something similar.
            # Actually, shifting all x by a constant doesn't change area.
            # So we can fix one coordinate or just let it float.
            # Let's use bounds [-1, 2] to be safe.
            
            res = minimize_scalar(f, bounds=(-1.0, 2.0), method='bounded', options={'xatol': 1e-8})
            
            if res.success and res.fun < current_area:
                x[i] = res.x
                current_area = res.fun
                improved = True
                if current_area < best_area:
                    best_area = current_area
                    best_x = x.copy()
        
        if not improved:
            # No improvement in a full pass. Maybe break or continue with smaller steps?
            # Brent is global on the 1D slice, so if no coordinate improves, we are at a local min.
            # We can break.
            break
            
    return best_x, best_area

def kakeya_needle_2d_positions64() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0 # 10s safety margin
    
    best_x = _keich_positions()
    best_area = _area_fast(best_x)
    
    # Initializations
    inits = [
        _keich_positions(),
        _random_positions(1),
        _random_positions(42),
        _random_positions(123),
        # Maybe a perturbed Keich
        _keich_positions() + np.random.RandomState(7).randn(64) * 0.1
    ]
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # Allocate remaining time minus a buffer for this run
        # If we have 10 inits, give each ~ (remaining_time) / (inits_left)
        # Simpler: give each a fixed chunk or just run until deadline if it's the last one.
        # Let's give each search 60 seconds if we have time, or whatever is left.
        time_left = deadline - time.time()
        if time_left < 10:
            break
        
        # Use 50% of remaining time for this local search to allow for multiple restarts?
        # Or just run one long search?
        # Inspiration 2 suggests time-budgeted coordinate descent.
        # Let's run local search for min(60, time_left * 0.8) seconds.
        search_time = min(60.0, time_left * 0.8)
        
        x_sol, area_sol = _local_search(init, search_time, seed=int(time.time()*1000) % 10000)
        
        if area_sol < best_area:
            best_area = area_sol
            best_x = x_sol
            
    return best_x

# EVOLVE-BLOCK-END
