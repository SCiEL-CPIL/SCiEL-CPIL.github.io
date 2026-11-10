# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import concurrent.futures
import os

# Global variables for worker processes
_GRID = None
_N = 8

def _worker_init(grid_data):
    """
    Initialize global variables in the worker process.
    """
    global _GRID
    _GRID = grid_data

def _run_greedy(seed):
    """
    Run a single greedy pass with a specific random seed.
    Returns the best set of points found.
    """
    global _GRID
    n = _N
    grid = _GRID
    
    # Shuffle the grid
    rng = np.random.RandomState(seed)
    order = grid[rng.permutation(len(grid))]
    
    arr = np.empty((0, 3), dtype=np.int64)
    
    # Pre-compute squared norms for the grid to save time
    # Actually, we only need it for points in arr and p.
    # Let's keep it simple but efficient.
    
    # Helper to check if p can be added to arr
    def can_add(p):
        m = len(arr)
        if m < 4:
            return True
        
        # Build the matrix W
        # Rows: q - p, |q|^2 - |p|^2
        # arr is (m, 3)
        # p is (3,)
        
        # q - p
        diffs = arr - p
        
        # |q|^2 - |p|^2
        sq_norms_q = (arr ** 2).sum(axis=1)
        sq_norm_p = (p ** 2).sum()
        
        # W is (m, 4)
        W = np.hstack((diffs, (sq_norms_q - sq_norm_p)[:, np.newaxis]))
        
        # We need to check all combinations of 4 rows
        # If m is large, this is expensive.
        # Optimization: If m is small, do it directly.
        
        # Generate combinations of indices
        # For m > 20, combinations are huge.
        # But we stop adding when no more points can be added.
        # Usually m stays around 30-50.
        
        indices = np.array(list(itertools.combinations(range(m), 4)))
        if len(indices) == 0:
            return True
            
        # Sub-matrices
        # indices is (C, 4)
        # We want W[indices] to be (C, 4, 4)
        sub_matrices = W[indices]
        
        # Compute determinants
        # np.linalg.det works on the last two dimensions
        dets = np.linalg.det(sub_matrices)
        
        # Check if any determinant is close to 0
        # Integer coords -> integer det. |det| >= 1 if non-zero.
        # Threshold 0.5
        return not np.any(np.abs(dets) < 0.5)

    # Greedy loop
    for p in order:
        if can_add(p):
            if len(arr) == 0:
                arr = p[None, :]
            else:
                arr = np.vstack([arr, p])
                
    return arr

def _local_search(seed, initial_points):
    """
    Perform a local search starting from initial_points.
    Tries to add points that were skipped or swap points.
    For now, a simple hill-climbing that tries to add any point from the grid
    that fits, in a different order, or removes one and adds two.
    Let's stick to a simpler "restart with bias" or "swap" strategy.
    
    A simple effective strategy: 
    1. Start with the greedy result.
    2. Try to add any point not in the set. If it fits, add it.
    3. If stuck, try swapping a point p in the set with a point q not in the set,
       if it allows adding a third point r.
    """
    global _GRID
    grid = _GRID
    points = initial_points.copy()
    point_set = set(map(tuple, points))
    
    # First pass: try to add any remaining point
    added = True
    while added:
        added = False
        for p in grid:
            if tuple(p) in point_set:
                continue
            if _can_add_to_set(points, p):
                points = np.vstack([points, p])
                point_set.add(tuple(p))
                added = True
    
    # Second pass: try swaps (remove one, add two)
    # This is expensive, so limit it.
    # We'll just try to find any single point that can be added after removing one existing point.
    # Actually, let's just return the improved set from the first pass.
    # A more sophisticated local search would be better but risky for time.
    
    return points

def _can_add_to_set(points, p):
    """
    Check if p can be added to the set of points.
    """
    m = len(points)
    if m < 4:
        return True
        
    diffs = points - p
    sq_norms_q = (points ** 2).sum(axis=1)
    sq_norm_p = (p ** 2).sum()
    W = np.hstack((diffs, (sq_norms_q - sq_norm_p)[:, np.newaxis]))
    
    indices = np.array(list(itertools.combinations(range(m), 4)))
    if len(indices) == 0:
        return True
        
    sub_matrices = W[indices]
    dets = np.linalg.det(sub_matrices)
    
    return not np.any(np.abs(dets) < 0.5)

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Returns a numpy array of points from the 8x8x8 grid such that no 5 are co-spherical.
    """
    global _GRID, _N
    _N = 8
    _GRID = np.array(list(itertools.product(range(_N), range(_N), range(_N))), dtype=np.int64)
    
    start_time = time.time()
    deadline = start_time + 550.0 # Safety margin
    
    # Determine number of CPUs
    n_cpus = os.cpu_count() or 4
    n_workers = min(n_cpus, 8) # Cap at 8 to avoid overhead
    
    best_set = None
    best_len = 0
    
    # We will run a mix of greedy passes and local searches on the best greedy results.
    # Strategy:
    # 1. Run greedy passes in parallel.
    # 2. Keep the top few best results.
    # 3. Run local search on the best result found so far in the main process (or a dedicated worker)
    #    to try to improve it.
    
    # To simplify and avoid complex inter-process communication for local search,
    # we will just do a lot of greedy passes and hope one is good.
    # Or, we can run greedy, then do a local search in the main process on the best found.
    
    # Let's do:
    # - Parallel greedy passes.
    # - Once a new best is found, we can try to improve it locally in the main thread?
    #   No, that blocks.
    # - Better: Just run many greedy passes. The variance is high.
    # - Even better: Use a "best-first" or "beam search" idea?
    
    # Let's stick to the current parallel greedy but increase the number of restarts
    # and ensure we use the full time budget.
    
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers, 
        initializer=_worker_init, 
        initargs=(_GRID,)
    ) as executor:
        
        # We'll generate an infinite stream of seeds
        seed_gen = iter(range(100000))
        
        # Keep track of futures
        running_futures = []
        
        while time.time() < deadline:
            # Submit new tasks if we have capacity
            # Keep up to n_workers * 2 tasks in flight
            while len(running_futures) < n_workers * 2 and time.time() < deadline:
                try:
                    seed = next(seed_gen)
                except StopIteration:
                    break
                future = executor.submit(_run_greedy, seed)
                running_futures.append(future)
            
            if not running_futures:
                break
            
            # Check for completed futures
            done, not_done = concurrent.futures.wait(running_futures, timeout=1.0)
            
            for f in done:
                running_futures.remove(f)
                try:
                    result = f.result()
                    if result is not None and len(result) > best_len:
                        best_len = len(result)
                        best_set = result.copy()
                        # If we found a new best, we might want to prioritize local search on it?
                        # For now, just record it.
                except Exception as e:
                    # Handle worker errors
                    pass
            
            # If no tasks are running and no more to submit, break
            if not running_futures and time.time() >= deadline:
                 break
                 
            # Small sleep to avoid busy waiting if we are not submitting
            if not running_futures:
                 time.sleep(0.1)
    
    # If we have time left, try to improve the best_set with a local search in the main process
    # This is safe because it's single-threaded and we have a deadline.
    if best_set is not None and time.time() < deadline - 10.0:
        # Try to add any point that fits
        improved = best_set.copy()
        point_set = set(map(tuple, improved))
        added = True
        while added and time.time() < deadline - 5.0:
            added = False
            for p in _GRID:
                if tuple(p) in point_set:
                    continue
                if _can_add_to_set(improved, p):
                    improved = np.vstack([improved, p])
                    point_set.add(tuple(p))
                    added = True
        if len(improved) > len(best_set):
            best_set = improved

    if best_set is None:
        # Fallback: return an empty set or a small valid set
        # A single point is always valid.
        return np.array([[0, 0, 0]], dtype=np.int64)
        
    return best_set.astype(np.int64)

# EVOLVE-BLOCK-END
