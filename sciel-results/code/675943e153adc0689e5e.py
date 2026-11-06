"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

N = 9
FULL_MASK = (1 << N) - 1
ALL_BITS = [1 << b for b in range(N)]
DEADLINE_FRACTION = 0.85  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _popcount(x):
    return x.bit_count()

def _greedy_seed():
    """Fast greedy to get a starting snake."""
    path = [0]
    used = {0}
    # For each vertex in path, precompute which neighbors are "forbidden" (Hamming dist 1)
    # We'll just do the simple check
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check against all non-consecutive vertices
            valid = True
            for i in range(len(path) - 1):
                if _popcount(v ^ path[i]) == 1:
                    valid = False
                    break
            if valid:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
    return path

def _dfs_extend(path, used, depth_limit, time_limit, best_path):
    """
    DFS with backtracking to extend the path.
    path: current snake
    used: set of vertices in path
    depth_limit: max length to try
    time_limit: deadline (absolute time)
    best_path: [list] to store the best found
    Returns True if found a path longer than current best.
    """
    cur = path[-1]
    # Generate candidate neighbors
    candidates = []
    for b in range(N):
        v = cur ^ (1 << b)
        if v in used:
            continue
        # Check induced constraint against all non-consecutive
        valid = True
        for i in range(len(path) - 1):
            if _popcount(v ^ path[i]) == 1:
                valid = False
                break
        if valid:
            candidates.append(v)
    
    if not candidates:
        if len(path) > len(best_path[0]):
            best_path[0] = list(path)
        return
    
    # Shuffle for diversity
    random.shuffle(candidates)
    
    for v in candidates:
        if time.time() > time_limit:
            if len(path) > len(best_path[0]):
                best_path[0] = list(path)
            return
        path.append(v)
        used.add(v)
        
        _dfs_extend(path, used, depth_limit, time_limit, best_path)
        
        path.pop()
        used.remove(v)
        
        if time.time() > time_limit:
            if len(path) > len(best_path[0]):
                best_path[0] = list(path)
            return

def _local_search_improve(path, time_limit):
    """
    Try to improve the path by local moves: 
    - Try to extend at the end
    - Try to re-route the last few steps
    """
    best = list(path)
    best_len = len(best)
    
    used = set(best)
    
    # Try extending
    while True:
        cur = best[-1]
        extended = False
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            valid = True
            for i in range(len(best) - 1):
                if _popcount(v ^ best[i]) == 1:
                    valid = False
                    break
            if valid:
                best.append(v)
                used.add(v)
                extended = True
                break
        if not extended:
            break
    
    return best

def _try_different_starts(starts, time_limit, best_path):
    """Try DFS from multiple starting vertices."""
    for start in starts:
        if time.time() > time_limit:
            break
        path = [start]
        used = {start}
        _dfs_extend(path, used, 200, time_limit, best_path)
        if len(best_path[0]) > 150:
            break

def construct():
    start_time = time.time()
    deadline = start_time + 120 * DEADLINE_FRACTION  # 102 seconds
    
    # Step 1: Get a greedy seed
    seed_path = _greedy_seed()
    best_path = [seed_path]
    
    # Step 2: Try to improve with DFS from the seed
    if time.time() < deadline:
        # Try DFS with random restarts
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Start from the greedy seed but with some randomization
            if attempt == 0:
                path = list(seed_path)
            else:
                # Try a different random start
                random_start = random.randint(0, (1 << N) - 1)
                path = [random_start]
            
            used = set(path)
            best_before = len(best_path[0])
            
            _dfs_extend(path, used, 250, deadline, best_path)
            
            if len(best_path[0]) > best_before:
                # Found improvement, continue with this
                seed_path = list(best_path[0])
    
    # Step 3: Local improvement on the best found
    if time.time() < deadline:
        improved = _local_search_improve(best_path[0], deadline)
        if len(improved) > len(best_path[0]):
            best_path[0] = improved
    
    # Return the best path found
    return best_path[0]

# EVOLVE-BLOCK-END
