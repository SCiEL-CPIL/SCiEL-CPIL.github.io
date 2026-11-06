import time
import random
import sys

N = 9
FULL_MASK = (1 << N) - 1
TIME_LIMIT = 115  # Safety margin before 120s kill

# Precompute neighbors for each vertex to speed up checks
# neighbors[v] is a list of vertices u such that u = v ^ (1 << b)
neighbors = [[] for _ in range(1 << N)]
for v in range(1 << N):
    for b in range(N):
        u = v ^ (1 << b)
        neighbors[v].append(u)

def _is_valid_extension(v, path, used):
    """
    Check if v can be appended to path.
    v must not be in used.
    v must not be adjacent to any vertex in path[:-1] (the non-consecutive ones).
    """
    if v in used:
        return False
    
    # Check against all previous vertices except the last one (which is the predecessor)
    # path[-1] is the current end. We are adding v.
    # So v must not be adjacent to path[0] ... path[-2]
    for i in range(len(path) - 1):
        if _hamming(v, path[i]) == 1:
            return False
            
    return True

def _hamming(x, y):
    return (x ^ y).bit_count()

def _get_candidates(cur, path, used):
    """
    Get valid candidates for the next step from cur.
    """
    cands = []
    for v in neighbors[cur]:
        if v in used:
            continue
        # Check induced condition
        valid = True
        for i in range(len(path) - 1):
            if _hamming(v, path[i]) == 1:
                valid = False
                break
        if valid:
            # Calculate heuristic: number of unvisited neighbors of v
            # that are also valid extensions from v?
            # Simple heuristic: number of unvisited neighbors
            deg = 0
            for u in neighbors[v]:
                if u not in used:
                    deg += 1
            cands.append((deg, v))
            
    return cands

def _dfs(path, used, best_path, deadline):
    """
    Recursive DFS with backtracking.
    """
    if time.time() > deadline:
        return
    
    cur = path[-1]
    
    # Update best path
    if len(path) > len(best_path):
        best_path.clear()
        best_path.extend(path)
        
    # Get candidates
    cands = _get_candidates(cur, path, used)
    
    if not cands:
        return
        
    # Sort by heuristic (Warnsdorff: lowest degree first)
    # Add randomness to ties
    random.shuffle(cands)
    cands.sort(key=lambda x: x[0])
    
    for deg, v in cands:
        # Try extending
        path.append(v)
        used.add(v)
        
        _dfs(path, used, best_path, deadline)
        
        path.pop()
        used.remove(v)
        
        if time.time() > deadline:
            return

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_path = []
    
    # To improve the score, we can run multiple random restarts
    # or use a smarter initialization.
    # Let's start with 0 and a fixed bit order, then randomize.
    
    # Initial attempt
    path = [0]
    used = {0}
    _dfs(path, used, best_path, deadline)
    
    # If we have time, try random restarts
    # Note: _dfs is recursive and might be slow.
    # We can also try to extend the best path found so far using local search?
    # But backtracking should handle that.
    
    # To ensure we make progress, we can run a few random restarts if the first one was fast
    # or just let the first DFS run for a while.
    
    # Actually, a single DFS might get stuck. Random restarts help.
    # Let's run a loop of random starts if time permits.
    
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Try a few random starts
        for _ in range(10): # Max 10 restarts, will break on time
            if time.time() > deadline - 2:
                break
            start_v = random.randint(0, FULL_MASK)
            path = [start_v]
            used = {start_v}
            _dfs(path, used, best_path, deadline)
            
    return list(best_path)
