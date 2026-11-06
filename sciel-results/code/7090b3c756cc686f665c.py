import time
import sys

N = 9
TIME_LIMIT = 110.0  # Safety margin from 120s
start_time = time.time()
best_path = []
best_len = 0

# Precompute neighbors for each vertex? 
# There are 512 vertices, each has 9 neighbors.
# neighbors[v] = [v ^ (1<<b) for b in range(N)]
neighbors = [[] for _ in range(1 << N)]
for v in range(1 << N):
    for b in range(N):
        neighbors[v].append(v ^ (1 << b))

def check_time():
    return time.time() - start_time > TIME_LIMIT

def is_valid_extension(v, path):
    # v is the new vertex to add at the end of path
    # Check if v is at distance 1 from any vertex in path[:-1]
    # path[-1] is the current end, so we check path[0:-1]
    # Note: path contains distinct vertices.
    for p in path[:-1]:
        if (v ^ p).bit_count() == 1:
            return False
    return True

def dfs(v, path, used_mask):
    global best_path, best_len
    
    if check_time():
        return
        
    current_len = len(path)
    if current_len > best_len:
        best_len = current_len
        best_path = path.copy()
        
    # Try to extend
    for next_v in neighbors[v]:
        if used_mask & (1 << next_v):
            continue
            
        # Check induced path constraint
        # next_v must not be adjacent to any vertex in path[:-1]
        valid = True
        # Optimization: If path is short, this is fast. If long, it's slow.
        # We can break early if found.
        for p in path[:-1]:
            if (next_v ^ p).bit_count() == 1:
                valid = False
                break
        
        if not valid:
            continue
            
        path.append(next_v)
        used_mask |= (1 << next_v)
        
        dfs(next_v, path, used_mask)
        
        path.pop()
        used_mask ^= (1 << next_v)
        
        if check_time():
            return

def construct():
    global start_time, best_path, best_len
    start_time = time.time()
    best_path = []
    best_len = 0
    
    # Symmetry breaking: Start at 0
    start_v = 0
    path = [start_v]
    used_mask = 1 << start_v
    
    # To ensure we find a decent solution quickly, we can run a greedy seed first
    # but the DFS will find it anyway.
    
    sys.setrecursionlimit(10000)
    
    dfs(start_v, path, used_mask)
    
    return best_path
