"""Snake-in-the-box in Q_9: backtracking search with bitset validity checks."""
import time

N = 9
V = 1 << N  # 512 vertices
DEADLINE = 150.0
SAFETY = 10.0  # return 10s before kill

# Precompute neighbors for each vertex
NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(V)]

# Precompute bit_count for all pairs? That's 512*512 = 262144, manageable.
# Actually we need (v ^ p).bit_count() for validity checks.
# Let's precompute a distance matrix as a list of bytearrays or use numpy.
# For speed in the inner loop, let's precompute a 2D array of hamming distances.
# dist[i][j] = hamming distance between i and j
# We can store this as a list of arrays or just compute on the fly.
# Since N=9, hamming distance is at most 9.
# Let's use a numpy array for fast access.
import numpy as np

# dist[i, j] = popcount(i ^ j)
X = np.arange(V, dtype=np.uint32)
# Compute pairwise XOR and popcount
# This is a 512x512 matrix
_xor = (X[:, None] ^ X[None, :])  # 512x512
# Popcount: sum of bits
# np.bitwise_count is not standard in all numpy versions, use a lookup or loop
# For 9 bits, we can do:
dist = np.zeros((V, V), dtype=np.uint8)
for b in range(N):
    dist += (_xor & (1 << b)).astype(np.uint8)
# Now dist[i, j] is the hamming distance

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - SAFETY
    
    best_path = None
    best_len = 0
    
    # State for backtracking
    path = [0]
    used = np.zeros(V, dtype=bool)
    used[0] = True
    # valid[v] = True if v can be appended to the current path
    # A vertex v is valid if:
    # 1. v is not used
    # 2. For all p in path[:-1], hamming(v, p) >= 2
    #    i.e., v is not a neighbor of any vertex in path except the last one
    #    Actually, the condition is: for all p in path, if p != path[-1], then dist(v, p) >= 2
    #    And dist(v, path[-1]) == 1 (since consecutive must differ by 1 bit)
    
    # Let's maintain a set of "blocked" vertices: vertices that are at distance 1 from any vertex in path[:-1]
    # If v is blocked, it cannot be the next vertex (unless it's a neighbor of the last vertex AND not blocked by others)
    # Actually, the condition for v to be a valid next vertex is:
    # - v is a neighbor of path[-1]
    # - v is not in used
    # - For all p in path[:-1], dist(v, p) >= 2
    #   This is equivalent to: v is not in the "neighborhood" of any p in path[:-1]
    #   Where neighborhood(p) = {u : dist(u, p) == 1}
    
    # Let's maintain blocked[v] = True if v is a neighbor of any vertex in path[:-1]
    # Initially, path = [0], so path[:-1] is empty, blocked is all False.
    
    blocked = np.zeros(V, dtype=bool)
    
    # When we add a new vertex v to the path:
    # - The old last vertex (now in path[:-1]) becomes a source of blocking
    # - So we need to mark all neighbors of the old last vertex as blocked
    # - But wait, the new vertex v itself might be a neighbor of some old vertex, 
    #   but that's fine as long as it's not a neighbor of any OTHER old vertex.
    #   Actually, when we extend, we check that v is not blocked.
    #   After adding v, the old last vertex u = path[-2] (if len > 1) or nothing (if len == 1) 
    #   becomes part of the "forbidden neighborhood" for future extensions.
    #   So we mark neighbors of u as blocked.
    
    # But we also need to unblock? No, once a vertex is in path[:-1], its neighbors are permanently blocked for future extensions.
    # However, when we backtrack, we need to unblock.
    
    # Let's track the count of how many path[:-1] vertices are neighbors of each v.
    # block_count[v] = number of p in path[:-1] such that dist(v, p) == 1
    # v is blocked if block_count[v] > 0
    
    block_count = np.zeros(V, dtype=np.int32)
    
    # Initialize: path = [0], path[:-1] = [], so block_count is all 0.
    
    def is_valid_next(v, last):
        if used[v]:
            return False
        if block_count[v] > 0:
            return False
        # Check dist(v, last) == 1
        # We know v is a neighbor of last if v in NEIGHBORS[last]
        # But we're iterating over neighbors, so this is guaranteed.
        return True
    
    # Backtracking search
    # We use iterative deepening or just DFS with a time limit.
    # Let's do DFS and keep the best path found.
    
    # To make it more efficient, we can use a heuristic ordering of neighbors.
    # But for now, let's just do a simple DFS.
    
    # We'll use a stack for iterative DFS to avoid recursion limits.
    # State: (path, used, block_count, last_vertex)
    # But copying used and block_count is expensive.
    
    # Instead, let's do recursive backtracking with undo.
    
    def backtrack():
        nonlocal best_path, best_len, deadline
        if time.time() > deadline:
            return
        
        cur = path[-1]
        
        # Try to extend
        for v in NEIGHBORS[cur]:
            if used[v] or block_count[v] > 0:
                continue
            
            # Add v
            path.append(v)
            used[v] = True
            
            # Mark neighbors of cur as blocked (cur is now in path[:-1])
            old_block_count = np.zeros(9, dtype=np.int32)  # store changes for undo
            changed = []
            for nb in NEIGHBORS[cur]:
                old_block_count[nb] = block_count[nb]
                block_count[nb] += 1
                changed.append(nb)
            
            # Update best
            if len(path) - 1 > best_len:
                best_len = len(path) - 1
                best_path = path.copy()
            
            # Recurse
            backtrack()
            
            # Undo
            for nb in changed:
                block_count[nb] = old_block_count[nb]
            
            used[v] = False
            path.pop()
            
            if time.time() > deadline:
                return
        
        # Also, we should consider that the current path might be a dead end,
        # but we've already tried all extensions.
    
    backtrack()
    
    if best_path is None:
        best_path = [0]
    
    return best_path
