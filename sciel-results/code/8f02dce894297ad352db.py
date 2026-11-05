import sys
import time
import random

# Edge list from the task description
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
TOTAL_EDGES = len(P['edge_order'])

# Precompute edge indices and adjacency lists
edge_to_index = {}
adj = [[] for _ in range(N)] # adj[u] = list of (v, index)
for i, edge in enumerate(P['edge_order']):
    u, v = edge
    if u > v:
        u, v = v, u
    edge_to_index[(u, v)] = i + 1
    adj[u].append((v, i + 1))
    adj[v].append((u, i + 1))

# Sort adjacency lists by arrival index to help with heuristics?
# Actually, for a given K, we only care about edges with index <= K.
# It's better to filter inside the DFS or precompute masks.

# Precompute masks for each vertex: adj_mask[u] is a bitmask of all neighbors
adj_mask = [0] * N
for u in range(N):
    for v, idx in adj[u]:
        adj_mask[u] |= (1 << v)

def is_connected(mask):
    """Check if the subgraph induced by 'mask' is connected."""
    if mask == 0:
        return True
    # Find the first set bit
    start = -1
    m = mask
    while m:
        if m & 1:
            start = 0
            break
        m >>= 1
        start += 1
    
    if start == -1:
        return True

    visited = 0
    stack = [start]
    visited |= (1 << start)
    
    while stack:
        u = stack.pop()
        # Get neighbors of u that are in 'mask'
        neighbors = adj_mask[u] & mask
        while neighbors:
            v_bit = neighbors & -neighbors
            v = v_bit.bit_length() - 1
            if not (visited & v_bit):
                visited |= v_bit
                stack.append(v)
            neighbors ^= v_bit
        
    return visited == mask

def find_hamiltonian_cycle(K, time_limit=5.0):
    """
    Finds a Hamiltonian cycle in the graph formed by the first K edges.
    Returns the list of vertices in order, or None if not found.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Build adjacency list for the first K edges
    # adj_k[u] = list of v such that edge (u,v) has index <= K
    adj_k = [[] for _ in range(N)]
    for u in range(N):
        for v, idx in adj[u]:
            if idx <= K:
                adj_k[u].append(v)
    
    # Check if any vertex has degree < 2
    for u in range(N):
        if len(adj_k[u]) < 2:
            return None

    # Heuristic: start with vertex 0
    start = 0
    visited = 1 << start
    path = [start]
    
    def dfs(u, visited, path, depth):
        if time.time() > deadline:
            return None
            
        if visited == (1 << N) - 1:
            # Check if we can close the cycle
            if 0 in adj_k[u]:
                return path[:]
            return None
        
        unvisited = (1 << N) - 1 - visited
        
        # Pruning 1: Connectivity of unvisited subgraph
        # Do this every few steps to save time
        if depth % 3 == 0:
            if not is_connected(unvisited):
                return None
        
        # Pruning 2: Degree check in unvisited subgraph
        # If any unvisited vertex has no unvisited neighbors, it's a dead end
        # (unless it's the last one, but we handle that in the base case)
        # Actually, if an unvisited vertex v has 0 neighbors in unvisited,
        # and v is not u (current), then v must be connected to a visited vertex.
        # But the cycle must close. This is complex.
        # Simpler: If any unvisited vertex has 0 neighbors in unvisited, 
        # and it's not the case that we are about to visit the last vertex, 
        # it might be isolated.
        
        # Let's just check if any unvisited vertex has 0 neighbors in unvisited.
        # If so, and there is more than 1 unvisited vertex, it's disconnected?
        # No, it could be connected to visited vertices.
        # But if it has 0 neighbors in unvisited, it MUST be visited next?
        # No, it must be connected to the path.
        
        # Let's use Warnsdorff's rule: sort nexts by number of unvisited neighbors
        nexts = []
        for v in adj_k[u]:
            if unvisited & (1 << v):
                # Count unvisited neighbors of v
                cnt = bin(adj_mask[v] & unvisited).count('1')
                nexts.append((cnt, v))
        
        if not nexts:
            return None
            
        # Sort by count (ascending)
        nexts.sort(key=lambda x: x[0])
        
        for _, v in nexts:
            res = dfs(v, visited | (1 << v), path + [v], depth + 1)
            if res:
                return res
        return None

    return dfs(start, visited, path, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s safety margin
    
    # Binary search for the minimum K
    lo = N
    hi = TOTAL_EDGES
    best_K = hi
    best_cycle = None
    
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        # Allocate more time for smaller K (harder to find)
        time_limit = 5.0
        if mid < 100:
            time_limit = 10.0
        elif mid < 200:
            time_limit = 8.0
            
        cycle = find_hamiltonian_cycle(mid, time_limit)
        if cycle:
            best_cycle = cycle
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_cycle is None:
        # Fallback to identity if search fails (should not happen)
        return list(range(N))
        
    return best_cycle
