import time
import sys
import numpy as np

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Build edge position map
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    a, b = min(u, v), max(u, v)
    edge_pos[(a, b)] = i + 1

def get_pos(u, v):
    a, b = min(u, v), max(u, v)
    return edge_pos[(a, b)]

def find_hamilton_cycle_bitset(adj_bitset, n, deadline, time_start):
    """
    Find Hamilton cycle using bitset-based backtracking.
    adj_bitset[v] is a numpy int64 bitmask of neighbors of v.
    """
    # Quick degree check
    for v in range(n):
        if bin(adj_bitset[v]).count('1') < 2:
            return None
    
    # Fix vertex 0 as start
    used_mask = 1 << 0
    path = [0]
    
    # Precompute neighbor lists sorted by degree (MRV heuristic)
    neighbor_order = []
    for v in range(n):
        neighbors = [u for u in range(n) if (adj_bitset[v] >> u) & 1]
        neighbors.sort(key=lambda x: bin(adj_bitset[x]).count('1'))
        neighbor_order.append(neighbors)
    
    def backtrack(depth):
        if time.time() - time_start > deadline:
            return None
        
        if depth == n:
            last = path[-1]
            if (adj_bitset[last] >> 0) & 1:
                return path[:]
            return None
        
        last = path[-1]
        
        for nxt in neighbor_order[last]:
            if (used_mask >> nxt) & 1:
                continue
            
            # Check if nxt can connect to 0 if it's the last vertex
            if depth == n - 1:
                if not ((adj_bitset[nxt] >> 0) & 1):
                    continue
            
            new_used = used_mask | (1 << nxt)
            path.append(nxt)
            
            # Pruning: check if remaining vertices can form a path
            # that connects back to 0
            if depth < n - 1:
                remaining_mask = ((1 << n) - 1) & ~new_used
                if remaining_mask == 0:
                    path.pop()
                    continue
                    
                # Check connectivity of remaining vertices
                # and that they can connect to path
                rem_vertices = [v for v in range(n) if (remaining_mask >> v) & 1]
                if rem_vertices:
                    # BFS on remaining subgraph
                    visited = set([rem_vertices[0]])
                    stack = [rem_vertices[0]]
                    while stack:
                        u = stack.pop()
                        for w in rem_vertices:
                            if w not in visited and ((adj_bitset[u] >> w) & 1):
                                visited.add(w)
                                stack.append(w)
                    if len(visited) != len(rem_vertices):
                        path.pop()
                        continue
                    
                    # Check at least one remaining vertex connects to path
                    path_set = set(path)
                    has_conn = False
                    for v in rem_vertices:
                        for w in path_set:
                            if (adj_bitset[v] >> w) & 1:
                                has_conn = True
                                break
                        if has_conn:
                            break
                    if not has_conn:
                        path.pop()
                        continue
            
            result = backtrack(depth + 1)
            if result is not None:
                return result
            
            path.pop()
        
        return None
    
    return backtrack(1)

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    n = N
    total_edges = n * (n - 1) // 2
    
    # Build sorted edges by position
    sorted_edges = [(pos, u, v) for (u, v), pos in edge_pos.items()]
    sorted_edges.sort()
    
    best_cycle = None
    best_score = total_edges + 1
    
    # Binary search for minimum threshold
    lo, hi = 24, total_edges
    
    while lo <= hi and (time.time() - start_time) < deadline:
        mid = (lo + hi) // 2
        
        # Build adjacency bitset for edges with position <= mid
        adj_bitset = np.zeros(n, dtype=np.int64)
        for pos, u, v in sorted_edges:
            if pos > mid:
                break
            adj_bitset[u] |= (1 << v)
            adj_bitset[v] |= (1 << u)
        
        cycle = find_hamilton_cycle_bitset(adj_bitset, n, deadline - (time.time() - start_time), start_time)
        
        if cycle is not None:
            best_cycle = cycle
            best_score = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_cycle is not None:
        return best_cycle
    
    # Fallback: full graph
    adj_bitset = np.zeros(n, dtype=np.int64)
    for (u, v) in EDGE_ORDER:
        adj_bitset[u] |= (1 << v)
        adj_bitset[v] |= (1 << u)
    
    cycle = find_hamilton_cycle_bitset(adj_bitset, n, deadline - (time.time() - start_time), start_time)
    if cycle is not None:
        return cycle
    
    return list(range(n))
