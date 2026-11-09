import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGES = P['edge_order']
TOTAL_EDGES = N * (N - 1) // 2

# Map edge to its 1-based index
# We need a fast lookup: edge (u, v) -> index
# Since N=24, we can use a 2D array or a dictionary.
# Dictionary is fine.
edge_index = {}
for i, e in enumerate(EDGES):
    u, v = e
    if u > v:
        u, v = v, u
    edge_index[(u, v)] = i + 1

def build_adjacency(threshold):
    """Build adjacency bitsets for edges with index <= threshold."""
    adj = [0] * N
    for i in range(threshold):
        u, v = EDGES[i]
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def has_hamiltonian_cycle(adj):
    """
    Check if graph with adjacency bitsets 'adj' has a Hamiltonian cycle.
    Returns the path if found, else None.
    Uses DFS with heuristic: visit neighbors with fewest unvisited neighbors first.
    """
    start = 0
    full_mask = (1 << N) - 1
    
    # Precompute degrees for heuristic? Or compute on the fly.
    # Computing on the fly is: popcount(adj[v] & ~visited)
    
    def popcount(x):
        return bin(x).count('1')
        
    def dfs(current, visited, path):
        if visited == full_mask:
            # Check if current connects back to start
            if adj[current] & (1 << start):
                return path[:]
            return None
        
        # Get unvisited neighbors
        neighbors_mask = adj[current] & ~visited
        if neighbors_mask == 0:
            return None
            
        # Heuristic: sort neighbors by number of unvisited neighbors (ascending)
        # This is a "fail fast" strategy
        neighbors = []
        temp_mask = neighbors_mask
        while temp_mask:
            # Get lowest set bit
            lsb = temp_mask & -temp_mask
            v = lsb.bit_length() - 1
            # Count unvisited neighbors of v
            v_neighbors = adj[v] & ~visited
            deg = popcount(v_neighbors)
            neighbors.append((deg, v))
            temp_mask ^= lsb
            
        neighbors.sort()
        
        for deg, v in neighbors:
            new_visited = visited | (1 << v)
            # Additional pruning: if any unvisited node has no connection to current or future?
            # Simple check: if v is not the last one, it must have at least one unvisited neighbor 
            # OR be able to connect to start if it's the last one.
            # But the sort heuristic usually handles this.
            # Let's add a strong pruning: 
            # If we pick v, and v is not the final node, v must have at least one neighbor 
            # in the remaining unvisited set (excluding v itself, which is now visited).
            # Wait, the "unvisited" set in the next step is `new_visited` complement.
            # So v's neighbors in the *next* step are `adj[v] & ~new_visited`.
            # If this is 0, and v is not the last node (i.e., new_visited != full_mask), then we are stuck.
            if new_visited != full_mask:
                if (adj[v] & ~new_visited) == 0:
                    continue
                    
            path.append(v)
            result = dfs(v, new_visited, path)
            if result is not None:
                return result
            path.pop()
            
        return None

    path = [start]
    result = dfs(start, 1 << start, path)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Binary search for the minimum threshold T
    lo = N  # Minimum edges needed is N
    hi = TOTAL_EDGES
    best_path = None
    best_T = hi
    
    # First, ensure we have a valid cycle at all (should be true for hi)
    # We can just start the binary search.
    
    while lo <= hi:
        if time.time() > deadline:
            break
            
        mid = (lo + hi) // 2
        
        adj = build_adjacency(mid)
        path = has_hamiltonian_cycle(adj)
        
        if path is not None:
            best_path = path
            best_T = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_path is None:
        # Fallback: return the trivial cycle if something went wrong (should not happen)
        return list(range(N))
        
    return best_path
# EVOLVE-BLOCK-END
