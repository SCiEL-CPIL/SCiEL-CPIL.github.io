"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']

# Build edge index lookup: edge (u,v) -> one-based arrival index
edge_idx = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    if u > v:
        u, v = v, u
    edge_idx[(u, v)] = i + 1

def get_edge_index(u, v):
    if u > v:
        u, v = v, u
    return edge_idx.get((u, v), float('inf'))

def build_adjacency(k):
    """Build adjacency list using only edges with index <= k."""
    adj = [set() for _ in range(N)]
    for i in range(k):
        u, v = EDGE_ORDER[i]
        adj[u].add(v)
        adj[v].add(u)
    return adj

def has_hamiltonian_cycle(k, start_vertex=0, deadline=None):
    """Check if there's a Hamiltonian cycle using only edges with index <= k.
    Returns the cycle as a list of vertices, or None."""
    adj = build_adjacency(k)
    
    # Quick check: every vertex must have degree >= 2
    for v in range(N):
        if len(adj[v]) < 2:
            return None
    
    # Check connectivity
    visited = [False] * N
    stack = [start_vertex]
    visited[start_vertex] = True
    count = 0
    while stack:
        u = stack.pop()
        count += 1
        for w in adj[u]:
            if not visited[w]:
                visited[w] = True
                stack.append(w)
    if count < N:
        return None
    
    # DFS with backtracking
    # Fix start_vertex as the first vertex to break rotational symmetry
    path = [start_vertex]
    visited_mask = 1 << start_vertex
    
    # Sort neighbors for determinism and pruning
    adj_sorted = [sorted(adj[v]) for v in range(N)]
    
    # Additional pruning: for each unvisited vertex, it must have at least 2 neighbors
    # in the subgraph induced by {unvisited} ∪ {last_vertex} ∪ {start_vertex}
    # This is complex; let's use a simpler but effective approach
    
    def dfs(current, path, visited_mask):
        if deadline and time.time() > deadline:
            return None
        
        if len(path) == N:
            # Check if we can close the cycle: edge from current to start_vertex
            if start_vertex in adj[current]:
                return path[:]
            return None
        
        remaining = N - len(path)
        
        # Pruning: for each unvisited vertex, check if it has enough connections
        # to be part of a cycle. At minimum, it needs 2 neighbors in 
        # (unvisited ∪ {current} ∪ {start_vertex if remaining==1})
        for v in range(N):
            if not (visited_mask & (1 << v)):
                # v is unvisited
                # v needs to connect to at least 2 vertices in the eventual cycle
                # The cycle will include: unvisited vertices + current + start_vertex
                # For now, just check degree in the subgraph
                deg = 0
                for w in adj[v]:
                    if (visited_mask & (1 << w)) or w == current or w == start_vertex:
                        deg += 1
                # v needs at least 2 edges in the final cycle
                # One edge will connect to a visited vertex (current or earlier)
                # and one to another vertex
                # Simple check: v must have at least 2 neighbors total in adj
                # This is already checked. Let's do a stronger check:
                # v must have at least 1 neighbor that is unvisited or is current
                # (since one edge goes to a previously visited vertex in the path)
                has_unvisited_neighbor = False
                has_current_or_start = False
                for w in adj[v]:
                    if not (visited_mask & (1 << w)):
                        has_unvisited_neighbor = True
                    if w == current or w == start_vertex:
                        has_current_or_start = True
                if not has_unvisited_neighbor and not has_current_or_start:
                    return None
                # Also: if remaining == 1, v must connect to both current and start
                if remaining == 1:
                    if current not in adj[v] or start_vertex not in adj[v]:
                        return None
        
        # Try extending the path
        for next_v in adj_sorted[current]:
            if not (visited_mask & (1 << next_v)):
                result = dfs(next_v, path + [next_v], visited_mask | (1 << next_v))
                if result is not None:
                    return result
        
        return None
    
    return dfs(start_vertex, path, visited_mask)

def find_minimal_k(deadline=None):
    """Binary search for the minimal K such that a Hamiltonian cycle exists using edges 1..K."""
    # Lower bound: need at least N edges (the cycle itself)
    # Upper bound: total number of edges
    total_edges = len(EDGE_ORDER)
    
    # First, find an upper bound by trying increasing K
    # The cycle needs N edges, so K >= N is a trivial lower bound
    # But we need the specific edges to form a cycle
    
    lo = N  # at least N edges needed
    hi = total_edges
    
    # First verify hi works
    cycle = has_hamiltonian_cycle(hi, deadline=deadline)
    if cycle is None:
        # Shouldn't happen with all edges, but handle gracefully
        return total_edges, list(range(N))
    
    # Binary search
    best_k = hi
    best_cycle = cycle
    
    while lo <= hi:
        if deadline and time.time() > deadline:
            break
        mid = (lo + hi) // 2
        cycle = has_hamiltonian_cycle(mid, deadline=deadline)
        if cycle is not None:
            best_k = mid
            best_cycle = cycle
            hi = mid - 1
        else:
            lo = mid + 1
    
    return best_k, best_cycle

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin
    
    # Try to find the minimal K
    try:
        best_k, best_cycle = find_minimal_k(deadline=deadline)
        if best_cycle is not None and len(best_cycle) == N:
            return best_cycle
    except Exception:
        pass
    
    # Fallback: return the identity permutation
    return list(range(N))

# EVOLVE-BLOCK-END
