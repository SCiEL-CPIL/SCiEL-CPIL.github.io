"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
TOTAL_EDGES = len(EDGE_ORDER)

# Precompute: for each edge index, which vertices it connects
# Build adjacency for a given prefix size

def build_adjacency(k):
    """Build adjacency list from first k edges (1-based index, so edges 1..k)"""
    adj = defaultdict(set)
    for i in range(k):
        u, v = EDGE_ORDER[i]
        adj[u].add(v)
        adj[v].add(u)
    return adj

def has_hamiltonian_cycle(k):
    """Check if the graph formed by first k edges has a Hamiltonian cycle."""
    adj = build_adjacency(k)
    
    # Quick check: every vertex must have degree >= 2
    for v in range(N):
        if len(adj[v]) < 2:
            return False, None
    
    # Use backtracking with pruning
    # Fix vertex 0 as the start to break symmetry
    # We build a path starting from 0, and need to close it back to 0
    
    # degrees_in_cycle: track how many cycle edges each vertex has so far
    # We'll use a recursive backtracking
    
    # To make it efficient, we order vertices by degree (most constrained first)
    # But we need to start at 0
    
    # Let's use a different approach: DP with bitmask
    # State: (current_vertex, visited_mask, prev_vertex) -> but that's too large
    # Actually for 24 vertices, 2^24 * 24 * 24 is too large
    
    # Let's use backtracking with strong pruning
    
    import sys
    sys.setrecursionlimit(10000)
    
    # We'll build the cycle edge by edge
    # cycle_edges: list of edges in the cycle so far
    # deg: degree of each vertex in the cycle so far
    # path: the current path (list of vertices)
    
    best_cycle = [None]
    
    def backtrack(path, deg, used_edges):
        if time.time() > deadline:
            return False
        
        if len(path) == N:
            # Check if we can close the cycle: last vertex must connect to first
            last = path[-1]
            first = path[0]
            if first in adj[last]:
                # Check that first and last don't already have degree 2
                if deg[first] < 2 and deg[last] < 2:
                    best_cycle[0] = list(path)
                    return True
            return False
        
        current = path[-1]
        
        # Try all neighbors of current that haven't been used yet
        # Sort by degree for better pruning (most constrained first)
        candidates = list(adj[current])
        candidates.sort(key=lambda x: len(adj[x]))
        
        for next_v in candidates:
            if next_v in path:
                continue
            if deg[next_v] >= 2:
                continue
            if deg[current] >= 2:
                continue
            
            # Pruning: check that all unvisited vertices still have potential
            # to reach degree 2
            new_deg = deg.copy()
            new_deg[current] += 1
            new_deg[next_v] += 1
            
            # Check feasibility
            feasible = True
            for v in range(N):
                if v not in path and v != next_v:
                    if len(adj[v]) < 2:
                        feasible = False
                        break
                    # Count how many of v's neighbors are unvisited or in path
                    # v needs 2 more edges, so it needs at least 2 neighbors
                    # that are either unvisited or the current endpoint
                    available = 0
                    for nb in adj[v]:
                        if nb not in path or nb == current:
                            available += 1
                    if available < (2 - new_deg[v]):
                        feasible = False
                        break
            
            if not feasible:
                continue
            
            path.append(next_v)
            if backtrack(path, new_deg, used_edges):
                return True
            path.pop()
        
        return False
    
    deadline = time.time() + 5  # 5 second timeout for each check
    
    path = [0]
    deg = {v: 0 for v in range(N)}
    deg[0] = 0  # Will be incremented when we add an edge
    
    # Actually, let me reconsider the degree tracking
    # When we add an edge (u,v), both u and v get +1 to their degree
    # At the end, each vertex should have degree exactly 2
    
    # Let me redo this more carefully
    # path = [v0, v1, ..., vk]
    # The edges in the path are (v0,v1), (v1,v2), ..., (vk-1,vk)
    # Vertex v0 has degree 1 (from edge v0-v1)
    # Vertex vk has degree 1 (from edge vk-1-vk)
    # Vertices v1..vk-1 have degree 2
    # To close: need edge (vk, v0), giving v0 and vk degree 2
    
    # So in the path, internal vertices have degree 2, endpoints have degree 1
    # When extending, the current endpoint gets +1 (becomes internal or stays endpoint)
    
    # Let me simplify: just track which vertices are in the path
    # and use the path structure to determine degrees
    
    def backtrack2(path):
        if time.time() > deadline:
            return False
        
        if len(path) == N:
            last = path[-1]
            first = path[0]
            if first in adj[last]:
                best_cycle[0] = list(path)
                return True
            return False
        
        current = path[-1]
        path_set = set(path)
        
        # Try neighbors
        candidates = list(adj[current])
        # Sort by: prefer unvisited, then by degree
        candidates.sort(key=lambda x: (x in path_set, len(adj[x])))
        
        for next_v in candidates:
            if next_v in path_set:
                continue
            
            # Check: next_v must be able to reach degree 2
            # It currently has 0 cycle edges. After adding (current, next_v), it has 1.
            # It needs 1 more. So it needs at least 1 other neighbor that is not in path
            # (or is the start vertex if we're at the end, but we're not)
            other_neighbors = [nb for nb in adj[next_v] if nb not in path_set and nb != current]
            if len(other_neighbors) < 1:
                continue
            
            # Check: current vertex after adding this edge has degree 2 (if len(path)>=2)
            # or degree 1 (if len(path)==1, making it the first edge)
            # Actually current is the last in path, so it currently has degree 1 (except if path len==1)
            # After adding, if len(path) >= 2, current becomes degree 2 (internal)
            # If len(path) == 1, current becomes degree 1 (still endpoint)
            
            # Pruning: check all unvisited vertices can still reach degree 2
            feasible = True
            new_path_set = path_set | {next_v}
            for v in range(N):
                if v in new_path_set:
                    continue
                # v needs 2 edges. How many neighbors can it use?
                # Neighbors in path: can connect to at most 2 (the endpoints of the path, 
                # but actually any vertex in path... no, in a simple path, internal vertices 
                # already have degree 2, so they can't take more edges)
                # Wait, in our construction, internal vertices in the path have degree 2,
                # so they can't accept more edges. Only the two endpoints can.
                # So v can connect to: 
                #   - unvisited neighbors (which will be in the path later)
                #   - the two endpoints of the current path (if they are neighbors)
                
                endpoints = {path[0], path[-1]}
                # After adding next_v, the new path is path + [next_v]
                # New endpoints: path[0] and next_v
                new_endpoints = {path[0], next_v}
                
                available_from_path = sum(1 for nb in adj[v] if nb in new_endpoints)
                available_unvisited = sum(1 for nb in adj[v] if nb not in new_path_set)
                
                total_available = available_from_path + available_unvisited
                if total_available < 2:
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            path.append(next_v)
            if backtrack2(path):
                return True
            path.pop()
        
        return False
    
    path = [0]
    if backtrack2(path):
        return True, best_cycle[0]
    return False, None


def construct():
    start_time = time.time()
    deadline = start_time + 75  # Leave 15s margin
    
    # Binary search on k (prefix size)
    # Lower bound: need at least N edges for a cycle, so k >= N = 24
    # Upper bound: TOTAL_EDGES
    lo = N
    hi = TOTAL_EDGES
    best_k = None
    best_cycle = None
    
    # First, check if the full graph has a Hamiltonian cycle
    # It should, since it's a complete graph
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        found, cycle = has_hamiltonian_cycle(mid)
        
        if found:
            best_k = mid
            best_cycle = cycle
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_cycle is not None:
        return best_cycle
    else:
        # Fallback: return the identity permutation (score = max index of edges in cycle 0-1-2-...-23-0)
        return list(range(N))

# EVOLVE-BLOCK-END
