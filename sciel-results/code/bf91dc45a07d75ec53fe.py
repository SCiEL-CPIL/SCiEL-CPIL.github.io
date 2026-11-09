"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import itertools
from functools import lru_cache

N = 24

# Generate the pinned edge list with random.Random(614)
def _build_edge_order():
    rng = random.Random(614)
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    rng.shuffle(edges)
    return edges

EDGE_ORDER = _build_edge_order()
EDGE_INDEX = {}
for idx, (u, v) in enumerate(EDGE_ORDER, 1):
    EDGE_INDEX[(u, v)] = idx
    EDGE_INDEX[(v, u)] = idx

def _edge_score(u, v):
    return EDGE_INDEX[(u, v)]

def construct():
    # We want to find a Hamiltonian cycle that minimizes the maximum edge arrival index.
    # This is equivalent to finding the smallest T such that there exists a Hamiltonian cycle
    # using only edges with arrival index <= T.
    
    # Strategy: 
    # 1. Try to find a Hamiltonian cycle in the subgraph of edges with index <= T for increasing T.
    # 2. Start with a greedy/heuristic approach to get a good upper bound on T.
    # 3. Then use backtracking or DFS to verify if a smaller T is possible.
    
    # First, let's get an initial upper bound using a greedy nearest-neighbor-like approach
    # or by constructing a cycle from early edges.
    
    # Let's try a simple heuristic: start from vertex 0, always go to the unvisited neighbor
    # with the smallest edge index, ensuring we can return to 0 at the end.
    
    def greedy_cycle():
        visited = [False] * N
        cycle = [0]
        visited[0] = True
        
        for _ in range(N - 1):
            last = cycle[-1]
            best_next = -1
            best_idx = float('inf')
            for v in range(N):
                if not visited[v]:
                    idx = _edge_score(last, v)
                    if idx < best_idx:
                        best_idx = idx
                        best_next = v
            
            if best_next == -1:
                return None
            
            # Check if we can still complete the cycle
            # For simplicity, just pick the best and hope for the best
            cycle.append(best_next)
            visited[best_next] = True
        
        # Check if we can close the cycle
        if _edge_score(cycle[-1], cycle[0]) < float('inf'):
            max_idx = max(_edge_score(cycle[i], cycle[i+1]) for i in range(N-1))
            max_idx = max(max_idx, _edge_score(cycle[-1], cycle[0]))
            return (cycle, max_idx)
        else:
            return None
    
    # Get initial upper bound
    result = greedy_cycle()
    if result:
        best_cycle, best_max_idx = result
    else:
        # Fallback: use the identity permutation
        best_cycle = list(range(N))
        best_max_idx = max(_edge_score(best_cycle[i], best_cycle[i+1]) for i in range(N-1))
        best_max_idx = max(best_max_idx, _edge_score(best_cycle[-1], best_cycle[0]))
    
    # Now try to improve by checking if we can find a cycle with a smaller max index
    # We'll use a DFS-based approach with pruning
    
    import sys
    sys.setrecursionlimit(10000)
    
    def find_cycle_with_max_idx(max_allowed):
        """Find a Hamiltonian cycle using only edges with index <= max_allowed."""
        # Build adjacency list for edges with index <= max_allowed
        adj = [[] for _ in range(N)]
        for u in range(N):
            for v in range(u + 1, N):
                if _edge_score(u, v) <= max_allowed:
                    adj[u].append(v)
                    adj[v].append(u)
        
        # Check if each vertex has degree >= 2 (necessary for Hamiltonian cycle)
        for u in range(N):
            if len(adj[u]) < 2:
                return None
        
        # DFS with backtracking
        visited = [False] * N
        path = [0]
        visited[0] = True
        
        def dfs():
            if len(path) == N:
                # Check if we can close the cycle
                if 0 in adj[path[-1]]:
                    return True
                return False
            
            last = path[-1]
            for v in adj[last]:
                if not visited[v]:
                    # Pruning: if v is not 0 and we've visited more than 1 vertex,
                    # we can't go to 0 until the end
                    if v == 0 and len(path) < N:
                        continue
                    
                    visited[v] = True
                    path.append(v)
                    
                    if dfs():
                        return True
                    
                    path.pop()
                    visited[v] = False
            
            return False
        
        if dfs():
            return path[:]
        return None
    
    # Binary search on the maximum index
    # Lower bound: at least N (each vertex needs at least 2 edges, so we need at least N edges in the cycle)
    # Actually, the minimum possible max index is at least N since we need N distinct edges
    lo = N  # minimum possible
    hi = best_max_idx
    
    while lo < hi:
        mid = (lo + hi) // 2
        cycle = find_cycle_with_max_idx(mid)
        if cycle is not None:
            hi = mid
            best_cycle = cycle
            best_max_idx = mid
        else:
            lo = mid + 1
    
    return best_cycle

# EVOLVE-BLOCK-END
