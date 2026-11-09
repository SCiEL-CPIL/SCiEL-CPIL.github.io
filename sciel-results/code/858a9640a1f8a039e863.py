"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
sys.setrecursionlimit(10000)

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle(adj, n, deadline):
    """Find a Hamiltonian cycle using backtracking with enhanced pruning."""
    start = 0
    path = [start]
    visited = [False] * n
    visited[start] = True
    remaining = n - 1
    
    def _is_connected(subset, adj, n):
        """Check if vertices in subset form a connected subgraph."""
        if len(subset) <= 1:
            return True
        # BFS/DFS from first vertex
        first = next(iter(subset))
        seen = {first}
        stack = [first]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v in subset and v not in seen:
                    seen.add(v)
                    stack.append(v)
        return len(seen) == len(subset)
    
    def backtrack():
        nonlocal remaining
        if time.time() > deadline:
            return None
        if len(path) == n:
            if path[-1] in adj[0]:
                return list(path)
            return None
        cur = path[-1]
        # Sort neighbors: prefer those with fewer unvisited neighbors (most constrained first)
        unvisited_neighbors = [nb for nb in adj[cur] if not visited[nb]]
        if not unvisited_neighbors:
            return None
        unvisited_neighbors.sort(key=lambda x: sum(1 for y in adj[x] if not visited[y]))
        
        for nb in unvisited_neighbors:
            # Pruning 1: nb must be able to connect back (to start or to some unvisited vertex)
            can_reach_end = False
            for nb2 in adj[nb]:
                if not visited[nb2] or nb2 == start:
                    can_reach_end = True
                    break
            if not can_reach_end:
                continue
            
            # Pruning 2: After visiting nb, check if remaining unvisited vertices are connected
            # (only do this check when few remain to avoid overhead)
            if remaining <= 6:
                unvisited_set = set()
                for i in range(n):
                    if not visited[i] and i != nb:
                        unvisited_set.add(i)
                if unvisited_set and not _is_connected(unvisited_set, adj, n):
                    continue
            
            visited[nb] = True
            path.append(nb)
            remaining -= 1
            result = backtrack()
            if result is not None:
                return result
            path.pop()
            visited[nb] = False
            remaining += 1
        return None
    
    return backtrack()

def construct():
    deadline = time.time() + 50  # 50 second safety margin
    
    n = P['n']
    all_edges = set()
    for u, v in P['edges']:
        all_edges.add(_edge_key(u, v))
    
    remaining_edges = set(all_edges)
    cycles = []
    
    for _ in range(3):  # Max possible is floor(70/21)=3
        if time.time() > deadline:
            break
        # Build adjacency from remaining edges
        adj = _build_adj([(u, v) for u, v in P['edges'] if _edge_key(u, v) in remaining_edges], n)
        
        # Check if graph is still 2-regular enough (each vertex needs degree >= 2)
        valid = True
        for i in range(n):
            if len(adj[i]) < 2:
                valid = False
                break
        if not valid:
            break
        
        cycle = _find_ham_cycle(adj, n, deadline)
        if cycle is None:
            break
        
        # Verify cycle is valid
        valid_cycle = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            if _edge_key(u, v) not in remaining_edges:
                valid_cycle = False
                break
        if not valid_cycle:
            break
        
        # Remove edges used by this cycle
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            remaining_edges.discard(_edge_key(u, v))
        
        cycles.append(cycle)
    
    # Ensure non-empty
    if not cycles:
        # Fallback: return the trivial cycle if it's valid
        trivial = list(range(n))
        valid_trivial = True
        for i in range(n):
            u = trivial[i]
            v = trivial[(i + 1) % n]
            if _edge_key(u, v) not in all_edges:
                valid_trivial = False
                break
        if valid_trivial:
            return [trivial]
        # Last resort: find any cycle
        return [list(range(n))]
    
    # Convert to ensure pure Python int types
    result = []
    for cyc in cycles:
        result.append([int(v) for v in cyc])
    return result
# EVOLVE-BLOCK-END
