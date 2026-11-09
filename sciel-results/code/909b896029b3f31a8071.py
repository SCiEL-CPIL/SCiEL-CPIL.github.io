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
    """Find a Hamiltonian cycle using backtracking with pruning."""
    # Use vertex 0 as start to avoid symmetry
    start = 0
    path = [start]
    visited = [False] * n
    visited[start] = True
    
    # Precompute unvisited count for pruning
    unvisited_count = [0] * n
    
    def backtrack():
        if time.time() > deadline:
            return None
        if len(path) == n:
            if path[-1] in adj[0]:
                return list(path)
            return None
        cur = path[-1]
        # Sort neighbors by degree for better pruning (minimum degree first)
        neighbors = sorted(adj[cur], key=lambda x: len(adj[x]))
        for nb in neighbors:
            if not visited[nb]:
                # Pruning: check if nb has at least one unvisited neighbor or is connected to 0
                has_path_back = False
                for nb2 in adj[nb]:
                    if not visited[nb2] or nb2 == start:
                        has_path_back = True
                        break
                if not has_path_back:
                    continue
                
                visited[nb] = True
                path.append(nb)
                result = backtrack()
                if result is not None:
                    return result
                path.pop()
                visited[nb] = False
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
    
    return cycles
# EVOLVE-BLOCK-END
