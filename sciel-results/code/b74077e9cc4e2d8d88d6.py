"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
sys.setrecursionlimit(10000)

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle(adj, n, deadline):
    """Find a Hamiltonian cycle using backtracking with pruning."""
    start = 0
    path = [start]
    visited = [False] * n
    visited[start] = True
    
    # Precompute neighbor lists sorted by degree
    sorted_neighbors = [sorted(adj[i], key=lambda x: len(adj[x])) for i in range(n)]
    
    def _connected_check(unvisited_set, from_vertex, to_vertex):
        """Check if there's a path from from_vertex to to_vertex using only unvisited_set vertices."""
        if from_vertex == to_vertex:
            return True
        if not unvisited_set:
            return False
        # BFS from from_vertex
        if from_vertex not in unvisited_set and from_vertex != to_vertex:
            # from_vertex is the current end, so we need to reach to_vertex via unvisited
            pass
        visited_bfs = set()
        queue = [from_vertex]
        visited_bfs.add(from_vertex)
        while queue:
            node = queue.pop(0)
            for nb in adj[node]:
                if nb == to_vertex:
                    return True
                if nb not in visited_bfs and nb in unvisited_set:
                    visited_bfs.add(nb)
                    queue.append(nb)
        return False
    
    def backtrack():
        if time.time() > deadline:
            return None
        if len(path) == n:
            if path[-1] in adj[0]:
                return list(path)
            return None
        cur = path[-1]
        # Sort neighbors by degree for better pruning
        for nb in sorted_neighbors[cur]:
            if not visited[nb]:
                # Pruning: if this is the second-to-last vertex, check connectivity
                if len(path) == n - 2:
                    # After visiting nb, only one unvisited vertex remains
                    # Check if that vertex connects to both nb and start
                    remaining = [v for v in range(n) if not visited[v] and v != nb]
                    if len(remaining) == 1:
                        r = remaining[0]
                        if r not in adj[nb] or r not in adj[0]:
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
    deadline = time.time() + 55  # 55 second safety margin (90s kill - 35s buffer)
    
    n = P['n']
    all_edges = set()
    for u, v in P['edges']:
        all_edges.add(frozenset((u, v)))
    
    remaining_edges = set(all_edges)
    cycles = []
    
    for _ in range(3):  # Max possible is floor(70/21)=3
        if time.time() > deadline:
            break
        # Build adjacency from remaining edges
        adj = _build_adj([(u, v) for u, v in P['edges'] if frozenset((u, v)) in remaining_edges], n)
        
        # Check if graph is still 2-regular enough (each vertex needs degree >= 2)
        valid = True
        for i in range(n):
            if len(adj[i]) < 2:
                valid = False
                break
        if not valid:
            break
        
        # Try multiple starting vertices to improve reliability
        cycle = None
        for start_attempt in range(n):
            if time.time() > deadline:
                break
            # Temporarily modify to try different start
            cycle = _find_ham_cycle(adj, n, deadline)
            if cycle is not None:
                break
        
        if cycle is None:
            break
        
        # Verify cycle is valid
        valid_cycle = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            if frozenset((u, v)) not in remaining_edges:
                valid_cycle = False
                break
        if not valid_cycle:
            break
        
        # Remove edges used by this cycle
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            remaining_edges.discard(frozenset((u, v)))
        
        cycles.append(cycle)
    
    # Ensure non-empty
    if not cycles:
        # Fallback: try to find at least one cycle
        adj_full = _build_adj(P['edges'], n)
        cycle = _find_ham_cycle(adj_full, n, deadline)
        if cycle is not None:
            return [cycle]
        # Last resort
        return [list(range(n))]
    
    return cycles
# EVOLVE-BLOCK-END
