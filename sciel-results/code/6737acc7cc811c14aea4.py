"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency(edges):
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _check_feasibility(adj, remaining_edges, n):
    """Check if each vertex still has degree >= 2 in remaining edges."""
    for v in range(n):
        deg = sum(1 for u in adj[v] if _edge_key(v, u) in remaining_edges)
        if deg < 2:
            return False
    return True

def _find_hamiltonian_cycles(adj, available_edges, max_cycles=10, deadline=None):
    """Find up to max_cycles edge-disjoint Hamiltonian cycles using backtracking with MRV."""
    n = P['n']
    cycles = []
    
    def backtrack(cycles_found, remaining_edges):
        if deadline is not None and time.time() > deadline:
            return False
        if len(cycles_found) >= max_cycles:
            return True
        
        # Check feasibility
        if not _check_feasibility(adj, remaining_edges, n):
            return False
        
        # Try to find one more Hamiltonian cycle using only remaining_edges
        # Start from vertex 0 to avoid symmetry
        def dfs(path, visited, used_edges, depth):
            if deadline is not None and time.time() > deadline:
                return None
            if len(path) == n:
                # Check if last vertex connects back to first
                first = path[0]
                last = path[-1]
                ek = _edge_key(first, last)
                if ek in remaining_edges and ek not in used_edges:
                    return path[:]
                return None
            
            last = path[-1]
            # MRV: sort neighbors by number of available edges (fewest first)
            neighbors = []
            for neighbor in adj[last]:
                if not visited[neighbor]:
                    ek = _edge_key(last, neighbor)
                    if ek in remaining_edges and ek not in used_edges:
                        # Count available edges from neighbor (for MRV)
                        avail_count = sum(1 for nb in adj[neighbor] 
                                         if not visited[nb] and _edge_key(neighbor, nb) in remaining_edges 
                                         and _edge_key(neighbor, nb) not in used_edges)
                        neighbors.append((avail_count, neighbor))
            
            neighbors.sort()  # Sort by fewest available options first
            
            for _, neighbor in neighbors:
                ek = _edge_key(last, neighbor)
                if ek not in remaining_edges or ek in used_edges:
                    continue
                visited[neighbor] = True
                used_edges.add(ek)
                result = dfs(path + [neighbor], visited, used_edges, depth + 1)
                if result is not None:
                    return result
                visited[neighbor] = False
                used_edges.remove(ek)
            return None
        
        visited = [False] * n
        visited[0] = True
        result = dfs([0], visited, set(), 0)
        if result is not None:
            cycles_found.append(result)
            # Remove edges used in this cycle from remaining
            for i in range(len(result)):
                u = result[i]
                v = result[(i + 1) % len(result)]
                ek = _edge_key(u, v)
                remaining_edges.discard(ek)
            return backtrack(cycles_found, remaining_edges)
        return False
    
    rem = set(available_edges)
    if backtrack(cycles, rem):
        return cycles
    return cycles

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 second safety margin
    
    adj = _build_adjacency(P['edges'])
    available = set(_edge_key(u, v) for u, v in P['edges'])
    
    # Try to find as many edge-disjoint Hamiltonian cycles as possible
    cycles = _find_hamiltonian_cycles(adj, available, max_cycles=10, deadline=deadline)
    
    if len(cycles) >= 1:
        return [list(c) for c in cycles]
    else:
        # Fallback: return the known Hamiltonian cycle
        return [list(range(P['n']))]
# EVOLVE-BLOCK-END
