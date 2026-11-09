"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

import sys
sys.setrecursionlimit(10000)

def construct():
    n = P['n']
    # Build adjacency and edge set
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Edge key function
    def edge_key(u, v):
        return (min(u, v), max(u, v))
    
    all_edges = set()
    for u, v in P['edges']:
        all_edges.add(edge_key(u, v))
    
    # Find multiple edge-disjoint Hamiltonian cycles
    used_edges = set()
    cycles = []
    
    def find_ham_cycle(used_edges_set):
        """Find a Hamiltonian cycle avoiding edges in used_edges_set."""
        # Try to find a Hamiltonian cycle using backtracking
        # Start from vertex 0 for determinism
        start = 0
        path = [start]
        visited = [False] * n
        visited[start] = True
        
        def backtrack():
            if len(path) == n:
                # Check if we can close the cycle
                last = path[-1]
                if edge_key(last, start) not in used_edges_set and start in adj[last]:
                    return True
                return False
            
            last = path[-1]
            # Try neighbors in a consistent order
            for nxt in sorted(adj[last]):
                if not visited[nxt] and edge_key(last, nxt) not in used_edges_set:
                    # Pruning: if nxt is not start and we haven't visited all, continue
                    # If nxt is start, only allow if we've visited all
                    if nxt == start and len(path) < n - 1:
                        continue
                    if nxt == start and len(path) == n - 1:
                        # Check if closing edge is available
                        if edge_key(last, start) in used_edges_set:
                            continue
                        # This would be the last step
                        path.append(nxt)
                        visited[nxt] = True
                        if len(path) == n:
                            # Check closing
                            if edge_key(path[-1], start) not in used_edges_set:
                                return True
                            path.pop()
                            visited[nxt] = False
                        else:
                            if backtrack():
                                return True
                            path.pop()
                            visited[nxt] = False
                    else:
                        path.append(nxt)
                        visited[nxt] = True
                        if backtrack():
                            return True
                        path.pop()
                        visited[nxt] = False
            
            return False
        
        if backtrack():
            # Extract cycle: path[0..n-1], closing edge path[n-1] -> path[0]
            cycle = path[:]
            return cycle
        return None
    
    # Find up to 10 edge-disjoint Hamiltonian cycles
    max_cycles = 10
    for _ in range(max_cycles):
        cycle = find_ham_cycle(used_edges)
        if cycle is None:
            break
        # Mark edges as used
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            used_edges.add(edge_key(u, v))
        cycles.append(cycle)
    
    if not cycles:
        # Fallback: return a single Hamiltonian cycle if we found one before
        # This shouldn't happen, but just in case
        return [list(range(n))]
    
    return cycles
# EVOLVE-BLOCK-END
