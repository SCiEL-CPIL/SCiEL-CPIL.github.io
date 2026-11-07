"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency with edge IDs
    adj = {i: [] for i in range(n)}
    edge_set = set()
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
        edge_set.add((min(u,v), max(u,v)))
    
    total_edges = len(edges)
    deadline = time.time() + 80  # 80s safety margin
    
    def find_ham_cycle(used_edges, start=0):
        """Find a Hamiltonian cycle avoiding used_edges, starting from 'start'."""
        # Backtracking DFS for Hamiltonian cycle
        visited = [False] * n
        visited[start] = True
        path = [start]
        path_edges = []
        
        def dfs(current):
            if time.time() > deadline:
                return None
            if len(path) == n:
                # Check if edge back to start exists and is unused
                for (nb, eid) in adj[current]:
                    if nb == start and eid not in used_edges:
                        return path + [start], path_edges + [eid]
                return None
            
            # Try neighbors in some order
            for (nb, eid) in adj[current]:
                if nb == start:
                    continue  # Only close at the end
                if not visited[nb] and eid not in used_edges:
                    visited[nb] = True
                    path.append(nb)
                    path_edges.append(eid)
                    
                    result = dfs(nb)
                    if result is not None:
                        return result
                    
                    path.pop()
                    path_edges.pop()
                    visited[nb] = False
            
            return None
        
        return dfs(start)
    
    # Find edge-disjoint Hamiltonian cycles
    cycles = []
    used_edges = set()
    
    # Try to find up to 3 cycles (max possible = 70//21 = 3)
    for cycle_num in range(3):
        if time.time() > deadline:
            break
        
        # Try different starting vertices to increase chances
        found = None
        for start in range(n):
            if time.time() > deadline:
                break
            result = find_ham_cycle(used_edges, start)
            if result is not None:
                found = result
                break
        
        if found is None:
            break
        
        cycle_path, cycle_eids = found
        # Remove the closing vertex (it's a duplicate of start)
        cycle = cycle_path[:-1]
        cycles.append(cycle)
        used_edges.update(cycle_eids)
    
    # Return at least one cycle (the seed's answer) if we found none better
    if not cycles:
        return [list(range(n))]
    
    return cycles
# EVOLVE-BLOCK-END
