import time
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def edge_key(u, v):
        return (min(u, v), max(u, v))
    
    def _find_one_hc(adj_list, used_edges, n, deadline, start=0):
        """Find a single Hamiltonian cycle using available edges with iterative deepening."""
        path = [start]
        visited = {start}
        
        def backtrack():
            if time.time() > deadline:
                return None
            
            if len(path) == n:
                last = path[-1]
                ek = edge_key(last, start)
                if ek not in used_edges and start in adj_list[last]:
                    return list(path)
                return None
            
            last = path[-1]
            
            # Get candidates: unvisited neighbors with available edges
            candidates = []
            for nb in adj_list[last]:
                if nb not in visited:
                    ek = edge_key(last, nb)
                    if ek not in used_edges:
                        # Count available edges from nb to unvisited vertices
                        avail = sum(1 for x in adj_list[nb] if x not in visited and edge_key(nb, x) not in used_edges)
                        candidates.append((avail, nb))
            
            # Sort by fewest available edges first (constraint propagation)
            candidates.sort()
            
            for _, nb in candidates:
                path.append(nb)
                visited.add(nb)
                result = backtrack()
                if result is not None:
                    return result
                path.pop()
                visited.remove(nb)
            
            return None
        
        return backtrack()
    
    # Main search
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    best_result = []
    random.seed(42)
    
    # Try to find up to 3 edge-disjoint Hamiltonian cycles
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        used_edges = set()
        cycles = []
        
        # Create shuffled adjacency for diversity
        shuffled_adj = [list(s) for s in adj]
        for lst in shuffled_adj:
            random.shuffle(lst)
        
        # Try to find up to 3 Hamiltonian cycles
        for _ in range(3):
            if time.time() > deadline:
                break
            
            cycle = _find_one_hc(shuffled_adj, used_edges, n, deadline)
            if cycle is None:
                break
            
            cycles.append(cycle)
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                used_edges.add(edge_key(u, v))
        
        if len(cycles) > len(best_result):
            best_result = cycles
            if len(best_result) >= 3:
                break
    
    # Ensure types are correct
    result = []
    for cycle in best_result:
        result.append([int(v) for v in cycle])
    
    # Fallback: if we couldn't find any, try once more with fixed order
    if not result:
        fixed_adj = [list(s) for s in adj]
        for lst in fixed_adj:
            lst.sort()
        used_edges = set()
        cycle = _find_one_hc(fixed_adj, used_edges, n, time.time() + 5)
        if cycle:
            result = [[int(v) for v in cycle]]
    
    return result
# EVOLVE-BLOCK-END
