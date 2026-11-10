"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_hamiltonian_cycle(adj, used_edges, start, deadline):
    """Find a Hamiltonian cycle starting at 'start' avoiding used_edges."""
    n = P['n']
    # used_edges is a set of frozensets or sorted tuples
    def edge_used(u, v):
        return (min(u,v), max(u,v)) in used_edges
    
    # DFS with backtracking
    path = [start]
    visited = {start}
    best_cycle = None
    
    def dfs(current, depth):
        nonlocal best_cycle
        if time.time() > deadline:
            return False
        if depth == n - 1:
            # Check if current connects back to start
            if not edge_used(current, start) and start in adj[current]:
                best_cycle = list(path)
                return True
            return False
        
        for neighbor in sorted(adj[current]):
            if neighbor not in visited and not edge_used(current, neighbor):
                visited.add(neighbor)
                path.append(neighbor)
                if dfs(neighbor, depth + 1):
                    return True
                path.pop()
                visited.remove(neighbor)
        return False
    
    if dfs(start, 0):
        return best_cycle
    return None

def _extract_cycles_from_coloring(coloring, adj):
    """Given edge coloring (dict edge->color), extract cycles for each color."""
    n = P['n']
    colors = set(coloring.values())
    cycles = []
    
    for color in colors:
        # Build subgraph of edges with this color
        sub_adj = [set() for _ in range(n)]
        for (u, v), c in coloring.items():
            if c == color:
                sub_adj[u].add(v)
                sub_adj[v].add(u)
        
        # Each vertex should have degree 2 in this subgraph
        valid = True
        for v in range(n):
            if len(sub_adj[v]) != 2:
                valid = False
                break
        
        if not valid:
            continue
        
        # Find the cycle
        visited = set()
        cycle = []
        # Start from smallest unvisited vertex
        start = 0
        for v in range(n):
            if v not in visited and sub_adj[v]:
                start = v
                break
        
        current = start
        cycle.append(current)
        visited.add(current)
        prev = -1
        while True:
            neighbors = list(sub_adj[current])
            next_v = None
            for nb in neighbors:
                if nb != prev:
                    next_v = nb
                    break
            if next_v is None or next_v == start:
                break
            cycle.append(next_v)
            visited.add(next_v)
            prev = current
            current = next_v
        
        if len(cycle) == n:
            cycles.append(cycle)
    
    return cycles

def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    adj = _build_adj()
    n = P['n']
    
    # Strategy: Find edge-disjoint Hamiltonian cycles using backtracking
    # Try to find 2 cycles first (theoretical max for 5-regular graph)
    
    best_result = [list(range(n))]  # Fallback: single Hamiltonian cycle
    
    # First, verify the seed cycle is valid
    seed_cycle = list(range(n))
    valid_seed = True
    for i in range(n):
        u = seed_cycle[i]
        v = seed_cycle[(i+1) % n]
        if v not in adj[u]:
            valid_seed = False
            break
    if not valid_seed:
        # Find any Hamiltonian cycle
        hc = _find_hamiltonian_cycle(adj, set(), 0, deadline)
        if hc:
            best_result = [hc]
        else:
            return []
    
    # Now try to find a second edge-disjoint Hamiltonian cycle
    used_edges = set()
    for i in range(n):
        u = best_result[0][i]
        v = best_result[0][(i+1) % n]
        used_edges.add((min(u,v), max(u,v)))
    
    # Try different starting points for the second cycle
    found_second = False
    for start in range(n):
        if time.time() > deadline:
            break
        hc2 = _find_hamiltonian_cycle(adj, used_edges, start, deadline)
        if hc2:
            best_result.append(hc2)
            found_second = True
            break
    
    # If we found 2 cycles, try to see if we can find more
    # But theoretically max is 2 for 5-regular with Hamiltonian cycles
    # Let's also try a different approach: find any edge-disjoint cycles (not necessarily Hamiltonian)
    # Actually, the task says "simple cycles" not necessarily Hamiltonian, but they must use
    # "exactly the same vertex set of size at least three"
    
    # Let's try to find more cycles by removing edges from the first cycle
    # and finding cycles in the remaining graph
    
    if not found_second:
        # Try a completely different approach: use random restarts to find 2 edge-disjoint HC
        random.seed(42)
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Randomly permute vertices for DFS ordering
            perm = list(range(n))
            random.shuffle(perm)
            
            # Try to find first HC with this ordering
            # This is complex, so let's just try different start vertices
            
            hc1 = _find_hamiltonian_cycle(adj, set(), perm[0], deadline)
            if hc1:
                used1 = set()
                for i in range(n):
                    u = hc1[i]
                    v = hc1[(i+1) % n]
                    used1.add((min(u,v), max(u,v)))
                
                hc2 = _find_hamiltonian_cycle(adj, used1, perm[1], deadline)
                if hc2:
                    best_result = [hc1, hc2]
                    break
    
    # Convert to proper types
    result = []
    for cycle in best_result:
        result.append([int(v) for v in cycle])
    
    return result
# EVOLVE-BLOCK-END
