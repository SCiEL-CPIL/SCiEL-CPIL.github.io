"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency(edges):
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_hc_with_pruning(adj, n, used_edges, deadline, rng, start=0):
    """
    Find a Hamiltonian cycle starting at 'start' avoiding used_edges.
    Uses strong pruning: if any unvisited vertex has 0 available edges, fail.
    If any unvisited vertex (not the one before start in the cycle) has only 1 available edge,
    that edge must be used.
    """
    all_vertices = set(range(n))
    visited = {start}
    path = [start]
    current = start
    
    # For each vertex, track which of its edges are still available
    # available_edges[v] = set of neighbors u such that (v,u) not in used_edges
    avail = [set() for _ in range(n)]
    for v in range(n):
        for u in adj[v]:
            if (v, u) not in used_edges and (u, v) not in used_edges:
                avail[v].add(u)
    
    # Symmetry breaking: second vertex must be the smallest neighbor of start
    # that is available
    start_neighbors = sorted([u for u in avail[start] if u > start])
    if not start_neighbors:
        return None
    
    def backtrack(path, visited, current):
        if time.time() > deadline:
            return None
        
        if len(path) == n:
            # Check if we can close the cycle back to start
            if start in avail[current] and start not in visited or len(visited) == n:
                # start is visited (it's the first), so just check edge exists
                if (current, start) not in used_edges and (start, current) not in used_edges:
                    return list(path)
            return None
        
        # Pruning: check if any unvisited vertex has 0 available edges to unvisited vertices
        # (excluding the edge to 'current' if that would be the only way in)
        for v in range(n):
            if v in visited:
                continue
            # Count available edges from v to unvisited vertices (including current)
            count = 0
            for u in avail[v]:
                if u not in visited or u == current:
                    count += 1
            if count == 0:
                return None
        
        # Get neighbors of current that are unvisited and edge is available
        candidates = []
        for u in avail[current]:
            if u not in visited:
                candidates.append(u)
        
        if not candidates:
            return None
        
        # If only one candidate, must take it
        if len(candidates) == 1:
            next_v = candidates[0]
            # Tentatively use the edge
            avail[current].remove(next_v)
            avail[next_v].remove(current)
            visited.add(next_v)
            path.append(next_v)
            
            result = backtrack(path, visited, next_v)
            if result is not None:
                return result
            
            # Undo
            path.pop()
            visited.remove(next_v)
            avail[current].add(next_v)
            avail[next_v].add(current)
            return None
        
        # Multiple candidates: try them in random order
        rng.shuffle(candidates)
        
        for next_v in candidates:
            # Tentatively use the edge
            avail[current].remove(next_v)
            avail[next_v].remove(current)
            visited.add(next_v)
            path.append(next_v)
            
            result = backtrack(path, visited, next_v)
            if result is not None:
                return result
            
            # Undo
            path.pop()
            visited.remove(next_v)
            avail[current].add(next_v)
            avail[next_v].add(current)
        
        return None
    
    # Try each possible second vertex (symmetry broken by fixing start=0)
    for second_v in start_neighbors:
        # Tentatively use edge (start, second_v)
        avail[start].remove(second_v)
        avail[second_v].remove(start)
        visited = {start, second_v}
        path = [start, second_v]
        
        result = backtrack(path, visited, second_v)
        if result is not None:
            return result
        
        # Undo
        avail[start].add(second_v)
        avail[second_v].add(start)
    
    return None

def _find_hc_simple(adj, n, used_edges, deadline, rng, start=0):
    """Simpler DFS without heavy pruning, for fallback."""
    all_vertices = set(range(n))
    visited = {start}
    path = [start]
    current = start
    
    def dfs(path, visited, current):
        if time.time() > deadline:
            return None
        
        if len(path) == n:
            if (current, start) not in used_edges and (start, current) not in used_edges:
                if start in adj[current]:
                    return list(path)
            return None
        
        neighbors = list(adj[current])
        rng.shuffle(neighbors)
        
        for next_v in neighbors:
            if next_v in visited:
                continue
            if (current, next_v) in used_edges or (next_v, current) in used_edges:
                continue
            
            new_path = path + [next_v]
            new_visited = visited | {next_v}
            result = dfs(new_path, new_visited, next_v)
            if result is not None:
                return result
        return None
    
    return dfs([start], {start}, start)

def construct():
    deadline = time.time() + 55  # Safety margin before 60s limit
    n = P['n']
    adj = _build_adjacency(P['edges'])
    
    best_cycles = []
    
    # Try multiple random seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        used_edges = set()
        cycles = []
        
        for cycle_idx in range(10):
            if time.time() > deadline:
                break
            
            found = False
            # Try with pruning first
            hc = _find_hc_with_pruning(adj, n, used_edges, deadline, rng, start=0)
            if hc is None:
                # Fallback to simpler DFS
                hc = _find_hc_simple(adj, n, used_edges, deadline, rng, start=0)
            
            if hc is not None:
                cycles.append(hc)
                for i in range(len(hc)):
                    u = hc[i]
                    v = hc[(i + 1) % len(hc)]
                    used_edges.add((u, v))
                    used_edges.add((v, u))
                found = True
            else:
                break
        
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
            if len(best_cycles) >= 3:  # Theoretical max is 3
                break
    
    # Convert to proper format
    result = []
    for cycle in best_cycles:
        cycle_ints = [int(v) for v in cycle]
        result.append(cycle_ints)
    
    # Fallback if no cycles found
    if not result:
        rng = random.Random(42)
        used_edges = set()
        hc = _find_hc_simple(adj, n, used_edges, deadline, rng, start=0)
        if hc is not None:
            result = [[int(v) for v in hc]]
    
    if not result:
        # Last resort: find any cycle
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w != u and w != v and w in adj[u]:
                            result = [[u, v, w]]
                            break
                    if result:
                        break
            if result:
                break
    
    return result
# EVOLVE-BLOCK-END
