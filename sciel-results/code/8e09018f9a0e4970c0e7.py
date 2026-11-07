"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle(vertices, adj, used_edges, start=None, time_limit=0.1):
    """Find a Hamiltonian cycle on the given vertex set using only unused edges."""
    k = len(vertices)
    if k < 3:
        return None
    vset = set(vertices)
    if start is None:
        start = min(vset)
    
    start_time = time.time()
    path = [start]
    visited = {start}
    
    def dfs(current):
        if time.time() - start_time > time_limit:
            return None
        if len(path) == k:
            if start in adj[current] and (min(current, start), max(current, start)) not in used_edges:
                return list(path)
            return None
        
        # Heuristic: Sort neighbors by number of available unused edges (Warnsdorff)
        neighbors = []
        for nxt in adj[current]:
            if nxt not in visited and nxt in vset:
                e = (min(current, nxt), max(current, nxt))
                if e not in used_edges:
                    # Count available edges from nxt to unvisited vertices
                    avail = 0
                    for nnxt in adj[nxt]:
                        if nnxt not in visited and nnxt in vset:
                            e2 = (min(nxt, nnxt), max(nxt, nnxt))
                            if e2 not in used_edges:
                                avail += 1
                    neighbors.append((avail, nxt))
        
        # Sort by fewest available edges first
        neighbors.sort(key=lambda x: x[0])
        
        for _, nxt in neighbors:
            e = (min(current, nxt), max(current, nxt))
            if e not in used_edges:
                visited.add(nxt)
                path.append(nxt)
                result = dfs(nxt)
                if result is not None:
                    return result
                path.pop()
                visited.remove(nxt)
        return None
    
    return dfs(start)

def _find_cycles_for_vertex_set(vertices, all_edges_set, adj, time_limit, max_cycles=10):
    """Find max number of edge-disjoint cycles on the given vertex set."""
    vertices = list(vertices)
    used_edges = set()
    cycles = []
    start_time = time.time()
    
    # Shuffle vertices to break symmetry if needed, but we use min(vset) as start
    # Randomizing start might help if stuck
    starts = list(vertices)
    random.shuffle(starts)
    
    for _ in range(max_cycles):
        if time.time() - start_time > time_limit:
            break
        
        found = False
        for s in starts:
            cycle = _find_ham_cycle(vertices, adj, used_edges, start=s, time_limit=min(time_limit - (time.time() - start_time), 0.5))
            if cycle is not None:
                # Mark edges as used
                valid = True
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i+1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    if e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                
                if valid:
                    cycles.append(cycle)
                    found = True
                    break
        
        if not found:
            break
    
    return cycles

def construct():
    start_time = time.time()
    deadline = 80  # seconds
    
    n = P['n']
    edges = P['edges']
    all_edges_set = set()
    for u, v in edges:
        all_edges_set.add((min(u, v), max(u, v)))
    
    adj = _build_adj(n, edges)
    
    best_cycles = []
    best_count = 0
    
    # Strategy 1: Full vertex set
    full_vertices = list(range(n))
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 1:
        cycles = _find_cycles_for_vertex_set(full_vertices, all_edges_set, adj, remaining_time)
        if len(cycles) > best_count:
            best_count = len(cycles)
            best_cycles = cycles
    
    # If we didn't get 3, try random subsets or restarts
    if best_count < 3:
        for attempt in range(10):
            if time.time() - start_time > deadline - 1:
                break
            # Try random permutation of vertices for start order
            random.seed(attempt)
            cycles = _find_cycles_for_vertex_set(full_vertices, all_edges_set, adj, min(remaining_time, 5))
            if len(cycles) > best_count:
                best_count = len(cycles)
                best_cycles = cycles
                if best_count == 3:
                    break
    
    # Ensure best_cycles is non-empty
    if not best_cycles:
        # Fallback: single Hamiltonian cycle on full set
        cycles = _find_cycles_for_vertex_set(full_vertices, all_edges_set, adj, deadline - (time.time() - start_time))
        if cycles:
            best_cycles = cycles
        else:
            # Last resort: just return the seed answer (might be invalid if no cycle found, but we assume at least 1 exists)
            best_cycles = [list(range(n))]
    
    # Convert to proper types (list of lists)
    result = []
    for cycle in best_cycles[:10]:
        result.append(list(cycle))
    
    return result

# EVOLVE-BLOCK-END
