"""Rewritten program: find edge-disjoint Hamiltonian cycles on the full 21-vertex set."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

DEADLINE = 80  # safety margin before 90s kill

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_id(u, v):
    """Canonical edge identifier."""
    if u > v:
        u, v = v, u
    return (u, v)

def _find_hamiltonian_cycles(adj, n, used_edges, max_cycles=100, time_limit=None, start_time=None):
    """
    Find Hamiltonian cycles on all n vertices avoiding used_edges.
    Uses backtracking with Warnsdorff-like heuristic.
    Returns list of cycles (each a list of vertices).
    """
    if time_limit is None:
        time_limit = 5.0
    if start_time is None:
        start_time = time.time()
    
    cycles = []
    
    # Precompute adjacency as sorted lists for efficiency
    adj_list = [sorted(adj[v]) for v in range(n)]
    
    # Fix start vertex to 0 to avoid symmetry
    start = 0
    
    def dfs(path, visited, current_edges):
        if time.time() - start_time > time_limit:
            return False  # signal timeout
        
        if len(cycles) >= max_cycles:
            return False
        
        if len(path) == n:
            last = path[-1]
            edge = _edge_id(last, start)
            if edge not in current_edges and edge not in used_edges:
                cycles.append(list(path))
            return True
        
        current = path[-1]
        
        # Get candidates: neighbors not yet visited, edge not used
        candidates = []
        for nb in adj_list[current]:
            if nb not in visited:
                e = _edge_id(current, nb)
                if e not in current_edges and e not in used_edges:
                    candidates.append(nb)
        
        if not candidates:
            return True
        
        # Warnsdorff heuristic: sort by number of onward options
        def warnsdorff_score(nb):
            count = 0
            for nn in adj_list[nb]:
                if nn not in visited:
                    e = _edge_id(nb, nn)
                    if e not in current_edges and e not in used_edges:
                        count += 1
            return count
        
        candidates.sort(key=warnsdorff_score)
        
        for nb in candidates:
            e = _edge_id(current, nb)
            visited.add(nb)
            current_edges.add(e)
            
            result = dfs(path + [nb], visited, current_edges)
            
            current_edges.discard(e)
            visited.discard(nb)
            
            if result is False:  # timeout
                return False
            if len(cycles) >= max_cycles:
                return False
        
        return True
    
    visited = {start}
    current_edges = set()
    dfs([start], visited, current_edges)
    
    return cycles

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    adj = _build_adjacency()
    n = P['n']
    
    all_edges = set()
    for u, v in P['edges']:
        all_edges.add(_edge_id(u, v))
    
    total_edges = len(all_edges)  # 63 edges
    max_possible = total_edges // n  # 63 // 21 = 3
    
    used_edges = set()
    cycles_found = []
    
    # Try to find up to max_possible edge-disjoint Hamiltonian cycles
    # Use randomized restarts if stuck
    attempts = 0
    max_attempts = 50
    
    while len(cycles_found) < max_possible and time.time() < deadline - 5:
        attempts += 1
        
        # Try to find a Hamiltonian cycle avoiding used_edges
        # Use randomized start vertex for diversity
        remaining_time = deadline - time.time() - 2
        if remaining_time <= 0:
            break
        
        time_per_try = min(3.0, remaining_time / max(1, max_possible - len(cycles_found)))
        
        # Shuffle vertex ordering for diversity in some attempts
        if attempts > 1 and attempts % 5 == 0:
            # Try with a different fixed start
            start_override = random.randint(0, n - 1)
        else:
            start_override = 0
        
        # Find cycles
        found = _find_hamiltonian_cycles(adj, n, used_edges, max_cycles=1, 
                                          time_limit=time_per_try, start_time=time.time())
        
        if found:
            # Verify the cycle is valid
            cycle = found[0]
            valid = True
            for i in range(n):
                u = cycle[i]
                v = cycle[(i + 1) % n]
                e = _edge_id(u, v)
                if e not in all_edges or e in used_edges:
                    valid = False
                    break
            if valid:
                # Check all vertices present
                if set(cycle) == set(range(n)):
                    cycles_found.append(cycle)
                    for i in range(n):
                        u = cycle[i]
                        v = cycle[(i + 1) % n]
                        used_edges.add(_edge_id(u, v))
        
        # If we've been trying for a while without success, add some randomization
        if attempts > 10 and len(cycles_found) < max_possible:
            # Try a more aggressive search with longer time limit
            remaining_time = deadline - time.time() - 2
            if remaining_time > 1:
                found = _find_hamiltonian_cycles(adj, n, used_edges, max_cycles=1,
                                                  time_limit=min(5.0, remaining_time),
                                                  start_time=time.time())
                if found:
                    cycle = found[0]
                    valid = True
                    for i in range(n):
                        u = cycle[i]
                        v = cycle[(i + 1) % n]
                        e = _edge_id(u, v)
                        if e not in all_edges or e in used_edges:
                            valid = False
                            break
                    if valid and set(cycle) == set(range(n)):
                        cycles_found.append(cycle)
                        for i in range(n):
                            u = cycle[i]
                            v = cycle[(i + 1) % n]
                            used_edges.add(_edge_id(u, v))
    
    # If we found at least one cycle, return them
    if cycles_found:
        # Ensure proper types
        result = []
        for cycle in cycles_found:
            result.append([int(v) for v in cycle])
        return result
    
    # Fallback: find any valid cycle (triangle or larger)
    # Try to find a triangle
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add(_edge_id(u, v))
    
    for u in range(n):
        if time.time() > deadline - 1:
            break
        for v in adj[u]:
            if v <= u:
                continue
            common = adj[u] & adj[v]
            for w in common:
                if w > v:
                    return [[u, v, w]]
    
    # Last resort: return a single edge-based cycle if possible
    # This shouldn't happen given the graph structure
    return [[0, 1, 2]]

# EVOLVE-BLOCK-END
