# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle_smart(vertices, adj, used_edges, start=None, deadline=None, rng=None):
    """
    Find a Hamiltonian cycle on the given vertex set using only unused edges.
    Uses backtracking with Warnsdorff's rule and degree pruning.
    """
    k = len(vertices)
    if k < 3:
        return None
    vset = frozenset(vertices)
    if start is None:
        start = min(vset)
    if rng is None:
        rng = random
    
    # Pre-compute available neighbors for each vertex given used_edges
    # avail[u] is a set of neighbors v such that (u,v) is in adj and not in used_edges
    avail = defaultdict(set)
    for u in vertices:
        for v in adj[u]:
            if v in vset:
                e = (min(u, v), max(u, v))
                if e not in used_edges:
                    avail[u].add(v)
    
    # Check if start has at least 2 available neighbors
    if len(avail[start]) < 2:
        return None

    path = [start]
    visited = {start}
    node_count = 0

    def get_candidates(current):
        candidates = []
        for nxt in avail[current]:
            if nxt not in visited:
                # Warnsdorff's rule: prefer nodes with fewer available exits
                # This helps avoid getting stuck in large open areas
                # Note: if nxt is the last node, it must connect back to start
                if len(path) == k - 1:
                    if start in avail[nxt]:
                        candidates.append((0, nxt)) # Highest priority
                else:
                    # Count available neighbors for nxt that are unvisited
                    # Also, if nxt is not the last, it needs at least 1 unvisited neighbor (other than current) 
                    # OR if it is the last, it needs start.
                    # Simple heuristic: number of unvisited neighbors in avail[nxt]
                    unvisited_neighbors = [n for n in avail[nxt] if n not in visited]
                    # If we are not at the last step, nxt must have at least 1 unvisited neighbor 
                    # to continue the path, UNLESS it is the last node.
                    # But here we are choosing the NEXT node.
                    # If len(path) < k-1, then after adding nxt, there are still nodes left.
                    # So nxt must have at least 1 unvisited neighbor (excluding current, which is now visited).
                    # Actually, current is visited. So we look at avail[nxt] \ {current} \ visited.
                    # But current is already in visited.
                    # So just count avail[nxt] not in visited.
                    if len(unvisited_neighbors) == 0:
                        # Dead end, unless this is the last node
                        if len(path) == k - 1:
                             if start in avail[nxt]:
                                candidates.append((0, nxt))
                        else:
                            continue 
                    else:
                        candidates.append((len(unvisited_neighbors), nxt))
        
        # Shuffle to break ties randomly for diversity
        if candidates:
            # Sort by heuristic, then shuffle ties? 
            # For now, just sort. To add randomness, we can shuffle the initial list or the final list.
            # Let's shuffle the candidates list before sorting to randomize tie-breaking?
            # No, better to sort by heuristic, and if heuristics are equal, randomize.
            # Simple approach: sort by (heuristic, random.random())
            if rng:
                scored = [(h, rng.random(), n) for h, n in candidates]
                scored.sort()
                return [n for _, _, n in scored]
            else:
                candidates.sort(key=lambda x: x[0])
                return [n for _, n in candidates]
        return []

    def dfs(current):
        nonlocal node_count
        node_count += 1
        if deadline is not None and node_count % 5000 == 0 and time.time() > deadline:
            return None
        
        if len(path) == k:
            if start in avail[current]:
                return list(path)
            return None
        
        for nxt in get_candidates(current):
            visited.add(nxt)
            path.append(nxt)
            result = dfs(nxt)
            if result is not None:
                return result
            path.pop()
            visited.remove(nxt)
        return None
    
    return dfs(start)

def _find_cycles_for_vertex_set(vertices, adj, time_limit, max_cycles=10, seed=42):
    """Find max number of edge-disjoint cycles on the given vertex set."""
    vertices = list(vertices)
    used_edges = set()
    cycles = []
    start_time = time.time()
    deadline = start_time + time_limit
    rng = random.Random(seed)
    
    for _ in range(max_cycles):
        if time.time() > deadline - 0.1:
            break
        
        # Randomize the starting vertex to find different cycles
        start = rng.choice(vertices)
        
        cycle = _find_ham_cycle_smart(vertices, adj, used_edges, start=start, deadline=deadline, rng=rng)
        if cycle is None:
            # Try a few more random starts before giving up
            found = False
            for _retry in range(5):
                if time.time() > deadline - 0.1:
                    break
                start = rng.choice(vertices)
                cycle = _find_ham_cycle_smart(vertices, adj, used_edges, start=start, deadline=deadline, rng=rng)
                if cycle is not None:
                    found = True
                    break
            if not found:
                break
        
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
    
    return cycles

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    all_edges_set = set()
    for u, v in edges:
        all_edges_set.add((min(u, v), max(u, v)))
    
    adj = _build_adj(n, edges)
    full_vertices = list(range(n))
    
    best_cycles = []
    best_count = 0
    
    # Try multiple random seeds
    for seed in [42, 123, 7, 999, 555, 1, 2, 3]:
        if time.time() > deadline - 2:
            break
        
        cycles = _find_cycles_for_vertex_set(full_vertices, adj, deadline - time.time(), max_cycles=10, seed=seed)
        
        if len(cycles) > best_count:
            best_count = len(cycles)
            best_cycles = cycles
        
        if best_count >= 10:
            break
    
    # If we didn't find many, try a few more seeds or different strategies
    if best_count < 5:
        for seed in range(100, 200):
            if time.time() > deadline - 1:
                break
            cycles = _find_cycles_for_vertex_set(full_vertices, adj, deadline - time.time(), max_cycles=10, seed=seed)
            if len(cycles) > best_count:
                best_count = len(cycles)
                best_cycles = cycles
            if best_count >= 10:
                break

    # Ensure best_cycles is non-empty
    if not best_cycles:
        cycles = _find_cycles_for_vertex_set(full_vertices, adj, deadline - time.time(), max_cycles=10)
        if cycles:
            best_cycles = cycles

    # Convert to proper types (list of tuples)
    result = []
    for cycle in best_cycles[:10]:
        result.append(tuple(cycle))
    
    return result

# EVOLVE-BLOCK-END
