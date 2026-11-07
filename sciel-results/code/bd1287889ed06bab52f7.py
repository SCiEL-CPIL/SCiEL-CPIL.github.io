# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle(vertices, adj, used_edges, start=None):
    """Find a Hamiltonian cycle on the given vertex set using only unused edges."""
    k = len(vertices)
    if k < 3:
        return None
    vset = set(vertices)
    if start is None:
        start = min(vset)
    
    # DFS backtracking with heuristic
    path = [start]
    visited = {start}
    
    def get_candidates(current):
        candidates = []
        for nxt in adj[current]:
            if nxt not in visited and nxt in vset:
                e = (min(current, nxt), max(current, nxt))
                if e not in used_edges:
                    # Heuristic: prefer nodes with fewer available moves
                    deg = 0
                    for nnxt in adj[nxt]:
                        if nnxt not in visited and nnxt in vset:
                            e2 = (min(nxt, nnxt), max(nxt, nnxt))
                            if e2 not in used_edges:
                                deg += 1
                    # Special case for last step: check if it can connect to start
                    if len(path) == k - 1:
                        if start in adj[nxt]:
                            e3 = (min(nxt, start), max(nxt, start))
                            if e3 not in used_edges:
                                candidates.append((0, nxt)) # Priority 0
                                continue
                    candidates.append((deg, nxt))
        candidates.sort(key=lambda x: x[0])
        return [c[1] for c in candidates]

    def dfs(current):
        if len(path) == k:
            # Check if we can close the cycle
            if start in adj[current]:
                e = (min(current, start), max(current, start))
                if e not in used_edges:
                    return list(path)
            return None
        
        for nxt in get_candidates(current):
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
    
    for _ in range(max_cycles):
        if time.time() - start_time > time_limit:
            break
        # Try a few random starts if the first one fails or to diversify
        # But for now, just use min vertex for determinism and speed
        cycle = _find_ham_cycle(vertices, adj, used_edges)
        if cycle is None:
            break
        # Mark edges as used
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            e = (min(u, v), max(u, v))
            if e in used_edges:
                break
            used_edges.add(e)
        cycles.append(cycle)
    
    return cycles

def construct():
    start_time = time.time()
    deadline = 80  # seconds, leave margin before 90s kill
    
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
            
    # If we didn't find 3, try a few random restarts with different start nodes
    if best_count < 3:
        import random
        random.seed(42)
        for _ in range(5):
            if time.time() - start_time > deadline - 1:
                break
            # Shuffle vertices order slightly? No, start node matters.
            # Try different start nodes for the first cycle
            start_node = random.choice(full_vertices)
            used_edges = set()
            cycles = []
            # Force start node for first cycle
            # We need a modified _find_ham_cycle or pass start
            # Current _find_ham_cycle takes start=None. Let's add a wrapper or modify call.
            # Actually, _find_ham_cycle has start param.
            # But _find_cycles_for_vertex_set doesn't pass it.
            # Let's just call _find_ham_cycle directly in a loop here for the first cycle, then continue normally.
            
            # Simple retry logic:
            c1 = _find_ham_cycle(full_vertices, adj, used_edges, start=start_node)
            if c1:
                for i in range(len(c1)):
                    u = c1[i]
                    v = c1[(i+1) % len(c1)]
                    used_edges.add((min(u, v), max(u, v)))
                cycles.append(c1)
                
                # Find second
                c2 = _find_ham_cycle(full_vertices, adj, used_edges)
                if c2:
                    for i in range(len(c2)):
                        u = c2[i]
                        v = c2[(i+1) % len(c2)]
                        used_edges.add((min(u, v), max(u, v)))
                    cycles.append(c2)
                    
                    # Find third
                    c3 = _find_ham_cycle(full_vertices, adj, used_edges)
                    if c3:
                        for i in range(len(c3)):
                            u = c3[i]
                            v = c3[(i+1) % len(c3)]
                            used_edges.add((min(u, v), max(u, v)))
                        cycles.append(c3)
                        
                    if len(cycles) > best_count:
                        best_count = len(cycles)
                        best_cycles = cycles

    # Ensure best_cycles is non-empty
    if not best_cycles:
        # Fallback: single Hamiltonian cycle on full set
        cycles = _find_cycles_for_vertex_set(full_vertices, all_edges_set, adj, deadline - (time.time() - start_time))
        if cycles:
            best_cycles = cycles
        else:
            # Last resort: just return the seed answer (might be invalid if no cycle exists, but graph is connected)
            # Find any cycle
            # BFS/DFS for a cycle
            pass # Should not happen for this graph

    # Convert to proper types
    result = []
    for cycle in best_cycles[:10]:
        result.append(tuple(cycle))
    
    return result

# EVOLVE-BLOCK-END
