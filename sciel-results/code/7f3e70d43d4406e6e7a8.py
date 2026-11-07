"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

_START_TIME = time.time()
GLOBAL_DEADLINE = _START_TIME + 78.0  # Safety margin before 90s kill

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def build_adjacency(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def find_hamiltonian_cycle(adj, n, start, edge_set, deadline, rng=None):
    """
    Iterative backtracking search for a Hamiltonian cycle with Warnsdorff's rule
    and degree-based pruning. Uses explicit stack to avoid recursion overhead.
    """
    if time.time() > deadline:
        return None
    
    if rng is None:
        rng = random.Random(42)
    
    visited = [False] * n
    path = [start]
    visited[start] = True
    
    # Precompute sorted neighbor lists for speed
    adj_list = [sorted(neigh) for neigh in adj]
    
    # Iterative DFS with explicit stack
    # Stack entries: (current_vertex, path_length, iterator_over_candidates)
    stack = [(start, 1, None)]
    
    while stack:
        if time.time() > deadline:
            return None
        
        current, path_len, it = stack[-1]
        
        if path_len == n:
            # Check if we can close the cycle
            if _edge_key(current, start) in edge_set:
                return list(path)
            stack.pop()
            path.pop()
            visited[current] = False
            continue
        
        if it is None:
            # First visit to this vertex: compute candidates
            candidates = []
            for nb in adj_list[current]:
                if nb == start:
                    continue
                if not visited[nb]:
                    if _edge_key(current, nb) in edge_set:
                        # Count unvisited neighbors of nb (Warnsdorff's rule)
                        cnt = 0
                        for nb2 in adj_list[nb]:
                            if not visited[nb2] and nb2 != start:
                                if _edge_key(nb, nb2) in edge_set:
                                    cnt += 1
                        # Pruning: if cnt == 0 and nb is not the last unvisited, it's a dead end
                        # Actually, if cnt == 0 and there are still unvisited vertices after nb, it's bad
                        candidates.append((cnt, nb))
            
            if not candidates:
                # Dead end
                stack.pop()
                path.pop()
                visited[current] = False
                continue
            
            # Sort by Warnsdorff's rule (fewest forward options first)
            candidates.sort(key=lambda x: x[0])
            
            # For diversity, occasionally shuffle among ties
            if len(candidates) > 1 and rng.random() < 0.3:
                # Group by count and shuffle within groups
                groups = defaultdict(list)
                for c in candidates:
                    groups[c[0]].append(c[1])
                shuffled = []
                for key in sorted(groups.keys()):
                    group = groups[key]
                    rng.shuffle(group)
                    shuffled.extend(group)
                candidates = [(groups[k][0], nb) for k, nb in zip([c[0] for c in candidates], shuffled)]
            
            it = iter(candidates)
            stack[-1] = (current, path_len, it)
        
        # Try to get next candidate
        found_next = False
        for _, nxt in it:
            if visited[nxt]:
                continue
            if _edge_key(current, nxt) not in edge_set:
                continue
            
            # Additional pruning: check if nxt would leave any unvisited vertex isolated
            # (i.e., has no available neighbors)
            visited[nxt] = True
            path.append(nxt)
            
            # Check if any unvisited vertex (other than start) has degree 0 in remaining graph
            valid = True
            remaining_unvisited = [v for v in range(n) if not visited[v] and v != start]
            if remaining_unvisited:
                for uv in remaining_unvisited:
                    deg = 0
                    for nb in adj_list[uv]:
                        if not visited[nb] or nb == start:
                            if _edge_key(uv, nb) in edge_set:
                                deg += 1
                    if deg == 0:
                        valid = False
                        break
            
            if valid:
                stack.append((nxt, path_len + 1, None))
                found_next = True
                break
            else:
                visited[nxt] = False
                path.pop()
        
        if not found_next:
            # No more candidates from this vertex
            stack.pop()
            path.pop()
            visited[current] = False
    
    return None

def construct():
    n = P['n']
    edges = P['edges']
    
    # Precompute all edge keys
    all_edge_keys = set()
    for u, v in edges:
        all_edge_keys.add(_edge_key(u, v))
    
    # Build initial adjacency
    adj = build_adjacency(edges, n)
    
    cycles = []
    used_edges = set()
    
    max_cycles_to_try = 3
    
    for cycle_idx in range(max_cycles_to_try):
        if time.time() > GLOBAL_DEADLINE:
            break
        
        # Build subgraph of unused edges
        remaining_edges = []
        rem_adj = [set() for _ in range(n)]
        for u, v in edges:
            key = _edge_key(u, v)
            if key not in used_edges:
                remaining_edges.append((u, v))
                rem_adj[u].add(v)
                rem_adj[v].add(u)
        
        # Quick check: min degree >= 2
        min_deg = min(len(neigh) for neigh in rem_adj)
        if min_deg < 2:
            break
            
        # Check connectivity of remaining graph
        visited_conn = [False] * n
        stack_conn = [0]
        visited_conn[0] = True
        while stack_conn:
            u = stack_conn.pop()
            for v in rem_adj[u]:
                if not visited_conn[v]:
                    visited_conn[v] = True
                    stack_conn.append(v)
        if not all(visited_conn):
            break
        
        # Deadline for this cycle search: allocate remaining time / remaining cycles
        remaining_time = GLOBAL_DEADLINE - time.time()
        cycles_left = max_cycles_to_try - cycle_idx
        deadline = min(GLOBAL_DEADLINE, time.time() + remaining_time / cycles_left * 0.9)
        
        cycle = None
        
        # Try multiple starting vertices with randomization
        # Use a fixed seed per cycle for reproducibility but different across cycles
        rng = random.Random(42 + cycle_idx * 1000)
        
        # Try starts in order, but also try random starts for diversity
        start_order = list(range(n))
        # Interleave: 0, 1, 2, ... then random
        for start_v in start_order:
            if time.time() > deadline:
                break
            cycle = find_hamiltonian_cycle(rem_adj, n, start_v, all_edge_keys - used_edges, deadline, rng)
            if cycle is not None:
                break
        
        if cycle is None:
            # Try a few more random starts
            for _ in range(5):
                if time.time() > deadline:
                    break
                start_v = rng.randint(0, n - 1)
                cycle = find_hamiltonian_cycle(rem_adj, n, start_v, all_edge_keys - used_edges, deadline, rng)
                if cycle is not None:
                    break
        
        if cycle is None:
            break
        
        # Mark edges as used
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            key = _edge_key(u, v)
            if key not in all_edge_keys:
                break
            used_edges.add(key)
        
        cycles.append(cycle)
        
        if len(cycles) == 3:
            break

    # Ensure we have at least 1 cycle
    if not cycles:
        adj_full = build_adjacency(edges, n)
        cycle = find_hamiltonian_cycle(adj_full, n, 0, all_edge_keys, min(GLOBAL_DEADLINE, time.time() + 5.0))
        if cycle:
            return [cycle]
        else:
            # Last resort: find any 3-cycle
            for u, v in edges:
                for w in range(n):
                    if w != u and w != v:
                        if _edge_key(u, v) in all_edge_keys and _edge_key(v, w) in all_edge_keys and _edge_key(w, u) in all_edge_keys:
                            return [[int(u), int(v), int(w)]]
            return [[0, 1, 2]]

    # Convert to proper format
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
    
    return result
# EVOLVE-BLOCK-END
