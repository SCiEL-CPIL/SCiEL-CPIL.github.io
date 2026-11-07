"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

_START_TIME = time.time()
GLOBAL_DEADLINE = _START_TIME + 82.0  # Safety margin

def _edge_key(u, v):
    return (min(u, v), max(u, v))

# Precompute edge lookup for O(1) checks
_ALL_EDGES = set()
_ADJ = [set() for _ in range(P['n'])]
for u, v in P['edges']:
    _ALL_EDGES.add(_edge_key(u, v))
    _ADJ[u].add(v)
    _ADJ[v].add(u)

# Precompute sorted adjacency lists
_ADJ_SORTED = [sorted(adj) for adj in _ADJ]


def find_hamiltonian_cycle(adj, n, start, available_edges, deadline, rng):
    """
    Fast iterative backtracking with Warnsdorff's rule and connectivity pruning.
    adj: list of sets (neighbors in available subgraph)
    available_edges: set of edge keys that can still be used
    """
    if time.time() > deadline:
        return None
    
    visited = [False] * n
    path = [start]
    visited[start] = True
    
    # Precompute sorted neighbor lists
    adj_list = [sorted(neigh) for neigh in adj]
    
    # Iterative DFS
    stack = [(start, 1, None)]
    
    while stack:
        if time.time() > deadline:
            return None
        
        current, path_len, it = stack[-1]
        
        if path_len == n:
            # Check closing edge
            if _edge_key(current, start) in available_edges:
                return list(path)
            stack.pop()
            path.pop()
            visited[current] = False
            continue
        
        if it is None:
            # Compute candidates with Warnsdorff's rule
            candidates = []
            for nb in adj_list[current]:
                if nb == start:
                    continue
                if not visited[nb]:
                    if _edge_key(current, nb) in available_edges:
                        # Count unvisited neighbors of nb (excluding start unless it's the last step)
                        cnt = 0
                        for nb2 in adj_list[nb]:
                            if not visited[nb2] and nb2 != start:
                                if _edge_key(nb, nb2) in available_edges:
                                    cnt += 1
                        # Bonus: if nb is the last vertex to visit, check closing edge
                        if cnt == 0 and _edge_key(nb, start) in available_edges:
                            cnt = 1  # Can close the cycle
                        candidates.append((cnt, nb))
            
            if not candidates:
                stack.pop()
                path.pop()
                visited[current] = False
                continue
            
            # Sort by Warnsdorff (fewest options first)
            candidates.sort(key=lambda x: x[0])
            
            # Randomize ties for diversity
            if len(candidates) > 1 and rng.random() < 0.4:
                groups = defaultdict(list)
                for c in candidates:
                    groups[c[0]].append(c[1])
                shuffled = []
                for key in sorted(groups.keys()):
                    group = groups[key]
                    rng.shuffle(group)
                    shuffled.extend(group)
                # Rebuild candidate list preserving order
                candidates = [(groups[c[0]][shuffled.index(c[1])] if c[1] in shuffled else c[1], c[1]) for c in candidates]
                # Simpler: just use the shuffled order
                candidates = [(groups[k][i], nb) for i, (k, nb) in enumerate([(c[0], c[1]) for c in candidates]) for k in [c[0]] for nb in [c[1]]]
                # Even simpler:
                candidates = [(cnt, nb) for cnt, nb in [(groups[c[0]][0], c[1]) for c in candidates]]
            
            it = iter(candidates)
            stack[-1] = (current, path_len, it)
        
        # Try next candidate
        found_next = False
        for _, nxt in it:
            if visited[nxt]:
                continue
            if _edge_key(current, nxt) not in available_edges:
                continue
            
            visited[nxt] = True
            path.append(nxt)
            
            # Pruning: check unvisited vertices maintain connectivity to start
            valid = True
            remaining_unvisited = [v for v in range(n) if not visited[v] and v != start]
            if remaining_unvisited:
                # Check each unvisited vertex has at least one edge to visited or other unvisited
                # More importantly: check the unvisited subgraph + start is connected
                # Quick check: each unvisited vertex must have degree >= 1 in (unvisited ∪ {start})
                for uv in remaining_unvisited:
                    deg = 0
                    for nb in adj_list[uv]:
                        if (not visited[nb]) or nb == start:
                            if _edge_key(uv, nb) in available_edges:
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
            stack.pop()
            path.pop()
            visited[current] = False
    
    return None


def construct():
    n = P['n']
    edges = P['edges']
    
    cycles = []
    used_edges = set()
    
    max_cycles_to_try = 3  # 70 edges / 21 = 3.33, so max 3
    
    for cycle_idx in range(max_cycles_to_try):
        if time.time() > GLOBAL_DEADLINE:
            break
        
        # Build subgraph of unused edges
        rem_adj = [set() for _ in range(n)]
        for u, v in edges:
            key = _edge_key(u, v)
            if key not in used_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
        
        # Quick feasibility check: min degree >= 2
        min_deg = min(len(neigh) for neigh in rem_adj)
        if min_deg < 2:
            break
        
        # Check connectivity
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
        
        # Time budget for this cycle
        remaining_time = GLOBAL_DEADLINE - time.time()
        cycles_left = max_cycles_to_try - cycle_idx
        deadline = min(GLOBAL_DEADLINE, time.time() + remaining_time / cycles_left * 0.85)
        
        available_edges = _ALL_EDGES - used_edges
        cycle = None
        
        # Strategy 1: Try vertices in order of degree (most constrained first)
        rng = random.Random(42 + cycle_idx * 1000)
        degree_order = sorted(range(n), key=lambda v: len(rem_adj[v]))
        
        for start_v in degree_order:
            if time.time() > deadline:
                break
            cycle = find_hamiltonian_cycle(rem_adj, n, start_v, available_edges, deadline, rng)
            if cycle is not None:
                break
        
        # Strategy 2: Random starts if not found
        if cycle is None:
            for _ in range(8):
                if time.time() > deadline:
                    break
                start_v = rng.randint(0, n - 1)
                cycle = find_hamiltonian_cycle(rem_adj, n, start_v, available_edges, deadline, rng)
                if cycle is not None:
                    break
        
        # Strategy 3: Try different random seeds
        if cycle is None:
            for seed_offset in range(5):
                if time.time() > deadline:
                    break
                rng2 = random.Random(12345 + cycle_idx * 7777 + seed_offset * 31)
                start_v = rng2.randint(0, n - 1)
                cycle = find_hamiltonian_cycle(rem_adj, n, start_v, available_edges, deadline, rng2)
                if cycle is not None:
                    break
        
        if cycle is None:
            break
        
        # Validate and mark edges
        valid = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            key = _edge_key(u, v)
            if key not in _ALL_EDGES or key in used_edges:
                valid = False
                break
        if not valid:
            break
        
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            used_edges.add(_edge_key(u, v))
        
        cycles.append(cycle)
    
    # Fallback: ensure at least 1 cycle
    if not cycles:
        cycle = find_hamiltonian_cycle(_ADJ, n, 0, _ALL_EDGES, min(GLOBAL_DEADLINE, time.time() + 10.0), random.Random(0))
        if cycle:
            return [[int(v) for v in cycle]]
        # Last resort: find any triangle
        for u, v in edges:
            for w in range(n):
                if w != u and w != v:
                    if _edge_key(v, w) in _ALL_EDGES and _edge_key(w, u) in _ALL_EDGES:
                        return [[int(u), int(v), int(w)]]
        return [[0, 1, 2]]
    
    # Convert to proper format
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
    
    return result
# EVOLVE-BLOCK-END
