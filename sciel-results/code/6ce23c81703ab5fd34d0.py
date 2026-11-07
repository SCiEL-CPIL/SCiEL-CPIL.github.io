"""Minimum biclique partition via direct recursive backtracking with memoization and branch-and-bound."""
import time
import sys
from functools import lru_cache

sys.setrecursionlimit(1000000)

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _validate(bicliques, adj, all_edges_set):
    n = P['n']
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        if set(left).intersection(right):
            return False
        for u in left:
            if u < 0 or u >= n or not isinstance(u, int):
                return False
            for v in right:
                if v < 0 or v >= n or not isinstance(v, int):
                    return False
                if v not in adj[u]:
                    return False
                e = (min(u, v), max(u, v))
                if e in covered:
                    return False
                covered.add(e)
    if covered != all_edges_set:
        return False
    return True

def _expand_maximal(left, right, adj, n):
    """Expand (left, right) to a maximal biclique by adding vertices connected to all of the other side."""
    left = set(left)
    right = set(right)
    changed = True
    while changed:
        changed = False
        new_left = set()
        for u in range(n):
            if u in left or u in right:
                continue
            if right.issubset(adj[u]):
                new_left.add(u)
        if new_left:
            left |= new_left
            changed = True
        new_right = set()
        for v in range(n):
            if v in left or v in right:
                continue
            if left.issubset(adj[v]):
                new_right.add(v)
        if new_right:
            right |= new_right
            changed = True
    right -= left
    left -= right
    return left, right

def _biclique_edges(left, right):
    s = set()
    for u in left:
        for v in right:
            s.add((min(u, v), max(u, v)))
    return s

def _greedy_partition(adj, all_edges_set, n, deadline):
    """Greedy: repeatedly pick the maximal biclique covering the most remaining edges."""
    remaining = set(all_edges_set)
    result = []
    while remaining:
        if time.time() > deadline:
            break
        # Pick a random-ish edge to seed (use first for determinism in greedy)
        # Actually, pick the edge with highest degree potential
        best_biclique = None
        best_count = 0
        # Try seeding from each remaining edge
        for e in list(remaining):
            u, v = e
            l, r = _expand_maximal({u}, {v}, adj, n)
            edges = _biclique_edges(l, r)
            count = len(edges & remaining)
            if count > best_count:
                best_count = count
                best_biclique = (l, r, edges)
        if best_biclique is None:
            e = next(iter(remaining))
            u, v = e
            l, r = _expand_maximal({u}, {v}, adj, n)
            edges = _biclique_edges(l, r)
            best_biclique = (l, r, edges)
        l, r, edges = best_biclique
        remaining -= edges
        result.append((l, r))
    return result

def _solve_bpp(adj, all_edges_set, n, deadline):
    """
    Direct backtracking for minimum biclique partition.
    State: frozenset of remaining edges.
    At each step, pick a remaining edge, generate all maximal bicliques that cover it,
    and branch on each.
    Use memoization: if we've seen a remaining-edge set with a solution of size k,
    we don't need to search it again if we already have a solution <= k.
    """
    memo = {}  # frozenset(remaining) -> best solution size found
    best_solution = [None]
    best_size = [float('inf')]
    
    # Precompute adjacency as bitmasks for speed
    n = P['n']
    adj_mask = [0] * n
    for u in range(n):
        for v in adj[u]:
            adj_mask[u] |= (1 << v)
    
    def expand_maximal_mask(left_mask, right_mask):
        """Expand to maximal biclique using bitmasks."""
        left = left_mask
        right = right_mask
        changed = True
        while changed:
            changed = False
            # Find vertices connected to all of right
            # A vertex u can be added to left if (adj_mask[u] & right) == right and u not in left|right
            candidates = 0
            for u in range(n):
                if left & (1 << u) or right & (1 << u):
                    continue
                if (adj_mask[u] & right) == right:
                    candidates |= (1 << u)
            if candidates:
                left |= candidates
                changed = True
            candidates = 0
            for v in range(n):
                if left & (1 << v) or right & (1 << v):
                    continue
                if (adj_mask[v] & left) == left:
                    candidates |= (1 << v)
            if candidates:
                right |= candidates
                changed = True
        right &= ~left
        left &= ~right
        return left, right
    
    def mask_to_edge_set(mask):
        """Convert a bitmask of edges to a frozenset of edges."""
        # This is slow; instead, we represent edges differently.
        # Let's use a different representation: edges as a frozenset of (u,v) tuples.
        pass
    
    # Use a simpler approach: represent remaining edges as a frozenset of (u,v) tuples.
    # For expansion, use sets.
    
    def solve(remaining, depth):
        nonlocal best_solution, best_size
        if time.time() > deadline:
            return
        if not remaining:
            if depth < best_size[0]:
                best_size[0] = depth
                best_solution[0] = []  # We need to track the actual bicliques, not just count
            return
        
        # Memoization check
        rem_key = frozenset(remaining)
        if rem_key in memo:
            if memo[rem_key] <= depth:
                return  # We've already found a solution for this state with <= depth
        else:
            memo[rem_key] = float('inf')
        
        # Lower bound: each biclique can cover at most max_biclique_edges edges
        # Compute a simple lower bound
        if depth + 1 >= best_size[0]:
            return
        
        # Pick the remaining edge with the fewest possible biclique covers
        # For each remaining edge, count how many maximal bicliques cover it
        # To save time, just pick the first edge
        target_edge = next(iter(remaining))
        u, v = target_edge
        
        # Generate all maximal bicliques that cover target_edge
        # We can seed with {u},{v} and also try other combinations
        seen_bicliques = set()
        candidates = []
        
        # Seed 1: {u} on left, {v} on right
        l, r = _expand_maximal({u}, {v}, adj, n)
        if l and r:
            key = (frozenset(l), frozenset(r))
            if key not in seen_bicliques:
                seen_bicliques.add(key)
                edges = _biclique_edges(l, r)
                if target_edge in edges:
                    candidates.append((l, r, edges))
        
        # Seed 2: {v} on left, {u} on right
        l, r = _expand_maximal({v}, {u}, adj, n)
        if l and r:
            key = (frozenset(l), frozenset(r))
            if key not in seen_bicliques:
                seen_bicliques.add(key)
                edges = _biclique_edges(l, r)
                if target_edge in edges:
                    candidates.append((l, r, edges))
        
        # Try seeding with u and each neighbor of v (other than u)
        for w in adj[v]:
            if w == u:
                continue
            l, r = _expand_maximal({u, w}, {v}, adj, n)
            if l and r:
                key = (frozenset(l), frozenset(r))
                if key not in seen_bicliques:
                    seen_bicliques.add(key)
                    edges = _biclique_edges(l, r)
                    if target_edge in edges:
                        candidates.append((l, r, edges))
        
        # Try seeding with v and each neighbor of u (other than v)
        for w in adj[u]:
            if w == v:
                continue
            l, r = _expand_maximal({v, w}, {u}, adj, n)
            if l and r:
                key = (frozenset(l), frozenset(r))
                if key not in seen_bicliques:
                    seen_bicliques.add(key)
                    edges = _biclique_edges(l, r)
                    if target_edge in edges:
                        candidates.append((l, r, edges))
        
        if not candidates:
            # Fallback: just the edge itself
            l = {u}
            r = {v}
            edges = {target_edge}
            candidates = [(l, r, edges)]
        
        # Sort candidates by coverage (more coverage first)
        candidates.sort(key=lambda x: len(x[2] & remaining), reverse=True)
        
        for l, r, edges in candidates:
            if time.time() > deadline:
                return
            new_remaining = remaining - edges
            if len(new_remaining) == len(remaining):
                continue  # No progress
            solve(new_remaining, depth + 1)
            if best_size[0] <= depth + 1:
                return  # Can't improve
        
        memo[rem_key] = depth
    
    # We need to track the actual bicliques, not just the count.
    # Let's restructure to track the solution.
    
    solution_stack = []
    best_solution = [None]
    best_size = [float('inf')]
    memo = {}
    
    def solve2(remaining, depth):
        nonlocal best_solution, best_size
        if time.time() > deadline:
            return
        if not remaining:
            if depth < best_size[0]:
                best_size[0] = depth
                best_solution[0] = list(solution_stack)
            return
        
        rem_key = frozenset(remaining)
        if rem_key in memo:
            if memo[rem_key] <= depth:
                return
        else:
            memo[rem_key] = float('inf')
        
        if depth + 1 >= best_size[0]:
            return
        
        target_edge = next(iter(remaining))
        u, v = target_edge
        
        seen_bicliques = set()
        candidates = []
        
        l, r = _expand_maximal({u}, {v}, adj, n)
        if l and r:
            key = (frozenset(l), frozenset(r))
            if key not in seen_bicliques:
                seen_bicliques.add(key)
                edges = _biclique_edges(l, r)
                if target_edge in edges:
                    candidates.append((l, r, edges))
        
        l, r = _expand_maximal({v}, {u}, adj, n)
        if l and r:
            key = (frozenset(l), frozenset(r))
            if key not in seen_bicliques:
                seen_bicliques.add(key)
                edges = _biclique_edges(l, r)
                if target_edge in edges:
                    candidates.append((l, r, edges))
        
        for w in adj[v]:
            if w == u:
                continue
            l, r = _expand_maximal({u, w}, {v}, adj, n)
            if l and r:
                key = (frozenset(l), frozenset(r))
                if key not in seen_bicliques:
                    seen_bicliques.add(key)
                    edges = _biclique_edges(l, r)
                    if target_edge in edges:
                        candidates.append((l, r, edges))
        
        for w in adj[u]:
            if w == v:
                continue
            l, r = _expand_maximal({v, w}, {u}, adj, n)
            if l and r:
                key = (frozenset(l), frozenset(r))
                if key not in seen_bicliques:
                    seen_bicliques.add(key)
                    edges = _biclique_edges(l, r)
                    if target_edge in edges:
                        candidates.append((l, r, edges))
        
        if not candidates:
            l = {u}
            r = {v}
            edges = {target_edge}
            candidates = [(l, r, edges)]
        
        candidates.sort(key=lambda x: len(x[2] & remaining), reverse=True)
        
        for l, r, edges in candidates:
            if time.time() > deadline:
                return
            new_remaining = remaining - edges
            if len(new_remaining) == len(remaining):
                continue
            solution_stack.append((l, r))
            solve2(new_remaining, depth + 1)
            solution_stack.pop()
            if best_size[0] <= depth + 1:
                return
        
        memo[rem_key] = depth
    
    solve2(all_edges_set, 0)
    return best_solution[0]

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = P['n']
    adj = _build_adjacency()
    all_edges_set = frozenset((min(u, v), max(u, v)) for u, v in P['edges'])
    
    # Phase 1: Greedy for initial incumbent
    greedy_result = _greedy_partition(adj, set(all_edges_set), n, deadline)
    best_result = greedy_result
    best_count = len(greedy_result) if greedy_result else float('inf')
    
    # Phase 2: Exact backtracking
    if time.time() < deadline:
        exact_result = _solve_bpp(adj, set(all_edges_set), n, deadline)
        if exact_result is not None:
            if len(exact_result) < best_count:
                best_count = len(exact_result)
                best_result = exact_result
    
    # Final fallback
    if best_result is None:
        best_result = [(frozenset({u}), frozenset({v})) for u, v in P['edges']]
    
    # Convert to proper types
    result = []
    for left, right in best_result:
        result.append([list(left), list(right)])
    
    return result
