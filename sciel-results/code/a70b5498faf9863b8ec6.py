import time
import random
import heapq

_DEADLINE = 140.0  # safety margin under 150s limit


def _build_adj(n, edges):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj


def _has_c5_fast(adj, n):
    """Check if graph contains a C5. Returns True if found."""
    for x in range(n):
        ax = adj[x]
        # Get neighbors of x
        neigh = []
        m = ax
        while m:
            lb = m & (-m)
            y = lb.bit_length() - 1
            neigh.append(y)
            m ^= lb
        ln = len(neigh)
        if ln < 4:
            continue
        # For each pair (a, b) of neighbors of x, check if there's a path a-?-b
        # where ? is a neighbor of both a and b, ? != x, and a != b
        for i in range(ln):
            a = neigh[i]
            for j in range(i + 1, ln):
                b = neigh[j]
                # a and b are both neighbors of x
                # Check if a and b are adjacent (would form triangle, not C5 directly)
                # For C5: x-a-?-b-x, need ? adjacent to both a and b, ? != x
                # Also need a not adjacent to b (otherwise x-a-b-x is triangle)
                # Actually C5 is x-a-?-b-x, so we need:
                # - a adjacent to x (yes)
                # - a adjacent to ?
                # - ? adjacent to b  
                # - b adjacent to x (yes)
                # - x adjacent to x? No, x-a-?-b-x is a cycle of length 4 if a-b edge exists
                # Wait, C5 has 5 vertices: x, a, ?, b, and one more
                # Let me reconsider: C5 is a 5-cycle. 
                # A 5-cycle through x: x-v1-v2-v3-v4-x
                # v1 and v4 are neighbors of x
                # v2 is neighbor of v1, v3 is neighbor of v4
                # v2-v3 is an edge
                # So: pick two neighbors v1, v4 of x (v1 != v4)
                # Find v2 in N(v1) \ {x, v4}, v3 in N(v4) \ {x, v1}
                # Check if v2-v3 is an edge and v2 != v3
                if (ax >> b) & 1:
                    # a and b are adjacent, so x-a-b-x is a triangle
                    # Can't be part of C5 through this pair in this way
                    continue
                # Find common neighbors of a and b, excluding x
                common = adj[a] & adj[b] & ~(1 << x) & ~(1 << a) & ~(1 << b)
                if common:
                    return True
    return False


def _find_c5_edge(adj, n):
    """Find an edge that is part of a C5. Returns (u, v) or None."""
    for x in range(n):
        ax = adj[x]
        neigh = []
        m = ax
        while m:
            lb = m & (-m)
            y = lb.bit_length() - 1
            neigh.append(y)
            m ^= lb
        ln = len(neigh)
        if ln < 4:
            continue
        for i in range(ln):
            a = neigh[i]
            for j in range(i + 1, ln):
                b = neigh[j]
                if (ax >> b) & 1:
                    continue
                common = adj[a] & adj[b] & ~(1 << x) & ~(1 << a) & ~(1 << b)
                if common:
                    # Found a C5: x-a-?-b-x where ? is in common
                    # The edges x-a, a-?, ?-b, b-x, and x-? are part of cycles
                    # Return edge (x, a) as it's part of this C5
                    return (x, a)
    return None


def _alpha_lower_bound(adj, n):
    """Quick lower bound on independence number using greedy."""
    # Greedy: repeatedly pick vertex with min degree in remaining graph
    remaining = (1 << n) - 1
    size = 0
    while remaining:
        # Find vertex with min degree in remaining
        best_v = -1
        best_deg = n + 1
        m = remaining
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            deg = bin(adj[v] & remaining).count('1')
            if deg < best_deg:
                best_deg = deg
                best_v = v
            m ^= lb
        if best_v >= 0:
            size += 1
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
    return size


def _has_independent_set(adj, n, k=9):
    """Check if there's an independent set of size k using backtracking."""
    # Order by degree (low degree first for better pruning)
    degs = [bin(adj[v]).count('1') for v in range(n)]
    order = sorted(range(n), key=lambda v: degs[v])
    
    def rec(candidates, size, start_idx):
        if size >= k:
            return True
        if start_idx >= n:
            return False
        # Bound: even if we take all remaining, can we reach k?
        remaining_count = n - start_idx
        if size + remaining_count < k:
            return False
        # Try to include order[start_idx]
        v = order[start_idx]
        if (candidates >> v) & 1:
            # Include v: remove v and all its neighbors
            newcand = candidates & ~(1 << v) & ~adj[v]
            if rec(newcand, size + 1, start_idx + 1):
                return True
            # Exclude v
            newcand2 = candidates & ~(1 << v)
            if rec(newcand2, size, start_idx + 1):
                return True
            return False
        else:
            # v not in candidates, skip
            return rec(candidates, size, start_idx + 1)
    
    return rec((1 << n) - 1, 0, 0)


def _is_valid(n, edges):
    adj = _build_adj(n, edges)
    if _has_c5_fast(adj, n):
        return False
    if _has_independent_set(adj, n, 9):
        return False
    return True


def _base_solution():
    """Return the known 32-vertex solution: 8 disjoint K4s."""
    edges = []
    for block in range(8):
        base = block * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return 32, edges


def _construct_dense(n, rng):
    """Create a dense random graph on n vertices."""
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < 0.7:  # 70% edge probability
                adj[u] |= 1 << v
                adj[v] |= 1 << u
    return adj


def _remove_c5_edges(adj, n, max_iter=10000):
    """Iteratively remove edges that are part of C5s until no C5 remains."""
    for it in range(max_iter):
        edge = _find_c5_edge(adj, n)
        if edge is None:
            break
        u, v = edge
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    return adj


def _add_edges_to_reduce_alpha(adj, n, target_alpha=8, max_iter=5000):
    """Add edges to reduce independence number below target."""
    for it in range(max_iter):
        if not _has_independent_set(adj, n, 9):
            break
        # Find an independent set of size 9 (or close)
        # Add edges within it to break it up
        # Simple approach: randomly add an edge between non-adjacent vertices
        u = rng.randint(0, n - 1) if 'rng' in dir() else random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v or (adj[u] >> v) & 1:
            continue
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj


def _local_search(n, time_budget):
    """Local search: start dense, remove C5 edges, then fix alpha."""
    global rng
    rng = random.Random(42)
    deadline = time.time() + time_budget
    
    best_n = 0
    best_edges = None
    
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        
        # Start with dense random graph
        adj = _construct_dense(n, rng)
        
        # Remove edges to eliminate C5s
        adj = _remove_c5_edges(adj, n, max_iter=5000)
        
        # Check if alpha is ok
        if not _has_independent_set(adj, n, 9):
            # Valid! Extract edges
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        edges.append([u, v])
            if n > best_n:
                best_n = n
                best_edges = edges
            break
        
        # If alpha too high, try adding some edges back carefully
        # This is tricky - adding edges may create C5s
        # Try a few random edge additions and re-check
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v or (adj[u] >> v) & 1:
                continue
            # Tentatively add edge
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            # Check if still C5-free and alpha ok
            if not _has_c5_fast(adj, n) and not _has_independent_set(adj, n, 9):
                edges = []
                for uu in range(n):
                    for vv in range(uu + 1, n):
                        if (adj[uu] >> vv) & 1:
                            edges.append([uu, vv])
                if n > best_n:
                    best_n = n
                    best_edges = edges
                break
            else:
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
    
    if best_edges is not None:
        return best_n, best_edges
    return None


def _search_from_base(n_target, time_budget):
    """Start from base K4x8 and try to extend to n_target."""
    rng = random.Random(12345)
    deadline = time.time() + time_budget
    
    base_n, base_edges = _base_solution()
    n = n_target
    adj = [0] * n
    for u, v in base_edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    
    # Add random edges for new vertices
    for v in range(32, n):
        for u in range(32):
            if rng.random() < 0.3:
                adj[v] |= 1 << u
                adj[u] |= 1 << v
        for u in range(32, v):
            if rng.random() < 0.4:
                adj[v] |= 1 << u
                adj[u] |= 1 << v
    
    best_n = 0
    best_edges = None
    
    for iteration in range(30000):
        if time.time() > deadline - 5:
            break
        
        # Check current state
        if not _has_c5_fast(adj, n) and not _has_independent_set(adj, n, 9):
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        edges.append([u, v])
            if n > best_n:
                best_n = n
                best_edges = edges
            break
        
        # Pick random edge to flip
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        current = (adj[u] >> v) & 1
        if current:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
    
    if best_edges is not None:
        return best_n, best_edges
    return None


def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    # Guaranteed baseline: n=32
    base_n, base_edges = _base_solution()
    
    # Try multiple strategies
    strategies = [
        ('dense', _local_search),
        ('base_extend', _search_from_base),
    ]
    
    for n_try in [35, 34, 33]:
        elapsed = time.time() - t0
        remaining = _DEADLINE - elapsed - 10
        if remaining < 5:
            break
        
        for strat_name, strat_func in strategies:
            elapsed = time.time() - t0
            remaining = _DEADLINE - elapsed - 5
            if remaining < 3:
                break
            
            result = strat_func(n_try, remaining)
            if result is not None:
                rn, redges = result
                if rn > base_n:
                    base_n = rn
                    base_edges = redges
                break  # Found a solution for this n, move to next n
    
    # Final validation
    if base_edges is not None:
        adj = _build_adj(base_n, base_edges)
        if _has_c5_fast(adj, base_n) or _has_independent_set(adj, base_n, 9):
            # Fallback to base
            base_n, base_edges = _base_solution()
    
    return {'n': base_n, 'edges': base_edges}
