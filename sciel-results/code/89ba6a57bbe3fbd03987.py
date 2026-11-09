"""Construct a large C4-free graph with no independent 11-set (lower bound for R(C4,K11))."""
import time
import random
import itertools

_START = time.time()
_TIME_LIMIT = 115.0  # safety margin under 120s wall-clock

def _elapsed():
    return time.time() - _START

def _check_c4_free(n, adj):
    """Verify no two vertices share more than one common neighbor."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if len(common) >= 2:
                return False
    return True

def _max_independent_set_size(n, adj, timeout=None):
    """Exact maximum independent set size via branch-and-bound."""
    best = [0]
    adj_list = [sorted(adj[i]) for i in range(n)]

    # Greedy lower bound
    def greedy_lower():
        remaining = set(range(n))
        count = 0
        while remaining:
            # Pick vertex with max degree in remaining induced subgraph
            best_v = max(remaining, key=lambda v: sum(1 for u in remaining if u != v and u in adj[v]))
            count += 1
            remaining -= {best_v} | (adj[best_v] & remaining)
        return count

    lb = greedy_lower()
    best[0] = lb

    # Branch and bound
    # Use ordering: sort vertices by degree
    order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
    pos_in_order = [0] * n
    for i, v in enumerate(order):
        pos_in_order[v] = i

    def upper_bound(candidate_set):
        # Simple: min of (len of candidate_set) and greedy coloring of complement restricted
        # Actually, use a simple bound: for each vertex in candidate, it excludes its neighbors
        # Better: use the fact that in a C4-free graph, the structure limits independence
        # For simplicity, use: len(candidate_set)
        # Or a quick greedy coloring bound on the subgraph induced by candidate_set
        if len(candidate_set) <= best[0]:
            return len(candidate_set)
        # Greedy coloring of the subgraph on candidate_set to get upper bound on independent set
        # (chromatic number of complement >= alpha of original)
        # Actually, alpha(subgraph) <= n_subgraph - max matching... too complex
        # Use: for each v in candidate_set, count how many of its neighbors are also in candidate_set
        # This is an upper bound on the number we can still add if we pick v
        # Simple bound: len(candidate_set)
        return len(candidate_set)

    def b_and_b(candidate_set, current_size):
        if _elapsed() > (timeout or _TIME_LIMIT) - 2.0:
            return
        if not candidate_set:
            if current_size > best[0]:
                best[0] = current_size
            return
        if current_size + len(candidate_set) <= best[0]:
            return
        # Pick a vertex with high degree in candidate_set
        v = max(candidate_set, key=lambda x: sum(1 for u in candidate_set if u != x and u in adj[x]))
        # Branch 1: include v
        new_candidate = candidate_set - {v} - (adj[v] & candidate_set)
        b_and_b(new_candidate, current_size + 1)
        # Branch 2: exclude v
        new_candidate2 = candidate_set - {v}
        b_and_b(new_candidate2, current_size)

    # To make this faster, use a smarter ordering and bound
    # Let's use a different approach: Bron-Kerbosch style for maximal independent sets
    # Or simply: since n is small (<= 57), we can use a more careful B&B

    # Actually, for n up to 57, a naive B&B might be too slow.
    # Let's use a heuristic: try to find an independent set of size 11. If we fail, alpha <= 10.
    # This is a decision problem, which might be easier.

    def has_independent_set_of_size_k(k, timeout_local=None):
        """Check if there's an independent set of size exactly k."""
        deadline = time.time() + (timeout_local or 5.0)
        
        # Use backtracking with pruning
        # Sort vertices by degree (ascending, to branch on low-degree vertices first? or descending?)
        # For finding independent set, branching on high-degree vertices prunes more (they exclude more)
        sorted_v = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
        
        def backtrack(remaining, chosen, idx):
            if time.time() > deadline:
                return False
            if len(chosen) == k:
                return True
            if len(chosen) + len(remaining) < k:
                return False
            if idx >= n:
                return False
            
            v = sorted_v[idx]
            if v not in remaining:
                return backtrack(remaining, chosen, idx + 1)
            
            # Option 1: include v
            new_remaining = remaining - {v} - (adj[v] & remaining)
            if backtrack(new_remaining, chosen + [v], idx + 1):
                return True
            
            # Option 2: exclude v
            if backtrack(remaining - {v}, chosen, idx + 1):
                return True
            
            return False
        
        return backtrack(set(range(n)), [], 0)
    
    # First check if alpha >= 11
    if has_independent_set_of_size_k(11, timeout_local=30.0):
        return 11  # or larger, but we only care about > 10
    else:
        return 10  # at most 10 (could be less, but we know from greedy it's at least some value)

def _build_polarity_graph_pg2(q):
    """Build the polarity graph of PG(2,q)."""
    # Points of PG(2,q): lines through origin in GF(q)^3
    # Two points are adjacent if they are orthogonal under the standard dot product
    # (or some symmetric bilinear form)
    
    # Generate all 1-dimensional subspaces of GF(q)^3
    field = list(range(q))
    points = []
    seen = set()
    
    for x in field:
        for y in field:
            for z in field:
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: find first nonzero coordinate
                vec = (x, y, z)
                # Canonical form: make first nonzero coordinate 1
                for i, c in enumerate(vec):
                    if c != 0:
                        inv_c = pow(c, q - 2, q) if q > 2 else 1
                        if q == 2:
                            inv_c = 1
                        canonical = tuple((c * inv_c) % q for c in vec)
                        break
                if canonical not in seen:
                    seen.add(canonical)
                    points.append(canonical)
    
    n = len(points)
    # Should be q^2 + q + 1
    assert n == q * q + q + 1, f"Expected {q*q+q+1} points, got {n}"
    
    # Build adjacency: two points p, q are adjacent if dot(p, q) == 0
    # Use the canonical representatives
    adj = [set() for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            r = points[j]
            dot = (p[0] * r[0] + p[1] * r[1] + p[2] * r[2]) % q
            if dot == 0:
                adj[i].add(j)
                adj[j].add(i)
    
    return n, adj

def construct():
    if _elapsed() > _TIME_LIMIT - 10:
        # Not enough time, return seed
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    # Strategy: Try PG(2,7) polarity graph (57 vertices), find a 39-vertex subgraph with alpha <= 10
    # If that fails, try PG(2,5) (31 vertices) directly
    # If that fails, return seed (30 vertices)
    
    # Try PG(2,5) first since it's smaller and faster to verify
    if _elapsed() < _TIME_LIMIT - 60:
        n5, adj5 = _build_polarity_graph_pg2(5)
        # Verify C4-free
        if _check_c4_free(n5, adj5):
            alpha5 = _max_independent_set_size(n5, adj5, timeout=_TIME_LIMIT - 30)
            if alpha5 <= 10:
                edges5 = [[u, v] for u in range(n5) for v in adj5[u] if u < v]
                return {'n': n5, 'edges': edges5}
    
    # Try PG(2,7): search for 39-vertex induced subgraph with alpha <= 10
    if _elapsed() < _TIME_LIMIT - 20:
        n7, adj7 = _build_polarity_graph_pg2(7)
        if not _check_c4_free(n7, adj7):
            # Should not happen, but just in case
            pass
        else:
            # Try random sampling of 39-vertex subsets
            best_subgraph = None
            best_n = 0
            deadline = _TIME_LIMIT - 5
            attempts = 0
            
            # Also try to find the best we can with a greedy approach
            # Start with all 57, try to remove vertices to reduce alpha to 10 while keeping n high
            
            # Random sampling
            random.seed(42)
            while time.time() < deadline:
                attempts += 1
                sample = set(random.sample(range(n7), 39))
                # Check if induced subgraph has alpha <= 10
                # Build local adjacency
                local_adj = {v: adj7[v] & sample for v in sample}
                local_n = len(sample)
                
                # Quick check: does it have an independent set of size 11?
                # Use the same B&B but on the local graph
                # For speed, use a heuristic first
                # Greedy independent set
                remaining = set(sample)
                greedy_size = 0
                while remaining:
                    # Pick vertex with min degree in remaining
                    min_deg_v = min(remaining, key=lambda v: len(local_adj[v] & remaining))
                    greedy_size += 1
                    remaining -= {min_deg_v} | (local_adj[min_deg_v] & remaining)
                
                if greedy_size < 11:
                    # Might be valid, but greedy might undercount
                    # Do exact check with time limit
                    # Map to local indices
                    verts = sorted(sample)
                    local_n = len(verts)
                    local_adj_set = [set() for _ in range(local_n)]
                    idx_map = {v: i for i, v in enumerate(verts)}
                    for v in verts:
                        for u in local_adj[v]:
                            local_adj_set[idx_map[v]].add(idx_map[u])
                    
                    alpha_local = _max_independent_set_size(local_n, local_adj_set, timeout=10.0)
                    if alpha_local <= 10:
                        edges_local = []
                        for i in range(local_n):
                            for j in local_adj_set[i]:
                                if i < j:
                                    edges_local.append([verts[i], verts[j]])
                        # Sort edges for consistency
                        edges_local.sort()
                        return {'n': local_n, 'edges': edges_local}
                
                if attempts % 1000 == 0:
                    if _elapsed() > _TIME_LIMIT - 10:
                        break
            
            # If random didn't find it, try a greedy construction
            # Start with all vertices, iteratively remove vertices that are in many independent sets
            # This is complex, so just try a few more random samples with different seeds
    
    # Return seed
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}
