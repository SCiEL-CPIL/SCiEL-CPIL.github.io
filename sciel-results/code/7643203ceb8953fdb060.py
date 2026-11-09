"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _creates_c6_fast(adj, n, u, v):
    """Fast check if adding edge (u,v) creates a C6.
    
    A C6 is formed if there exists a simple path of length 5 from u to v
    in the current graph (not using edge u-v since it's not there yet).
    Path: u - a - b - c - d - v
    
    We enumerate paths of length 5 from u to v.
    """
    # For each neighbor a of u (a != v), for each neighbor b of a (b != u, b != v),
    # for each neighbor c of b (c != a, c != u, c != v),
    # for each neighbor d of c (d != b, d != u, d != v, d != a),
    # check if d is adjacent to v (d != v)
    
    adj_u = adj[u]
    adj_v = adj[v]
    
    for a in adj_u:
        if a == v:
            continue
        adj_a = adj[a]
        for b in adj_a:
            if b == u or b == v or b == a:
                continue
            adj_b = adj[b]
            for c in adj_b:
                if c == u or c == v or c == a or c == b:
                    continue
                adj_c = adj[c]
                for d in adj_c:
                    if d == u or d == v or d == a or d == b or d == c:
                        continue
                    if d in adj_v:
                        return True
    return False

def _greedy_build(n, seed, deadline):
    """Build a C6-free graph greedily with a given seed."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    edges = set()
    
    # Start with 8 disjoint K5s for a good base
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = 5 * k + a, 5 * k + b
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
    
    # Generate candidate edges and sort by a heuristic
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in edges:
                # Heuristic: prefer edges between vertices with lower degrees
                candidates.append((len(adj[u]) + len(adj[v]), u, v))
    
    # Shuffle within same degree sum to add randomness
    candidates.sort(key=lambda x: (x[0], rng.random()))
    
    for _, u, v in candidates:
        if time.time() > deadline:
            break
        if not _creates_c6_fast(adj, n, u, v):
            edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
    
    return edges, adj

def _local_search(edges, adj, n, deadline, seed):
    """Local search: try swap moves (remove one edge, add another)."""
    rng = random.Random(seed)
    current_count = len(edges)
    best_count = current_count
    best_edges = edges.copy()
    best_adj = [s.copy() for s in adj]
    
    # Precompute all possible edges
    all_edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.add((u, v))
    
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random edge to remove
        if not edges:
            break
        removed = rng.choice(list(edges))
        ru, rv = removed
        
        # Try to find a replacement edge
        added = None
        # Try random non-edges
        for _ in range(200):
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) in edges:
                continue
            if (u, v) == removed:
                continue
            if not _creates_c6_fast(adj, n, u, v):
                added = (u, v)
                break
        
        if added is None:
            no_improve += 1
            continue
        
        au, av = added
        
        # Perform the swap
        edges.discard(removed)
        adj[ru].discard(rv)
        adj[rv].discard(ru)
        
        edges.add(added)
        adj[au].add(av)
        adj[av].add(au)
        
        new_count = len(edges)
        
        if new_count >= best_count:
            if new_count > best_count:
                best_count = new_count
                best_edges = edges.copy()
                best_adj = [s.copy() for s in adj]
                no_improve = 0
            else:
                # Same count, keep going
                pass
        else:
            # Revert
            edges.discard(added)
            adj[au].discard(av)
            adj[av].discard(au)
            edges.add(removed)
            adj[ru].add(rv)
            adj[rv].add(ru)
            no_improve += 1
    
    return best_edges, best_adj, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    n = 40
    best_overall = 0
    best_edges_overall = set()
    
    # Multi-start local search
    num_starts = 10
    time_per_start = (deadline - start_time) / num_starts
    
    for s in range(num_starts):
        if time.time() > deadline:
            break
        
        seed_start = 42 + s * 1000
        edges, adj = _greedy_build(n, seed_start, deadline - 5)
        
        if len(edges) > best_overall:
            best_overall = len(edges)
            best_edges_overall = edges.copy()
        
        # Local search with remaining time
        ls_deadline = min(deadline - 2, start_time + time_per_start * (s + 1))
        edges, adj, count = _local_search(edges, adj, n, ls_deadline, seed_start + 1)
        
        if count > best_overall:
            best_overall = count
            best_edges_overall = edges.copy()
    
    # Final verification: make sure no C6 exists
    # Quick check by trying to find any C6
    final_adj = [set() for _ in range(n)]
    for u, v in best_edges_overall:
        final_adj[u].add(v)
        final_adj[v].add(u)
    
    # Verify C6-free by checking all potential 6-cycles
    # For each pair of non-adjacent vertices, check if there's a path of length 5
    # Actually, just verify by checking if _creates_c6_fast would return True for any edge
    # But since we built it incrementally, it should be fine.
    # Do a spot check:
    has_c6 = False
    for u in range(n):
        if has_c6:
            break
        for v in final_adj[u]:
            if v <= u:
                continue
            # Temporarily remove edge and check
            final_adj[u].discard(v)
            final_adj[v].discard(u)
            if _creates_c6_fast(final_adj, n, u, v):
                has_c6 = True
            final_adj[u].add(v)
            final_adj[v].add(u)
    
    if has_c6:
        # This shouldn't happen, but fallback to just the K5s
        fallback = set()
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u, v = 5 * k + a, 5 * k + b
                    fallback.add((u, v))
        return [list(e) for e in fallback]
    
    return [list(e) for e in best_edges_overall]

# EVOLVE-BLOCK-END
