import time
import random
import numpy as np

N = 40
DEADLINE = 115  # wall-clock seconds (safety margin under 150s kill)

def _creates_c6_and_path(adj, u, v):
    """
    Returns (True, path) if adding (u, v) creates a C6, else (False, None).
    path is a list of 6 vertices [u, a, b, c, d, v].
    """
    nu = adj[u]
    nv = adj[v]

    a_mask = nu & ~(1 << v)
    while a_mask:
        lsb_a = a_mask & -a_mask
        a = lsb_a.bit_length() - 1
        a_mask ^= lsb_a

        d_mask = nv & ~(1 << u) & ~(1 << a)
        while d_mask:
            lsb_d = d_mask & -d_mask
            d = lsb_d.bit_length() - 1
            d_mask ^= lsb_d

            forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << d)
            b_mask = adj[a] & ~forbidden
            if not b_mask:
                continue
            c_mask = adj[d] & ~forbidden
            if not c_mask:
                continue

            bm = b_mask
            while bm:
                lsb_b = bm & -bm
                b = lsb_b.bit_length() - 1
                bm ^= lsb_b
                if adj[b] & c_mask:
                    # Find a specific c
                    cm = adj[b] & c_mask
                    lsb_c = cm & -cm
                    c = lsb_c.bit_length() - 1
                    return True, [u, a, b, c, d, v]

    return False, None

def _creates_c6(adj, u, v):
    """
    Returns True iff adding edge (u, v) creates a C6.
    """
    res, _ = _creates_c6_and_path(adj, u, v)
    return res

def _count_edges(adj):
    total = 0
    for i in range(N):
        total += bin(adj[i]).count("1")
    return total // 2

def _add_edge(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def _make_all_edges():
    return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _k5_seed_adj():
    """
    Build a C6-free seed from 8 disjoint K5's (40 vertices).
    Returns adj list.
    """
    adj = [0] * N
    for k in range(8):
        base = k * 5
        for i in range(5):
            for j in range(i + 1, 5):
                u = base + i
                v = base + j
                _add_edge(adj, u, v)
    return adj

def _greedy_fill(adj, all_edges, deadline):
    """
    Greedily add edges in the given order if they don't create C6.
    """
    for u, v in all_edges:
        if time.time() > deadline:
            break
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, u, v):
            continue
        _add_edge(adj, u, v)

def _directed_local_search(adj, deadline):
    """
    Local search using path-breaking swaps.
    1. Identify all blocked edges.
    2. For each blocked edge, find a C6 path that blocks it.
    3. Try to swap an edge from that path with the blocked edge.
    4. Repeat until no improvements or time runs out.
    """
    all_edges = _make_all_edges()
    best_count = _count_edges(adj)
    
    while time.time() < deadline:
        improved = False
        
        # Collect blocked edges
        blocked = []
        for u, v in all_edges:
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                blocked.append((u, v))
        
        if not blocked:
            break
            
        # Shuffle to avoid bias
        random.shuffle(blocked)
        
        for u, v in blocked:
            if time.time() > deadline:
                break
            
            # Get a specific C6 path that blocks (u, v)
            _, path = _creates_c6_and_path(adj, u, v)
            if not path:
                continue
                
            # path: [u, a, b, c, d, v]
            # Edges in path: (u,a), (a,b), (b,c), (c,d), (d,v)
            # We want to remove one of these edges and add (u,v).
            # Removing (u,a) or (d,v) might be less disruptive to the rest of the graph?
            # Actually, removing any edge in the path breaks the cycle.
            # Let's try removing each edge in the path (excluding u,v endpoints logic is handled by adjacency)
            
            path_edges = [(path[0], path[1]), (path[1], path[2]), (path[2], path[3]), (path[3], path[4]), (path[4], path[5])]
            
            # Shuffle path edges to try different breaks
            random.shuffle(path_edges)
            
            for e1 in path_edges:
                u1, v1 = e1
                # Ensure we are removing an existing edge
                if not ((adj[u1] >> v1) & 1):
                    continue
                
                # Temporarily remove e1
                _remove_edge(adj, u1, v1)
                
                # Try to add (u, v)
                if not _creates_c6(adj, u, v):
                    _add_edge(adj, u, v)
                    # Check if we can add any other edges that were previously blocked
                    # (Optional: quick pass)
                    # For now, just accept the swap. Edge count is unchanged, but structure changed.
                    # This might allow future additions.
                    improved = True
                    break
                else:
                    # Revert
                    _add_edge(adj, u1, v1)
                    # If we couldn't add (u,v) even after removing e1, revert and try next e1
                    # Note: It's possible that removing e1 creates a DIFFERENT C6 for (u,v)?
                    # No, removing an edge cannot create a new C6. It can only remove C6s.
                    # So if (u,v) was blocked by path P, and we remove an edge from P, (u,v) MUST be valid.
                    # Wait, is it possible that (u,v) is blocked by MULTIPLE paths, and removing one edge from one path still leaves another path?
                    # Yes. So _creates_c6 might still return True.
                    # In that case, we revert.
                    pass
        
        # After processing blocked edges, try to add any newly unblocked edges
        if improved:
            # Quick greedy pass
            # We don't have the original 'all_edges' order handy in a way that's efficient to re-scan fully every time?
            # We can just scan all non-edges.
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                if (adj[u] >> v) & 1:
                    continue
                if _creates_c6(adj, u, v):
                    continue
                _add_edge(adj, u, v)
                improved = True # Mark improved to continue loop
        
        if not improved:
            # If no swaps worked and no adds worked, we are stuck.
            # Try random restarts or break?
            # Let's break to avoid infinite loop if stuck.
            break

    return adj

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    rng = random.Random(12345)
    all_edges = _make_all_edges()

    best_adj = [0] * N
    best_count = 0

    def refresh_best(adj):
        nonlocal best_adj, best_count
        c = _count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]

    # Phase 1: K5 seed + Greedy + Directed Local Search
    seed = _k5_seed_adj()
    order = all_edges[:]
    rng.shuffle(order)
    _greedy_fill(seed, order, deadline)
    refresh_best(seed)
    
    # Apply directed local search
    seed = _directed_local_search(seed, deadline)
    refresh_best(seed)

    # Phase 2: Multiple random-start greedy builds with directed local search
    # Use remaining time for restarts
    while time.time() < deadline:
        adj = [0] * N
        order = all_edges[:]
        rng.shuffle(order)
        _greedy_fill(adj, order, deadline)
        
        if time.time() > deadline:
            break
            
        adj = _directed_local_search(adj, deadline)
        refresh_best(adj)

    # Reconstruct edge list from best_adj
    edges = []
    for i in range(N):
        m = best_adj[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if j > i:
                edges.append([i, j])
            m ^= lsb
    return edges

if __name__ == "__main__":
    construct()
