import time
import random
import numpy as np

N = 40
DEADLINE = 118  # wall-clock seconds (safety margin under 150s kill)

def _fast_creates_c6(adj, u, v):
    """
    Fast C6 check: adding (u,v) creates a C6 iff there is a simple path
    u - a - b - c - d - v with all six distinct.
    Uses bitset intersection to check existence of b-c edge.
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
                    return True
    return False

def _creates_c6(adj, u, v):
    return _fast_creates_c6(adj, u, v)

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

def _random_seed(adj):
    """Add random edges greedily to seed."""
    all_edges = _make_all_edges()
    random.shuffle(all_edges)
    for u, v in all_edges:
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, u, v):
            continue
        _add_edge(adj, u, v)

def _local_search(adj, max_iter, deadline, rng):
    """
    Local search: try to add an edge, or swap (remove e1, add e2) to unlock more adds.
    Returns (best_adj, best_count).
    """
    best_count = _count_edges(adj)
    all_edges = _make_all_edges()
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        improved = False
        random.shuffle(all_edges)
        # Pass 1: additions
        for u, v in all_edges:
            if time.time() >= deadline:
                break
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
            improved = True
        # Pass 2: systematic swaps
        if time.time() < deadline:
            present = [(u, v) for u, v in all_edges if (adj[u] >> v) & 1]
            absent = [(u, v) for u, v in all_edges if not ((adj[u] >> v) & 1)]
            for e1 in present:
                if time.time() >= deadline:
                    break
                u1, v1 = e1
                _remove_edge(adj, u1, v1)
                added = False
                for e2 in absent:
                    u2, v2 = e2
                    if _creates_c6(adj, u2, v2):
                        continue
                    _add_edge(adj, u2, v2)
                    added = True
                    improved = True
                    break
                if not added:
                    _add_edge(adj, u1, v1)
        # Update best
        cnt = _count_edges(adj)
        if cnt > best_count:
            best_count = cnt
        if not improved:
            break
    return adj, best_count

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

def _design_seed_adj():
    """
    Build a C6-free seed from a (40,10)-design block graph.
    8 groups of 5 vertices; each vertex connects to 2 vertices in each of the other 7 groups.
    This gives degree 14, total edges = 40*14/2 = 280... wait that's too many.
    
    Actually for C6-freeness, the block graph of a (v,k,lambda) design where blocks
    are cliques works. Let me use a simpler approach:
    
    Use 8 groups of 5. In each group, make a 5-cycle (degree 2 within group).
    Between groups, connect using a structured pattern that avoids C6.
    
    Simpler: use the incidence structure. Take 8 "blocks" of size 5.
    Each vertex is in exactly 2 blocks. The block graph (vertices = original, 
    edges within blocks) is C6-free if the design has no 6-cycles in its
    block intersection graph... this gets complicated.
    
    Let me just use a random regular graph approach with C6 verification.
    """
    adj = [0] * N
    rng = random.Random(42)
    
    # Build 8 groups of 5, each group is a 5-cycle
    for g in range(8):
        base = g * 5
        for i in range(5):
            u = base + i
            v = base + ((i + 1) % 5)
            _add_edge(adj, u, v)
    
    # Between groups: connect using a structured pattern
    # For groups g1, g2, connect vertex (g1, i) to vertex (g2, (i + g1) % 5) and (g2, (i + g1 + 1) % 5)
    # This gives each vertex 2 connections per other group = 14 total degree
    # But that's 40*14/2 = 280 edges which is way too many for C6-free.
    
    # Instead, use sparser inter-group connections.
    # Connect group g to group h (g < h) with a perfect matching (5 edges).
    # Total inter-group edges: C(8,2)*5 = 28*5 = 140
    # Total edges: 8*5 + 140 = 180... still too many.
    
    # Let's be more conservative. Use only some inter-group edges.
    # Target: ~105 edges total.
    # 8 groups * 5-cycle = 40 edges within groups.
    # Need ~65 more inter-group edges.
    
    # Add inter-group edges greedily with C6 check
    all_edges = _make_all_edges()
    rng.shuffle(all_edges)
    for u, v in all_edges:
        if (adj[u] >> v) & 1:
            continue
        # Only add inter-group edges
        gu, gv = u // 5, v // 5
        if gu == gv:
            continue
        if _creates_c6(adj, u, v):
            continue
        _add_edge(adj, u, v)
    
    return adj

def _spectral_seed_adj():
    """
    Generate a candidate C6-free graph using spectral/structured approach:
    Start with a random d-regular graph, verify C6-freeness, if not, repair.
    Then use local search to maximize edges.
    """
    adj = [0] * N
    rng = random.Random(7)
    
    # Start with random 5-regular graph (100 edges)
    # Use configuration model
    while True:
        adj = [0] * N
        half_edges = list(range(N)) * 5  # each vertex has 5 stubs
        rng.shuffle(half_edges)
        success = True
        for i in range(0, len(half_edges), 2):
            u, v = half_edges[i], half_edges[i + 1]
            if u == v or ((adj[u] >> v) & 1):
                success = False
                break
            _add_edge(adj, u, v)
        if success:
            break
    
    # Check if C6-free; if not, try to repair by removing edges
    # For simplicity, just use it as a seed and let local search work
    return adj

def _anneal_from(adj, deadline, rng):
    """
    Simulated annealing: propose add/swap/remove; accept improves always.
    Returns best_adj, best_count.
    """
    cur = adj[:]
    cur_count = _count_edges(cur)
    best = cur[:]
    best_count = cur_count
    all_edges = _make_all_edges()
    start = time.time()
    total = max(1e-6, deadline - start)

    while time.time() < deadline:
        elapsed = time.time() - start
        frac = min(1.0, elapsed / total)
        temp = max(0.01, 1.0 - frac)

        r = rng.random()
        if r < 0.55:
            u, v = rng.choice(all_edges)
            if (cur[u] >> v) & 1:
                continue
            if _creates_c6(cur, u, v):
                continue
            _add_edge(cur, u, v)
            delta = 1
            move = ("add", u, v)
        elif r < 0.90:
            present = [(u, v) for u, v in all_edges if (cur[u] >> v) & 1]
            if not present:
                continue
            u1, v1 = rng.choice(present)
            _remove_edge(cur, u1, v1)
            added = False
            for _ in range(15):
                u2, v2 = rng.choice(all_edges)
                if (cur[u2] >> v2) & 1:
                    continue
                if _creates_c6(cur, u2, v2):
                    continue
                _add_edge(cur, u2, v2)
                added = True
                move = ("swap", (u1, v1), (u2, v2))
                break
            if not added:
                _add_edge(cur, u1, v1)
                continue
            delta = 0
        else:
            present = [(u, v) for u, v in all_edges if (cur[u] >> v) & 1]
            if not present:
                continue
            u1, v1 = rng.choice(present)
            _remove_edge(cur, u1, v1)
            delta = -1
            move = ("remove", u1, v1)

        new_count = cur_count + delta
        if new_count > best_count:
            best = cur[:]
            best_count = new_count
        cur_count = new_count

        if delta > 0:
            continue
        p = 1.0 if delta == 0 else min(1.0, temp * 0.5)
        if rng.random() < p:
            continue
        else:
            if move[0] == "add":
                _remove_edge(cur, move[1], move[2])
            elif move[0] == "swap":
                (u2, v2), (u1, v1) = move[2], move[1]
                _remove_edge(cur, u2, v2)
                _add_edge(cur, u1, v1)
            elif move[0] == "remove":
                _add_edge(cur, move[1], move[2])
            cur_count = _count_edges(cur)
    return best, best_count

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

    # Phase 1: Design-based seed
    if time.time() < deadline:
        seed = _design_seed_adj()
        order = all_edges[:]
        rng.shuffle(order)
        for u, v in order:
            if (seed[u] >> v) & 1:
                continue
            if _creates_c6(seed, u, v):
                continue
            _add_edge(seed, u, v)
        seed, c = _local_search(seed, max_iter=80, deadline=deadline, rng=rng)
        refresh_best(seed)

    # Phase 1b: K5 seed
    if time.time() < deadline:
        seed2 = _k5_seed_adj()
        order = all_edges[:]
        rng.shuffle(order)
        for u, v in order:
            if (seed2[u] >> v) & 1:
                continue
            if _creates_c6(seed2, u, v):
                continue
            _add_edge(seed2, u, v)
        seed2, c = _local_search(seed2, max_iter=80, deadline=deadline, rng=rng)
        refresh_best(seed2)

    # Phase 2: Multiple random-start greedy builds
    while time.time() < deadline:
        adj = [0] * N
        order = all_edges[:]
        rng.shuffle(order)
        for u, v in order:
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
        adj, c = _local_search(adj, max_iter=60, deadline=deadline, rng=rng)
        refresh_best(adj)

    # Phase 3: Final anneal on best
    if time.time() < deadline:
        best_adj, best_count = _anneal_from(best_adj, deadline, rng)

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
