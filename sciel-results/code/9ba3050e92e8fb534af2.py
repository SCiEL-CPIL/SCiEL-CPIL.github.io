import time
import random
import numpy as np

N = 40
DEADLINE = 110  # Safety margin under 120s kill

def _creates_c6(adj, u, v):
    """
    Returns True iff adding edge (u, v) creates a C6.
    Path: u-a-b-c-d-v
    """
    nu = adj[u]
    nv = adj[v]

    # Iterate a in N(u), a != v
    a_mask = nu & ~(1 << v)
    while a_mask:
        lsb_a = a_mask & -a_mask
        a = lsb_a.bit_length() - 1
        a_mask ^= lsb_a

        # Iterate d in N(v), d not in {u, a}
        d_mask = nv & ~(1 << u) & ~(1 << a)
        while d_mask:
            lsb_d = d_mask & -d_mask
            d = lsb_d.bit_length() - 1
            d_mask ^= lsb_d

            forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << d)

            # b in N(a), c in N(d), b,c distinct from {u,v,a,d}
            b_mask = adj[a] & ~forbidden
            if not b_mask:
                continue
            c_mask = adj[d] & ~forbidden
            if not c_mask:
                continue

            # Any b in b_mask connected to any c in c_mask?
            bm = b_mask
            while bm:
                lsb_b = bm & -bm
                b = lsb_b.bit_length() - 1
                bm ^= lsb_b
                if adj[b] & c_mask:
                    return True

    return False

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

def _greedy_fill(adj, all_edges, rng):
    """Add as many edges as possible in random order."""
    order = all_edges[:]
    rng.shuffle(order)
    count = 0
    for u, v in order:
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, u, v):
            continue
        _add_edge(adj, u, v)
        count += 1
    return count

def _local_search(adj, all_edges, rng, deadline):
    """
    Local search with 1-swaps and 2-swaps.
    """
    best_count = _count_edges(adj)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try adding edges
        order = all_edges[:]
        rng.shuffle(order)
        for u, v in order:
            if time.time() > deadline: break
            if (adj[u] >> v) & 1: continue
            if _creates_c6(adj, u, v): continue
            _add_edge(adj, u, v)
            best_count += 1
            improved = True

        # 2. Try 1-swaps (remove e1, add e2)
        if time.time() > deadline: break
        current_edges = [(i, j) for i in range(N) for j in range(i+1, N) if (adj[i] >> j) & 1]
        rng.shuffle(current_edges)
        for e1 in current_edges:
            if time.time() > deadline: break
            u1, v1 = e1
            _remove_edge(adj, u1, v1)
            
            # Try to add a random edge
            found = False
            for _ in range(20): # Try a few random additions
                u2, v2 = rng.choice(all_edges)
                if (adj[u2] >> v2) & 1: continue
                if _creates_c6(adj, u2, v2): continue
                _add_edge(adj, u2, v2)
                found = True
                break
            
            if found:
                improved = True
                # Count might stay same, but we unlocked a state
                # We don't update best_count here as it's neutral, 
                # but next add pass might improve.
            else:
                _add_edge(adj, u1, v1)

        # 3. Try 2-swaps (remove e1, e2, add e3, e4)
        if time.time() > deadline: break
        current_edges = [(i, j) for i in range(N) for j in range(i+1, N) if (adj[i] >> j) & 1]
        rng.shuffle(current_edges)
        for i in range(len(current_edges)):
            if time.time() > deadline: break
            for j in range(i+1, min(i+10, len(current_edges))): # Limit pairs
                if time.time() > deadline: break
                e1 = current_edges[i]
                e2 = current_edges[j]
                u1, v1 = e1
                u2, v2 = e2
                
                _remove_edge(adj, u1, v1)
                _remove_edge(adj, u2, v2)
                
                found = False
                # Try to add 2 random edges
                for _ in range(10):
                    a1, b1 = rng.choice(all_edges)
                    if (adj[a1] >> b1) & 1: continue
                    if _creates_c6(adj, a1, b1): continue
                    _add_edge(adj, a1, b1)
                    
                    for _ in range(10):
                        a2, b2 = rng.choice(all_edges)
                        if (adj[a2] >> b2) & 1: continue
                        if _creates_c6(adj, a2, b2): continue
                        _add_edge(adj, a2, b2)
                        found = True
                        break
                    if found: break
                
                if found:
                    improved = True
                else:
                    _add_edge(adj, u1, v1)
                    _add_edge(adj, u2, v2)
                    
        # Update best
        cnt = _count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            if not improved: break # If we didn't make structural moves but count went up? Unlikely.
            
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

def _random_regular_seed(deg, rng):
    """
    Generate a random regular graph seed.
    """
    adj = [0] * N
    # Simple random regular graph generation (not perfect, but good enough for seed)
    # Use configuration model or just random edges and prune
    # For simplicity, let's just use a random graph with p=0.2 and prune C6s
    # Actually, let's just use the K5 seed or a random sparse graph
    # Let's try a random 6-regular graph
    # This is hard to do perfectly fast. Let's stick to K5 or random sparse.
    
    # Random sparse seed
    for i in range(N):
        for j in range(i+1, N):
            if rng.random() < 0.15:
                _add_edge(adj, i, j)
    # Prune C6s
    all_edges = _make_all_edges()
    for u, v in all_edges:
        if (adj[u] >> v) & 1:
            _remove_edge(adj, u, v)
            if _creates_c6(adj, u, v):
                continue # It was creating C6, so don't add back
            else:
                _add_edge(adj, u, v)
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    rng = random.Random(42)
    all_edges = _make_all_edges()

    best_adj = [0] * N
    best_count = 0

    def refresh_best(adj):
        nonlocal best_adj, best_count
        c = _count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]

    # Phase 1: K5 Seed
    adj = _k5_seed_adj()
    _greedy_fill(adj, all_edges, rng)
    adj, c = _local_search(adj, all_edges, rng, deadline)
    refresh_best(adj)

    # Phase 2: Random Sparse Seeds
    while time.time() < deadline:
        adj = _random_regular_seed(6, rng)
        _greedy_fill(adj, all_edges, rng)
        adj, c = _local_search(adj, all_edges, rng, deadline)
        refresh_best(adj)
        
        # If we are close to deadline, do a quick local search on best_adj
        if time.time() > deadline - 10:
            break

    # Final Polish on best_adj
    if time.time() < deadline:
        adj = best_adj[:]
        adj, c = _local_search(adj, all_edges, rng, deadline)
        refresh_best(adj)

    # Reconstruct edge list
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
