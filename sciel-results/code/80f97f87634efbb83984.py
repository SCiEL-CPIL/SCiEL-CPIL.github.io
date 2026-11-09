import time
import random
import numpy as np

N = 40
DEADLINE = 115  # safety margin under 120s limit

def _creates_c6(adj, u, v):
    """
    Returns True iff adding edge (u, v) creates a C6.
    A C6 through edge (u,v) is u-a-b-c-d-v (or u-d-c-b-a-v), all distinct.
    We check: exists a in N(u)\{v}, d in N(v)\{u}, a!=d,
              b in N(a)\{u,v,d}, c in N(d)\{u,v,a},
              b != c, and c in N(b) (i.e., b-c edge exists).
    """
    nu = adj[u]
    nv = adj[v]
    u_bit = 1 << u
    v_bit = 1 << v

    a_mask = nu & ~v_bit
    while a_mask:
        lsb_a = a_mask & -a_mask
        a = lsb_a.bit_length() - 1
        a_mask ^= lsb_a
        a_bit = 1 << a

        d_mask = nv & ~u_bit & ~a_bit
        while d_mask:
            lsb_d = d_mask & -d_mask
            d = lsb_d.bit_length() - 1
            d_mask ^= lsb_d

            # b in N(a) \ {u, v, d}, c in N(d) \ {u, v, a}
            b_mask = adj[a] & ~u_bit & ~v_bit & ~(1 << d)
            if not b_mask:
                continue
            c_mask = adj[d] & ~u_bit & ~v_bit & ~a_bit
            if not c_mask:
                continue

            # Check if any b in b_mask is adjacent to any c in c_mask, b != c
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

def _greedy_from_order(adj, order):
    count = 0
    for u, v in order:
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, u, v):
            continue
        _add_edge(adj, u, v)
        count += 1
    return count

def _local_search(adj, max_iter, deadline, rng):
    best_count = _count_edges(adj)
    all_edges = _make_all_edges()
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        improved = False
        random.shuffle(all_edges)
        for u, v in all_edges:
            if time.time() >= deadline:
                break
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
            improved = True
        
        # Edge-swap: remove one edge, try to add two edges to net +1
        for _ in range(3):
            if time.time() >= deadline:
                break
            random.shuffle(all_edges)
            for e1 in all_edges:
                if time.time() >= deadline:
                    break
                u1, v1 = e1
                if not ((adj[u1] >> v1) & 1):
                    continue
                _remove_edge(adj, u1, v1)
                added = 0
                for _t in range(12):
                    if time.time() >= deadline:
                        break
                    u2, v2 = rng.choice(all_edges)
                    if (adj[u2] >> v2) & 1:
                        continue
                    if _creates_c6(adj, u2, v2):
                        continue
                    _add_edge(adj, u2, v2)
                    added += 1
                    improved = True
                    if added == 2:
                        break
                if added < 1:
                    _add_edge(adj, u1, v1)
        
        cnt = _count_edges(adj)
        if cnt > best_count:
            best_count = cnt
        if not improved:
            break
    return adj, best_count

def _k5_seed_adj():
    adj = [0] * N
    for k in range(8):
        base = k * 5
        for i in range(5):
            for j in range(i + 1, 5):
                u = base + i
                v = base + j
                _add_edge(adj, u, v)
    return adj

def _perturb(adj, rng, k=3):
    edges = []
    for i in range(N):
        m = adj[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if j > i:
                edges.append((i, j))
            m ^= lsb
    
    if not edges:
        return adj[:]
        
    new_adj = adj[:]
    for _ in range(k):
        if not edges: break
        idx = rng.randrange(len(edges))
        u, v = edges[idx]
        edges.pop(idx)
        _remove_edge(new_adj, u, v)
    return new_adj

def _greedy_fill(adj, all_edges, rng):
    # Multiple passes with different orderings to find more edges
    for _pass in range(3):
        order = all_edges[:]
        rng.shuffle(order)
        for u, v in order:
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
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

    # Phase 1: K5 seed + local search
    seed = _k5_seed_adj()
    order = all_edges[:]
    rng.shuffle(order)
    _greedy_from_order(seed, order)
    seed, c = _local_search(seed, max_iter=200, deadline=deadline, rng=rng)
    refresh_best(seed)

    # Phase 2: Iterated Local Search with varying perturbation strength
    while time.time() < deadline:
        k = rng.randint(1, 6)
        perturbed = _perturb(best_adj, rng, k)
        perturbed = _greedy_fill(perturbed, all_edges, rng)
        perturbed, c = _local_search(perturbed, max_iter=150, deadline=deadline, rng=rng)
        refresh_best(perturbed)

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
