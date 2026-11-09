"""Combined C6-free graph maximizer: bitmask C6 check, greedy + systematic local search + polish."""
import time
import random
import numpy as np

N = 40
DEADLINE = 115  # safety margin under 150s kill

def _creates_c6(adj, u, v):
    """
    Returns True iff adding edge (u, v) creates a C6.
    A C6 is formed iff there exists a simple path of length 5 from u to v:
        u - a - b - c - d - v
    adj: list of int bitmasks, adj[i] has bit j set iff edge (i, j) present.
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

            # Check if any b in b_mask is connected to any c in c_mask
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

def _k5_seed_adj():
    """Build a C6-free seed from 8 disjoint K5's (40 vertices)."""
    adj = [0] * N
    for k in range(8):
        base = k * 5
        for i in range(5):
            for j in range(i + 1, 5):
                u = base + i
                v = base + j
                _add_edge(adj, u, v)
    return adj

def _greedy_build(adj, seed, deadline):
    """Greedy build: add edges in heuristic order (low degree sum first)."""
    rng = random.Random(seed)
    all_edges = _make_all_edges()
    # Filter to non-edges only
    non_edges = [(u, v) for u, v in all_edges if not ((adj[u] >> v) & 1)]
    # Sort by degree sum heuristic
    non_edges.sort(key=lambda e: (bin(adj[e[0]]).count("1") + bin(adj[e[1]]).count("1"), rng.random()))
    for u, v in non_edges:
        if time.time() > deadline:
            break
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, u, v):
            continue
        _add_edge(adj, u, v)
    return adj

def _systematic_local_search(adj, deadline, rng):
    """
    Systematic local search:
    1. Repeatedly try to add any valid non-edge (sorted by degree sum).
    2. Try systematic swaps: remove each existing edge, try to add a valid non-edge.
    3. Repeat until no improvement or deadline.
    """
    best_adj = adj[:]
    best_count = _count_edges(adj)
    all_edges = _make_all_edges()
    no_improve = 0
    max_no_improve = 20

    while time.time() < deadline and no_improve < max_no_improve:
        improved = False

        # Pass 1: Try to add any valid non-edge, sorted by degree sum
        non_edges = [(u, v) for u, v in all_edges if not ((adj[u] >> v) & 1)]
        non_edges.sort(key=lambda e: bin(adj[e[0]]).count("1") + bin(adj[e[1]]).count("1"))
        for u, v in non_edges:
            if time.time() >= deadline:
                break
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
            improved = True

        # Pass 2: Systematic swaps
        if time.time() < deadline:
            existing_edges = [(u, v) for u, v in all_edges if (adj[u] >> v) & 1]
            # Shuffle for variety
            rng.shuffle(existing_edges)
            for u1, v1 in existing_edges:
                if time.time() >= deadline:
                    break
                if not ((adj[u1] >> v1) & 1):
                    continue
                # Temporarily remove
                _remove_edge(adj, u1, v1)
                # Try to find a valid non-edge to add
                non_edges_now = [(u, v) for u, v in all_edges if not ((adj[u] >> v) & 1)]
                non_edges_now.sort(key=lambda e: bin(adj[e[0]]).count("1") + bin(adj[e[1]]).count("1"))
                found = False
                for u2, v2 in non_edges_now:
                    if time.time() >= deadline:
                        break
                    if _creates_c6(adj, u2, v2):
                        continue
                    _add_edge(adj, u2, v2)
                    found = True
                    improved = True
                    break
                if not found:
                    # Revert
                    _add_edge(adj, u1, v1)

        cnt = _count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
            no_improve = 0
        else:
            no_improve += 1

    return best_adj, best_count

def _random_local_search(adj, deadline, rng):
    """
    Random local search with simulated annealing-like acceptance.
    """
    cur = adj[:]
    cur_count = _count_edges(cur)
    best = cur[:]
    best_count = cur_count
    all_edges = _make_all_edges()

    start = time.time()
    total = max(1e-6, deadline - start)
    temp = 1.0
    min_temp = 0.01

    while time.time() < deadline:
        elapsed = time.time() - start
        frac = min(1.0, elapsed / total)
        temp = max(min_temp, 1.0 - frac)

        r = rng.random()
        if r < 0.6:
            # Add a random edge
            u, v = rng.choice(all_edges)
            if (cur[u] >> v) & 1:
                continue
            if _creates_c6(cur, u, v):
                continue
            _add_edge(cur, u, v)
            delta = 1
            move = ("add", u, v)
        elif r < 0.9:
            # Swap
            e1 = rng.choice(all_edges)
            u1, v1 = e1
            if not ((cur[u1] >> v1) & 1):
                continue
            _remove_edge(cur, u1, v1)
            added = False
            for _ in range(20):
                e2 = rng.choice(all_edges)
                u2, v2 = e2
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
            # Remove a random edge
            e1 = rng.choice(all_edges)
            u1, v1 = e1
            if not ((cur[u1] >> v1) & 1):
                continue
            _remove_edge(cur, u1, v1)
            delta = -1
            move = ("remove", u1, v1)

        new_count = cur_count + delta
        if new_count > best_count:
            best = cur[:]
            best_count = new_count
        cur_count = new_count

        # Accept/reject
        if delta > 0:
            continue
        p = 1.0 if delta == 0 else min(1.0, temp * 0.4)
        if rng.random() < p:
            continue
        else:
            # Revert
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

def _final_polish(adj, deadline):
    """
    Final polish: systematically try all non-edges sorted by degree sum.
    """
    all_edges = _make_all_edges()
    improved = True
    while improved and time.time() < deadline:
        improved = False
        non_edges = [(u, v) for u, v in all_edges if not ((adj[u] >> v) & 1)]
        non_edges.sort(key=lambda e: bin(adj[e[0]]).count("1") + bin(adj[e[1]]).count("1"))
        for u, v in non_edges:
            if time.time() >= deadline:
                break
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
            improved = True
    return adj

def _verify_c6_free(adj):
    """Verify that the graph is C6-free by checking each edge."""
    for u in range(N):
        for v in range(u + 1, N):
            if not ((adj[u] >> v) & 1):
                continue
            # Temporarily remove edge and check
            _remove_edge(adj, u, v)
            if _creates_c6(adj, u, v):
                _add_edge(adj, u, v)
                return False
            _add_edge(adj, u, v)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    rng = random.Random(12345)

    best_adj = [0] * N
    best_count = 0

    # Multi-start with different seeds
    num_starts = 5
    time_per_start = (deadline - start_time - 5) / num_starts

    for s in range(num_starts):
        if time.time() > deadline - 5:
            break

        seed = 42 + s * 1000
        adj = _k5_seed_adj()

        # Greedy build
        greedy_deadline = start_time + time_per_start * (s + 0.3)
        adj = _greedy_build(adj, seed, greedy_deadline)

        # Systematic local search
        sys_deadline = start_time + time_per_start * (s + 0.7)
        adj, count = _systematic_local_search(adj, sys_deadline, rng)

        # Random local search
        rand_deadline = start_time + time_per_start * (s + 1.0)
        adj, count = _random_local_search(adj, rand_deadline, rng)

        # Final polish
        polish_deadline = min(deadline - 2, start_time + time_per_start * (s + 1.0) + 2)
        adj = _final_polish(adj, polish_deadline)
        count = _count_edges(adj)

        if count > best_count:
            best_count = count
            best_adj = adj[:]

    # Final polish on best
    final_deadline = deadline - 1
    best_adj = _final_polish(best_adj, final_deadline)
    best_count = _count_edges(best_adj)

    # Verify C6-free
    if not _verify_c6_free(best_adj):
        # Fallback to just K5s
        fallback_adj = _k5_seed_adj()
        return [[u, v] for u in range(N) for v in range(u + 1, N) if (fallback_adj[u] >> v) & 1]

    return [[u, v] for u in range(N) for v in range(u + 1, N) if (best_adj[u] >> v) & 1]
