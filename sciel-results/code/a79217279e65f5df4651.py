"""Construct a K3,3-free graph on 20 vertices maximizing edges."""
import time
import random

def construct():
    start = time.time()
    deadline = start + 140  # safety margin before 150s kill

    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    N = len(all_edges)  # 190

    def popcount(x):
        return bin(x).count('1')

    def is_valid(adj, n):
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    common = adj[i] & adj[j] & adj[k]
                    valid = common & ~((1 << i) | (1 << j) | (1 << k))
                    if popcount(valid) >= 3:
                        return False
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def fast_check(adj, u, v):
        """Check if adding edge (u,v) creates a K3,3.
        A new K3,3 must include the new edge, so it must have u and v on opposite sides,
        with 2 other vertices on u's side and 2 other on v's side, all 4 cross-adjacent.
        """
        # Side A = {u, a, b}, Side B = {v, c, d}
        # Need: a~v, b~v, a~c, a~d, b~c, b~d, and u~c, u~d, v~a, v~b (v~a,v~b given by a,b in N(v))
        # Actually a,b must be neighbors of v (since they connect to v on side B)
        # c,d must be neighbors of u (since they connect to u on side A)
        Nv = adj[v]
        Nu = adj[u]
        # a, b in Nv, not u, not v; c, d in Nu, not u, not v
        # Check all pairs (a,b) from Nv and (c,d) from Nu
        # a,b must be common neighbors of v (they are, by being in Nv)
        # Need a~c, a~d, b~c, b~d and u~c, u~d (c,d in Nu so u~c,u~d given) and v~a, v~b (given)
        # So just need: a,b in Nv; c,d in Nu; and a~c, a~d, b~c, b~d
        # Also a != c, a != d, b != c, b != d (disjoint sides)
        cand_a = Nv & ~((1 << u) | (1 << v))
        cand_c = Nu & ~((1 << u) | (1 << v))
        # iterate pairs from cand_a and cand_c
        # For each pair (a,b) in cand_a and (c,d) in cand_c, check a~c,a~d,b~c,b~d
        # But we also need a != c etc. Since a in Nv and c in Nu, and u,v not in either,
        # a and c could be equal only if a in Nv ∩ Nu. If a == c, then sides not disjoint.
        # Let's just check the 4 adjacencies and disjointness.
        # To be efficient, for each a in cand_a, b in cand_a (a<b), check:
        #   need 2 vertices c,d in cand_c such that a~c,a~d,b~c,b~d and c,d != a,b
        # Let's precompute for speed
        # For each pair (a,b):
        #   common_c = cand_c & adj[a] & adj[b]  # c,d must be neighbors of both a and b
        #   also c,d must not be a or b
        #   if popcount(common_c) >= 2: violation
        # Wait, c,d must be in cand_c (neighbors of u) AND neighbors of a AND neighbors of b.
        # And c,d distinct from a,b.
        # So common_c = cand_c & adj[a] & adj[b] & ~((1<<a)|(1<<b))
        # if popcount(common_c) >= 2 -> K3,3
        # Iterate over pairs (a,b) in cand_a
        # cand_a is a bitmask; iterate set bits
        ca = cand_a
        # collect list of a's
        alist = []
        x = ca
        while x:
            lb = x & (-x)
            alist.append(lb.bit_length() - 1)
            x -= lb
        for ii in range(len(alist)):
            a = alist[ii]
            for jj in range(ii + 1, len(alist)):
                b = alist[jj]
                common_c = cand_c & adj[a] & adj[b] & ~((1 << a) | (1 << b))
                if popcount(common_c) >= 2:
                    return False
        return True

    def count_edges(adj, n):
        total = 0
        for i in range(n):
            total += popcount(adj[i])
        return total // 2

    # Greedy construction with random edge order
    def greedy_build(order, adj0=None):
        adj = adj0[:] if adj0 else [0] * n
        edges = count_edges(adj, n)
        for (u, v) in order:
            if adj[u] & (1 << v):
                continue
            add_edge(adj, u, v)
            if fast_check(adj, u, v):
                edges += 1
            else:
                remove_edge(adj, u, v)
        return adj, edges

    best_adj = [0] * n
    best_edges = 0

    random.seed(42)

    # Multiple greedy trials with different random orderings
    for trial in range(80):
        if time.time() > deadline - 15:
            break
        order = all_edges[:]
        random.shuffle(order)
        adj, edges = greedy_build(order)

        # Local search: try adding any missing edge
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for (u, v) in all_edges:
                if time.time() > deadline - 10:
                    break
                if adj[u] & (1 << v):
                    continue
                add_edge(adj, u, v)
                if fast_check(adj, u, v):
                    edges += 1
                    improved = True
                else:
                    remove_edge(adj, u, v)

        # Simulated annealing: try swaps to escape local optima
        if time.time() < deadline - 10:
            T = 5.0
            for sa_iter in range(2000):
                if time.time() > deadline - 10:
                    break
                T *= 0.995
                # pick a random existing edge to remove
                existing = [(u, v) for (u, v) in all_edges if adj[u] & (1 << v)]
                if not existing:
                    break
                (u1, v1) = random.choice(existing)
                # pick a random missing edge to add
                missing = [(u, v) for (u, v) in all_edges if not (adj[u] & (1 << v))]
                if not missing:
                    break
                (u2, v2) = random.choice(missing)
                # apply swap
                remove_edge(adj, u1, v1)
                add_edge(adj, u2, v2)
                # check validity: removing an edge can't create K3,3, only adding can
                # but we removed one and added one; need full check? Removing safe, adding:
                # the new edge (u2,v2) is the only risk, but removal might have been needed
                # Actually removing (u1,v1) then adding (u2,v2): check if adding (u2,v2) valid
                # given current adj (after removal)
                if fast_check(adj, u2, v2):
                    new_edges = count_edges(adj, n)
                    delta_e = new_edges - edges
                    if delta_e >= 0 or random.random() < pow(2.718281828, delta_e / max(T, 0.1)):
                        edges = new_edges
                    else:
                        # revert
                        remove_edge(adj, u2, v2)
                        add_edge(adj, u1, v1)
                else:
                    # revert
                    remove_edge(adj, u2, v2)
                    add_edge(adj, u1, v1)

        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]

    # Final local search on best
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for (u, v) in all_edges:
            if time.time() > deadline - 2:
                break
            if best_adj[u] & (1 << v):
                continue
            add_edge(best_adj, u, v)
            if fast_check(best_adj, u, v):
                best_edges += 1
                improved = True
            else:
                remove_edge(best_adj, u, v)

    # Verify
    if not is_valid(best_adj, n):
        # fallback: rebuild conservatively
        best_adj = [0] * n
        best_edges = 0
        for (u, v) in all_edges:
            add_edge(best_adj, u, v)
            if fast_check(best_adj, u, v):
                best_edges += 1
            else:
                remove_edge(best_adj, u, v)

    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edge_list.append([i, j])

    return edge_list
