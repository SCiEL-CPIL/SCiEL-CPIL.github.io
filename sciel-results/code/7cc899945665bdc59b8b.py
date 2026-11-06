import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118  # safety margin under 120s limit

    n = 40
    all_mask = (1 << n) - 1

    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_pairs = len(all_pairs)

    # ---------- Fast C6 detection and edge-addability ----------
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) creates a C6. Returns True if safe."""
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        if nu == 0 or nv == 0:
            return True

        am = nu
        while am:
            lb_a = am & (-am)
            ai = lb_a.bit_length() - 1
            am ^= lb_a

            dm = nv & ~lb_a
            while dm:
                lb_d = dm & (-dm)
                di = lb_d.bit_length() - 1
                dm ^= lb_d

                b_cand = adj[ai] & ~(1 << u) & ~(1 << v) & ~lb_a & ~lb_d
                if b_cand == 0:
                    continue

                c_cand = adj[di] & ~(1 << u) & ~(1 << v) & ~lb_a & ~lb_d
                if c_cand == 0:
                    continue

                bm = b_cand
                while bm:
                    lb_b = bm & (-bm)
                    bi = lb_b.bit_length() - 1
                    bm ^= lb_b
                    if adj[bi] & c_cand:
                        return False
        return True

    def count_edges(adj):
        c = 0
        for i in range(n):
            c += adj[i].bit_count()
        return c // 2

    def edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def find_one_c6(adj):
        """Return one C6 as list of 6 edges [(u0,u1),...], or None."""
        for u in range(n):
            nu = adj[u]
            nv_cand = all_mask & ~(1 << u) & ~nu
            v = nv_cand
            while v:
                lb_v = v & (-v)
                vi = lb_v.bit_length() - 1
                v ^= lb_v

                am = nu
                while am:
                    lb_a = am & (-am)
                    ai = lb_a.bit_length() - 1
                    am ^= lb_a

                    dm = adj[vi] & ~lb_a
                    while dm:
                        lb_d = dm & (-dm)
                        di = lb_d.bit_length() - 1
                        dm ^= lb_d

                        b_cand = adj[ai] & ~(1 << u) & ~lb_v & ~lb_a & ~lb_d
                        c_cand = adj[di] & ~(1 << u) & ~lb_v & ~lb_a & ~lb_d

                        bm = b_cand
                        found = None
                        while bm:
                            lb_b = bm & (-bm)
                            bi = lb_b.bit_length() - 1
                            bm ^= lb_b
                            cm = adj[bi] & c_cand
                            if cm:
                                lb_c = cm & (-cm)
                                ci = lb_c.bit_length() - 1
                                found = (bi, ci)
                                break
                        if found:
                            bi, ci = found
                            return [(u, ai), (ai, bi), (bi, ci), (ci, di), (di, vi), (vi, u)]
        return None

    # ---------- Construction helpers ----------
    def build_from_params(rng, intra_p, inter_p):
        adj = [0] * n
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    if rng.random() < intra_p[k]:
                        u = base + a
                        v = base + b
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        for k1 in range(8):
            for k2 in range(k1 + 1, 8):
                p = inter_p[k1][k2]
                base1 = 5 * k1
                base2 = 5 * k2
                for a in range(5):
                    for b in range(5):
                        if rng.random() < p:
                            u = base1 + a
                            v = base2 + b
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
        return adj

    def prune_c6s(adj, rng, max_iter=300):
        """Remove edges to break C6s. Returns (adj, num_removed)."""
        removed = 0
        for _ in range(max_iter):
            if time.time() > deadline - 25:
                break
            c6 = find_one_c6(adj)
            if c6 is None:
                break
            idx = rng.randint(0, 5)
            u, v = c6[idx]
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            removed += 1
        return adj, removed

    def greedy_add(adj, rng):
        """Try to add edges in random order, skipping those that create C6."""
        count = count_edges(adj)
        non_edges = []
        for u, v in all_pairs:
            if not ((adj[u] >> v) & 1):
                non_edges.append((u, v))
        rng.shuffle(non_edges)
        for u, v in non_edges:
            if time.time() > deadline - 8:
                break
            if not ((adj[u] >> v) & 1):
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    count += 1
        return adj, count

    def local_search(adj, rng, budget_time):
        """Local search with add, swap, and 2-opt moves."""
        count = count_edges(adj)
        best_adj = adj[:]
        best_count = count

        existing = []
        non_existing = []
        for u, v in all_pairs:
            if (adj[u] >> v) & 1:
                existing.append((u, v))
            else:
                non_existing.append((u, v))

        no_improve = 0
        max_no_improve = 300

        while time.time() < budget_time:
            if not non_existing:
                break

            r = rng.random()
            if r < 0.45:
                idx = rng.randint(0, len(non_existing) - 1)
                u, v = non_existing[idx]
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    count += 1
                    non_existing.pop(idx)
                    existing.append((u, v))
                    no_improve = 0
                    if count > best_count:
                        best_count = count
                        best_adj = adj[:]
                else:
                    no_improve += 1
            elif r < 0.75:
                if not existing or not non_existing:
                    no_improve += 1
                    continue
                idx_rem = rng.randint(0, len(existing) - 1)
                u1, v1 = existing[idx_rem]
                idx_add = rng.randint(0, len(non_existing) - 1)
                u2, v2 = non_existing[idx_add]

                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)

                if can_add_edge(adj, u2, v2):
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)

                    existing.pop(idx_rem)
                    non_existing.append((u1, v1))
                    non_existing.pop(idx_add)
                    existing.append((u2, v2))
                    no_improve = 0
                    if count > best_count:
                        best_count = count
                        best_adj = adj[:]
                else:
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    no_improve += 1
            else:
                if len(existing) < 2 or len(non_existing) < 2:
                    no_improve += 1
                    continue
                idx_rem1 = rng.randint(0, len(existing) - 1)
                u1, v1 = existing[idx_rem1]
                idx_rem2 = rng.randint(0, len(existing) - 1)
                if idx_rem2 == idx_rem1:
                    no_improve += 1
                    continue
                u2, v2 = existing[idx_rem2]

                idx_add1 = rng.randint(0, len(non_existing) - 1)
                wa, xa = non_existing[idx_add1]
                idx_add2 = rng.randint(0, len(non_existing) - 1)
                if idx_add2 == idx_add1:
                    no_improve += 1
                    continue
                wb, xb = non_existing[idx_add2]

                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                adj[u2] &= ~(1 << v2)
                adj[v2] &= ~(1 << u2)

                ok1 = can_add_edge(adj, wa, xa)
                if ok1:
                    adj[wa] |= (1 << xa)
                    adj[xa] |= (1 << wa)
                    ok2 = can_add_edge(adj, wb, xb)
                    if ok2:
                        adj[wb] |= (1 << xb)
                        adj[xb] |= (1 << wb)

                        if idx_rem1 < idx_rem2:
                            existing.pop(idx_rem2)
                            existing.pop(idx_rem1)
                        else:
                            existing.pop(idx_rem1)
                            existing.pop(idx_rem2)

                        existing.append((wa, xa))
                        existing.append((wb, xb))

                        if idx_add1 < idx_add2:
                            non_existing.pop(idx_add2)
                            non_existing.pop(idx_add1)
                        else:
                            non_existing.pop(idx_add1)
                            non_existing.pop(idx_add2)

                        non_existing.append((u1, v1))
                        non_existing.append((u2, v2))

                        no_improve = 0
                        if count > best_count:
                            best_count = count
                            best_adj = adj[:]
                    else:
                        adj[wa] &= ~(1 << xa)
                        adj[xa] &= ~(1 << wa)
                        adj[u1] |= (1 << v1)
                        adj[v1] |= (1 << u1)
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        no_improve += 1
                else:
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    no_improve += 1

            if no_improve > max_no_improve:
                no_improve = 0
                for _ in range(10):
                    if not non_existing:
                        break
                    idx = rng.randint(0, len(non_existing) - 1)
                    u, v = non_existing[idx]
                    if can_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        count += 1
                        non_existing.pop(idx)
                        existing.append((u, v))
                        if count > best_count:
                            best_count = count
                            best_adj = adj[:]
                    else:
                        break

        return best_adj, best_count

    # ---------- Main search ----------
    best_adj = [0] * n
    best_count = 0

    # Helper to create K5 seed
    def make_k5_seed():
        adj = [0] * n
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    u = base + a
                    v = base + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    # Initial construction
    best_adj = make_k5_seed()
    best_count = count_edges(best_adj)
    
    # Greedy add on initial seed
    if time.time() < deadline - 15:
        best_adj, best_count = greedy_add(best_adj, random.Random(1))

    # Local search on initial seed
    if time.time() < deadline - 10:
        ls_time = deadline - 10
        best_adj, best_count = local_search(best_adj, random.Random(2), ls_time)

    # Diversified restarts: K5 seed + random sparse inter-cluster edges
    for restart in range(8):
        if time.time() > deadline - 20:
            break
        
        adj = make_k5_seed()
        rng_r = random.Random(100 + restart)
        
        # Add random inter-cluster edges with low probability, checking C6
        for k1 in range(8):
            for k2 in range(k1 + 1, 8):
                base1 = 5 * k1
                base2 = 5 * k2
                for a in range(5):
                    for b in range(5):
                        if rng_r.random() < 0.08:
                            u = base1 + a
                            v = base2 + b
                            if can_add_edge(adj, u, v):
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
        
        # Greedy add
        if time.time() < deadline - 10:
            adj, cnt = greedy_add(adj, rng_r)
            
            if cnt > best_count:
                best_count = cnt
                best_adj = adj[:]
                
            # Local search
            if time.time() < deadline - 5:
                ls_time = deadline - 5
                adj, cnt = local_search(adj, rng_r, ls_time)
                if cnt > best_count:
                    best_count = cnt
                    best_adj = adj[:]

    # Final polish on best found
    if time.time() < deadline - 2:
        best_adj, best_count = local_search(best_adj, random.Random(999), deadline - 2)

    result = edges_from_adj(best_adj)
    return result
