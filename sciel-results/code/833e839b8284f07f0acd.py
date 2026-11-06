import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin under 120s limit

    n = 40
    all_mask = (1 << n) - 1

    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_pairs = len(all_pairs)

    # ---------- Fast C6 detection and edge-addability ----------
    
    def can_add_edge(adj, u, v):
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)

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
                c_cand = adj[di] & ~(1 << u) & ~(1 << v) & ~lb_a & ~lb_d

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

    def build_sparse_graph(rng, p):
        adj = [0] * n
        for u, v in all_pairs:
            if rng.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def prune_c6s(adj, rng, max_iter=50):
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

        T = 5.0
        cooling = 0.99
        min_T = 0.1
        no_improve = 0
        kick_counter = 0

        while time.time() < budget_time:
            if not non_existing:
                break

            r = rng.random()
            if r < 0.6:
                # Add move
                idx = rng.randint(0, len(non_existing) - 1)
                u, v = non_existing[idx]
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    count += 1
                    non_existing.pop(idx)
                    existing.append((u, v))
                    no_improve = 0
                    kick_counter = 0
                    if count > best_count:
                        best_count = count
                        best_adj = adj[:]
                else:
                    no_improve += 1
            else:
                # Swap or Kick
                if no_improve > 50:
                    # Kick: remove a random edge to escape local minimum
                    if existing:
                        idx_rem = rng.randint(0, len(existing) - 1)
                        u1, v1 = existing[idx_rem]
                        adj[u1] &= ~(1 << v1)
                        adj[v1] &= ~(1 << u1)
                        count -= 1
                        existing.pop(idx_rem)
                        non_existing.append((u1, v1))
                        no_improve = 0
                        kick_counter += 1
                        if kick_counter > 5:
                            # Reset kick counter to prevent infinite kicking
                            kick_counter = 0
                else:
                    # Swap: remove one edge, add another
                    if not existing or not non_existing:
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
                        kick_counter = 0
                        if count > best_count:
                            best_count = count
                            best_adj = adj[:]
                    else:
                        adj[u1] |= (1 << v1)
                        adj[v1] |= (1 << u1)
                        no_improve += 1

            T *= cooling
            if T < min_T:
                T = min_T

        return best_adj, best_count

    best_adj = None
    best_count = 0

    rng_master = random.Random(42)

    # Try multiple sparse initializations
    for p_val in [0.1, 0.15, 0.2, 0.25]:
        if time.time() > deadline - 30:
            break
        for restart in range(5):
            if time.time() > deadline - 25:
                break
            rng = random.Random(rng_master.randint(0, 1000000))
            adj = build_sparse_graph(rng, p_val)
            # Light pruning
            adj, _ = prune_c6s(adj, rng, max_iter=20)
            adj, cnt = greedy_add(adj, rng)
            
            if cnt > best_count:
                best_count = cnt
                best_adj = adj[:]

    # If we didn't find a good one, fallback
    if best_adj is None or best_count < 80:
        best_adj = [0] * n
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    u = base + a
                    v = base + b
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
        best_count = count_edges(best_adj)

    # Final local search
    remaining_time = deadline - time.time() - 2
    if remaining_time > 5:
        best_adj, best_count = local_search(best_adj, random.Random(999), time.time() + remaining_time)

    # Final greedy add pass
    if time.time() < deadline - 1:
        best_adj, best_count = greedy_add(best_adj, random.Random(777))

    if not best_adj:
        best_adj = [0] * n
        best_count = 0

    result = edges_from_adj(best_adj)
    return result
