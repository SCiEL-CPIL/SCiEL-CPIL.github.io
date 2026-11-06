import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118  # safety margin under 120s limit

    n = 40
    G = 8  # number of groups
    S = 5  # vertices per group
    all_mask = (1 << n) - 1

    # Group assignment: vertex v belongs to group v // S
    def group_of(v):
        return v // S

    def vertices_in_group(g):
        base = g * S
        return list(range(base, base + S))

    # Precompute all inter-group pairs (no intra-group edges allowed)
    inter_pairs = []
    for g1 in range(G):
        for g2 in range(g1 + 1, G):
            for v1 in vertices_in_group(g1):
                for v2 in vertices_in_group(g2):
                    inter_pairs.append((v1, v2))

    # ---------- C6 detection ----------
    def find_one_c6(adj):
        """Find one C6 in the graph. Returns list of 6 vertices or None."""
        for u in range(n):
            nu = adj[u]
            non_adj = all_mask & ~(1 << u) & ~nu
            v = non_adj
            while v:
                lb_v = v & (-v)
                vi = lb_v.bit_length() - 1
                v ^= lb_v

                # Path u-a-b-c-d-v
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
                        while bm:
                            lb_b = bm & (-bm)
                            bi = lb_b.bit_length() - 1
                            bm ^= lb_b
                            cm = adj[bi] & c_cand
                            if cm:
                                lb_c = cm & (-cm)
                                ci = lb_c.bit_length() - 1
                                return [u, ai, bi, ci, di, vi]
        return None

    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) creates a C6. Returns True if safe."""
        # Quick check: if u and v are in the same group, they can't be adjacent
        # (independent set constraint), so this should never be called
        if group_of(u) == group_of(v):
            return False

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

    # ---------- Independent-set model construction ----------
    # Each group is an independent set. Edges only between groups.
    # We represent the graph as a 8x8 matrix of 5x5 bitmasks for inter-group edges.
    
    def random_adjacency(rng, density_range=(0.3, 0.6)):
        """Generate a random adjacency respecting the independent-set structure."""
        adj = [0] * n
        lo, hi = density_range
        for g1 in range(G):
            for g2 in range(g1 + 1, G):
                p = rng.uniform(lo, hi)
                base1 = g1 * S
                base2 = g2 * S
                for a in range(S):
                    for b in range(S):
                        if rng.random() < p:
                            u = base1 + a
                            v = base2 + b
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
        return adj

    def prune_c6s(adj, rng, max_iter=500):
        """Remove edges to break C6s."""
        removed = 0
        for _ in range(max_iter):
            if time.time() > deadline - 20:
                break
            c6 = find_one_c6(adj)
            if c6 is None:
                break
            # Remove a random edge from the C6
            idx = rng.randint(0, 5)
            u, v = c6[idx], c6[(idx + 1) % 6]
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            removed += 1
        return adj, removed

    def greedy_add_all(adj, rng):
        """Try to add all possible inter-group edges that don't create C6."""
        count = count_edges(adj)
        # Try all inter-group non-edges in random order
        candidates = []
        for u, v in inter_pairs:
            if not ((adj[u] >> v) & 1):
                candidates.append((u, v))
        rng.shuffle(candidates)
        
        for u, v in candidates:
            if time.time() > deadline - 5:
                break
            if not ((adj[u] >> v) & 1) and can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
        return adj, count

    def local_search_is(adj, rng, budget_time):
        """Local search optimized for independent-set structure."""
        count = count_edges(adj)
        best_adj = adj[:]
        best_count = count

        # Build candidate lists
        non_edges = []
        for u, v in inter_pairs:
            if not ((adj[u] >> v) & 1):
                non_edges.append((u, v))

        existing = []
        for u, v in inter_pairs:
            if (adj[u] >> v) & 1:
                existing.append((u, v))

        no_improve = 0
        max_no_improve = 500

        while time.time() < budget_time:
            if not non_edges:
                break

            r = rng.random()
            if r < 0.5:
                # Add move
                idx = rng.randint(0, len(non_edges) - 1)
                u, v = non_edges[idx]
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    count += 1
                    non_edges.pop(idx)
                    existing.append((u, v))
                    no_improve = 0
                    if count > best_count:
                        best_count = count
                        best_adj = adj[:]
                else:
                    no_improve += 1
            elif r < 0.8:
                # Swap: remove one, add another
                if not existing or not non_edges:
                    no_improve += 1
                    continue
                idx_rem = rng.randint(0, len(existing) - 1)
                u1, v1 = existing[idx_rem]
                idx_add = rng.randint(0, len(non_edges) - 1)
                u2, v2 = non_edges[idx_add]

                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)

                if can_add_edge(adj, u2, v2):
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    existing.pop(idx_rem)
                    non_edges.append((u1, v1))
                    non_edges.pop(idx_add)
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
                # 2-opt: remove two, add two
                if len(existing) < 2 or len(non_edges) < 2:
                    no_improve += 1
                    continue
                idx_rem1 = rng.randint(0, len(existing) - 1)
                u1, v1 = existing[idx_rem1]
                idx_rem2 = rng.randint(0, len(existing) - 1)
                if idx_rem2 == idx_rem1:
                    no_improve += 1
                    continue
                u2, v2 = existing[idx_rem2]

                idx_add1 = rng.randint(0, len(non_edges) - 1)
                wa, xa = non_edges[idx_add1]
                idx_add2 = rng.randint(0, len(non_edges) - 1)
                if idx_add2 == idx_add1:
                    no_improve += 1
                    continue
                wb, xb = non_edges[idx_add2]

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
                            non_edges.pop(idx_add2)
                            non_edges.pop(idx_add1)
                        else:
                            non_edges.pop(idx_add1)
                            non_edges.pop(idx_add2)

                        non_edges.append((u1, v1))
                        non_edges.append((u2, v2))

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
                # Escape: try several greedy adds
                for _ in range(20):
                    if not non_edges:
                        break
                    idx = rng.randint(0, len(non_edges) - 1)
                    u, v = non_edges[idx]
                    if can_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        count += 1
                        non_edges.pop(idx)
                        existing.append((u, v))
                        if count > best_count:
                            best_count = count
                            best_adj = adj[:]
                    else:
                        break

        return best_adj, best_count

    # ---------- Main search ----------
    best_adj = None
    best_count = 0

    random.seed(42)
    rng = random.Random(12345)

    # Phase 1: Generate diverse starting points with independent-set structure
    num_starts = 15
    for i in range(num_starts):
        if time.time() > deadline - 60:
            break
        rng2 = random.Random(i * 1000 + 42)
        # Vary the density
        density_lo = 0.2 + 0.3 * (i % 5) / 5.0
        density_hi = density_lo + 0.2 + 0.2 * (i % 3) / 3.0
        adj = random_adjacency(rng2, (density_lo, density_hi))
        adj, _ = prune_c6s(adj, rng2)
        adj, cnt = greedy_add_all(adj, rng2)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]

    # Fallback if nothing found
    if best_adj is None or best_count < 50:
        best_adj = [0] * n
        best_count = 0

    # Phase 2: Local search
    remaining = deadline - time.time() - 8
    if remaining > 15:
        ls_budget = time.time() + remaining * 0.75
        best_adj, best_count = local_search_is(best_adj, random.Random(777), ls_budget)

    # Phase 3: Final greedy add
    if time.time() < deadline - 3:
        best_adj, best_count = greedy_add_all(best_adj, random.Random(999))

    # Phase 4: Final short local search
    if time.time() < deadline - 2:
        best_adj, best_count = local_search_is(best_adj, random.Random(888), deadline - 2)

    if not best_adj:
        best_adj = [0] * n
        best_count = 0

    result = edges_from_adj(best_adj)
    return result
