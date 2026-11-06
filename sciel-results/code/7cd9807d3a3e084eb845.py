import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118  # safety margin under 120s limit

    n = 40
    all_mask = (1 << n) - 1

    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_pairs = len(all_pairs)

    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) creates a C6. Returns True if safe."""
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)

        # We need path u-a-b-c-d-v with all distinct vertices
        # a in N(u)\{v}, d in N(v)\{u}
        # b in N(a)\{u,v,d}, c in N(d)\{u,v,a}
        # and b~c

        am = nu
        while am:
            lb_a = am & (-am)
            ai = lb_a.bit_length() - 1
            am ^= lb_a

            # d in N(v) \ {a}
            dm = nv & ~lb_a
            while dm:
                lb_d = dm & (-dm)
                di = lb_d.bit_length() - 1
                dm ^= lb_d

                # b candidates: neighbors of a, excluding u, v, a, d
                b_cand = adj[ai] & ~(1 << u) & ~(1 << v) & ~lb_a & ~lb_d
                # c candidates: neighbors of d, excluding u, v, a, d
                c_cand = adj[di] & ~(1 << u) & ~(1 << v) & ~lb_a & ~lb_d

                # Check if any b in b_cand is adjacent to some c in c_cand
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
        """Return one C6 as list of 6 edges, or None."""
        for u in range(n):
            nu = adj[u]
            # v must be non-adjacent to u
            nv_cand = all_mask & ~(1 << u) & ~nu
            v = nv_cand
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

    def greedy_add(adj, rng, max_time=None):
        """Try to add edges in random order, skipping those that create C6."""
        count = count_edges(adj)
        non_edges = []
        for u, v in all_pairs:
            if not ((adj[u] >> v) & 1):
                non_edges.append((u, v))
        rng.shuffle(non_edges)
        for u, v in non_edges:
            if max_time is not None and time.time() > max_time:
                break
            if not ((adj[u] >> v) & 1):
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    count += 1
        return adj, count

    def local_search(adj, rng, budget_time):
        """Local search with add and swap moves."""
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
        max_no_improve = 150

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
                    if count > best_count:
                        best_count = count
                        best_adj = adj[:]
                else:
                    no_improve += 1
            else:
                # Swap: remove one edge, add another
                if not existing or not non_existing:
                    continue
                idx_rem = rng.randint(0, len(existing) - 1)
                u1, v1 = existing[idx_rem]
                idx_add = rng.randint(0, len(non_existing) - 1)
                u2, v2 = non_existing[idx_add]

                # Remove u1-v1
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
                    # Reject
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    no_improve += 1

            if no_improve > max_no_improve:
                # Reset and try again
                no_improve = 0
                max_no_improve = int(max_no_improve * 1.2)
                if max_no_improve > 500:
                    max_no_improve = 500

        return best_adj, best_count

    def build_random_graph(rng, density):
        """Build a random graph with given edge density."""
        adj = [0] * n
        for u, v in all_pairs:
            if rng.random() < density:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def prune_c6s(adj, rng, max_iter=500):
        """Remove edges to break C6s. Returns (adj, num_removed)."""
        removed = 0
        for _ in range(max_iter):
            if time.time() > deadline - 30:
                break
            c6 = find_one_c6(adj)
            if c6 is None:
                break
            # Remove a random edge from the C6
            idx = rng.randint(0, 5)
            u, v = c6[idx]
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            removed += 1
        return adj, removed

    # ---------- Main search ----------
    best_adj = None
    best_count = 0

    random.seed(42)
    rng = random.Random(12345)

    # Try multiple random initializations with different densities
    densities = [0.15, 0.2, 0.25, 0.3, 0.35]
    
    for density in densities:
        if time.time() > deadline - 50:
            break
        for restart in range(5):
            if time.time() > deadline - 45:
                break
            rng2 = random.Random(hash((density, restart)) % (2**32))
            adj = build_random_graph(rng2, density)
            adj, _ = prune_c6s(adj, rng2)
            adj, cnt = greedy_add(adj, rng2, max_time=deadline - 40)
            if cnt > best_count:
                best_count = cnt
                best_adj = adj[:]

    # Also try structured constructions
    # Construction 1: Complete bipartite-like with random perturbation
    for trial in range(3):
        if time.time() > deadline - 40:
            break
        rng3 = random.Random(1000 + trial)
        adj = [0] * n
        # Split into 4 groups of 10
        for g1 in range(4):
            for g2 in range(g1 + 1, 4):
                base1 = g1 * 10
                base2 = g2 * 10
                for a in range(10):
                    for b in range(10):
                        if rng3.random() < 0.25:
                            u = base1 + a
                            v = base2 + b
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
        adj, _ = prune_c6s(adj, rng3)
        adj, cnt = greedy_add(adj, rng3, max_time=deadline - 35)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]

    # If no good solution, fallback
    if best_adj is None or best_count < 70:
        best_adj = [0] * n
        # Simple construction: complete graph on 14 vertices, plus random edges
        for u in range(14):
            for v in range(u + 1, 14):
                best_adj[u] |= (1 << v)
                best_adj[v] |= (1 << u)
        rng4 = random.Random(777)
        for u in range(14, 40):
            for v in range(u + 1, 40):
                if rng4.random() < 0.3:
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
        best_adj, _ = prune_c6s(best_adj, rng4)
        best_adj, best_count = greedy_add(best_adj, rng4, max_time=deadline - 30)

    # Final local search from best solution
    remaining_time = deadline - time.time() - 3
    if remaining_time > 5 and best_adj is not None:
        best_adj, best_count = local_search(best_adj, random.Random(999), time.time() + remaining_time)

    # Final greedy add pass
    if time.time() < deadline - 2 and best_adj is not None:
        best_adj, best_count = greedy_add(best_adj, random.Random(777), max_time=deadline - 1)

    if best_adj is None:
        best_adj = [0] * n
        best_count = 0

    result = edges_from_adj(best_adj)
    return result
