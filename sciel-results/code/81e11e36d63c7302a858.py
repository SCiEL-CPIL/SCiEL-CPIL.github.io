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
        # Exclude u,v from neighbor sets
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)

        # We need: u - a - b - c - d - v with all distinct, a in N(u), d in N(v)
        # b in N(a)\{u,v,d}, c in N(d)\{u,v,a}, and b~c.

        # Iterate a in N(u)
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
                # Union of neighbors of b in b_cand, intersect with c_cand
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

    def greedy_add(adj, rng):
        """Try to add edges in random order, skipping those that create C6."""
        count = count_edges(adj)
        # Build list of non-edges
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
        """Local search with add and swap moves."""
        count = count_edges(adj)
        best_adj = adj[:]
        best_count = count

        # Maintain edge lists
        existing = []
        non_existing = []
        for u, v in all_pairs:
            if (adj[u] >> v) & 1:
                existing.append((u, v))
            else:
                non_existing.append((u, v))

        T = 5.0
        cooling = 0.995
        min_T = 0.5
        no_improve = 0

        while time.time() < budget_time:
            if not non_existing:
                break

            r = rng.random()
            if r < 0.55:
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

            T *= cooling
            if T < min_T:
                T = min_T

            if no_improve > 200 and T <= min_T:
                # Restart temperature occasionally
                no_improve = 0

        return best_adj, best_count

    def build_group_greedy(rng):
        """
        Construct graph by partitioning 40 vertices into 8 groups of 5.
        Treat groups as independent sets initially.
        Iteratively add inter-group edges that do not form a C6.
        """
        adj = [0] * n
        groups = [[5 * k + i for i in range(5)] for k in range(8)]
        
        # Generate all inter-group edges
        inter_edges = []
        for k1 in range(8):
            for k2 in range(k1 + 1, 8):
                for u in groups[k1]:
                    for v in groups[k2]:
                        inter_edges.append((u, v))
        
        rng.shuffle(inter_edges)
        
        # Add inter-group edges greedily
        for u, v in inter_edges:
            if time.time() > deadline - 30:
                break
            if not ((adj[u] >> v) & 1):
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
        return adj

    # ---------- Main search ----------
    best_adj = None
    best_count = 0

    random.seed(12345)
    rng = random.Random(54321)

    # Try multiple restarts of the group-based greedy construction
    for restart in range(10):
        if time.time() > deadline - 40:
            break
        
        rng2 = random.Random(1000 + restart)
        adj = build_group_greedy(rng2)
        
        # After inter-group edges, try to add intra-group edges via greedy_add
        adj, cnt = greedy_add(adj, rng2)
        
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]

    # If no good solution yet, fallback to K5 base (though group greedy should be better)
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

    # Final local search from best solution
    remaining_time = deadline - time.time() - 3
    if remaining_time > 5:
        best_adj, best_count = local_search(best_adj, random.Random(999), time.time() + remaining_time)

    # Final greedy add pass
    if time.time() < deadline - 2:
        best_adj, best_count = greedy_add(best_adj, random.Random(777))

    if not best_adj:
        best_adj = [0] * n
        best_count = 0

    result = edges_from_adj(best_adj)
    return result
