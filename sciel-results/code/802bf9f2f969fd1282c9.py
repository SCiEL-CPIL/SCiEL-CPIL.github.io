import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit

    n = 18

    # Precompute all subsets of size 12 as bitmasks
    subsets_12 = []
    for combo in combinations(range(n), 12):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        subsets_12.append(mask)

    # Precompute subsets of size 11
    subsets_11 = []
    for combo in combinations(range(n), 11):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        subsets_11.append(mask)

    def is_triangle_free(mask, adj):
        """Check if the subgraph induced by 'mask' is triangle-free."""
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            nu = adj[u] & mask
            v_mask = nu
            while v_mask:
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                if (adj[u] & adj[v] & mask) != 0:
                    return False
        return True

    def has_triangle_free_size(adj, size, subsets):
        """Check if there exists a triangle-free induced subgraph of given size."""
        for mask in subsets:
            if is_triangle_free(mask, adj):
                return True
        return False

    def max_triangle_free_size(adj):
        """Find the exact maximum size of a triangle-free induced subgraph."""
        # Check from size 12 down to smaller sizes
        if has_triangle_free_size(adj, 12, subsets_12):
            return 12
        if has_triangle_free_size(adj, 11, subsets_11):
            return 11
        # For sizes < 11, use a simpler check: try all subsets of that size
        # But this is expensive, so we'll use a greedy estimate for sizes < 11
        # Actually, let's just check size 10 with a limited approach
        # For now, return 10 if no size 11 found (conservative)
        return 10

    def generate_random_max_k4_free(rng):
        adj = [0] * n
        edges_to_try = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(edges_to_try)

        for u, v in edges_to_try:
            common = adj[u] & adj[v]
            has_edge_in_common = False
            c_mask = common
            while c_mask:
                lsb = c_mask & -c_mask
                w = lsb.bit_length() - 1
                c_mask ^= lsb
                if (adj[w] & common) != 0:
                    has_edge_in_common = True
                    break
            if not has_edge_in_common:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

        return adj

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    def is_k4_free(adj):
        """Check if graph is K4-free using bitmasks."""
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    continue
                for k in range(j + 1, n):
                    if not (adj[i] & (1 << k)) or not (adj[j] & (1 << k)):
                        continue
                    # Check if there's a vertex l > k that forms K4 with i,j,k
                    candidates = adj[i] & adj[j] & adj[k]
                    candidates &= ~((1 << i) | (1 << j) | (1 << k))
                    # Check if any candidate l > k
                    for l in range(k + 1, n):
                        if candidates & (1 << l):
                            return False
        return True

    # Seed graph: K6,6,6
    seed_adj = [0] * n
    for i in range(6):
        for j in range(6, 12):
            seed_adj[i] |= (1 << j)
            seed_adj[j] |= (1 << i)
        for j in range(12, 18):
            seed_adj[i] |= (1 << j)
            seed_adj[j] |= (1 << i)
    for i in range(6, 12):
        for j in range(12, 18):
            seed_adj[i] |= (1 << j)
            seed_adj[j] |= (1 << i)

    best_adj = seed_adj[:]
    best_score = 12  # K6,6,6 has max triangle-free size 12

    rng = random.Random(42)

    # Phase 1: Random search to find graphs with score < 12
    trial = 0
    while time.time() < deadline:
        trial += 1
        adj = generate_random_max_k4_free(rng)

        # Quick check: does it have a triangle-free subgraph of size 12?
        if has_triangle_free_size(adj, 12, subsets_12):
            continue

        # Score is <= 11, which is better than 12
        # Now do local search to try to reduce further
        score = max_triangle_free_size(adj)
        if score < best_score:
            best_adj = adj[:]
            best_score = score

        # Local search: try to reduce max triangle-free size
        improved = True
        local_deadline = min(deadline, time.time() + 5.0)  # 5s per local search
        while improved and time.time() < local_deadline:
            improved = False
            all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
            rng.shuffle(all_edges)

            for u, v in all_edges:
                if time.time() >= local_deadline:
                    break

                old_u = adj[u]
                old_v = adj[v]
                has_edge = bool(adj[u] & (1 << v))

                # Try flipping the edge
                if has_edge:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

                # Check K4-free
                if not is_k4_free(adj):
                    adj[u] = old_u
                    adj[v] = old_v
                    continue

                # Check if score improved
                if has_triangle_free_size(adj, 12, subsets_12):
                    adj[u] = old_u
                    adj[v] = old_v
                    continue

                if has_triangle_free_size(adj, 11, subsets_11):
                    # Score is 11, not better than current best if best_score <= 11
                    if best_score <= 11:
                        adj[u] = old_u
                        adj[v] = old_v
                        continue
                    # Accept if it improves over best
                    new_score = 11
                    if new_score < best_score:
                        best_adj = adj[:]
                        best_score = new_score
                        improved = True
                    else:
                        adj[u] = old_u
                        adj[v] = old_v
                    continue

                # Score is 10 or less
                new_score = 10
                if new_score < best_score:
                    best_adj = adj[:]
                    best_score = new_score
                    improved = True
                else:
                    adj[u] = old_u
                    adj[v] = old_v

        # If we found score 10, we can stop early
        if best_score <= 10:
            break

    return adj_to_edges(best_adj)
