import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit

    n = 16
    target_edges = 65

    # Precompute all possible edges as (i, j) pairs with i < j
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    pair_index = {}
    for idx, (i, j) in enumerate(pairs):
        pair_index[(i, j)] = idx

    # Precompute all triples (i, j, k) with i < j < k
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triples.append((i, j, k))

    # For each pair, precompute which vertices are common neighbors
    # This helps with fast triangle detection
    # For pair (i,j), the triangles containing it are (i,j,k) for all k

    # Initialize with baseline: K_{8,8} + edge (0,1)
    # Using bitset representation: adj[v] is an integer with bit u set if edge (v,u) exists
    adj = [0] * n
    # K_{8,8}: vertices 0-7 connect to 8-15
    for u in range(8):
        for v in range(8, 16):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    # Add edge (0,1)
    adj[0] |= (1 << 1)
    adj[1] |= (1 << 0)

    # Compute degrees
    deg = [adj[v].bit_count() for v in range(n)]

    # Compute current triangles and their degree sums
    def compute_triangles(adj):
        tris = []
        for i, j, k in triples:
            if (adj[i] >> j) & 1 and (adj[j] >> k) & 1 and (adj[k] >> i) & 1:
                tris.append((i, j, k))
        return tris

    def compute_score(adj, deg):
        max_sum = 0
        for i, j, k in triples:
            if (adj[i] >> j) & 1 and (adj[j] >> k) & 1 and (adj[k] >> i) & 1:
                s = deg[i] + deg[j] + deg[k]
                if s > max_sum:
                    max_sum = s
        return max_sum

    current_score = compute_score(adj, deg)
    best_score = current_score
    best_adj = adj[:]

    # For fast incremental updates, we need to track which triangles exist
    # and their degree sums. When we swap edges, only triangles involving
    # the affected vertices change.

    # Let's use a different approach: maintain the set of active triangles
    # and their degree sums, and update incrementally.

    # Actually, for n=16, there are only C(16,3) = 560 triples.
    # So we can just recompute the score by checking all 560 triples.
    # With bitset checks, this is very fast.

    # Let's optimize the inner loop by precomputing for each vertex pair
    # which triples contain that pair

    # For a swap of edges (i,j) and (k,l):
    # - Removing edge (i,j) affects all triangles containing edge (i,j)
    # - Adding edge (k,l) affects all triangles containing edge (k,l)

    # Precompute: for each pair (i,j), list of vertices m such that (i,j,m) is a triple
    pair_to_third = {}
    for i, j, k in triples:
        if (i, j) not in pair_to_third:
            pair_to_third[(i, j)] = []
        pair_to_third[(i, j)].append(k)

    # Fast score computation using bitsets
    def fast_score(adj, deg):
        max_sum = 0
        for i, j, k in triples:
            if (adj[i] >> j) & 1 and (adj[j] >> k) & 1 and (adj[k] >> i) & 1:
                s = deg[i] + deg[j] + deg[k]
                if s > max_sum:
                    max_sum = s
        return max_sum

    # Simulated Annealing with bitset representation
    T = 5.0
    cool_rate = 0.9995
    min_T = 0.05

    # Count iterations for monitoring
    iter_count = 0

    while time.time() - start_time < deadline:
        # Find a valid swap: pick an existing edge and a non-edge
        # Existing edges: we need to know which pairs are currently edges
        # Let's just randomly pick pairs and check

        valid_swap = False
        for _ in range(20):
            p1 = random.choice(pairs)
            p2 = random.choice(pairs)
            i, j = p1
            k, l = p2
            if (adj[i] >> j) & 1 and not ((adj[k] >> l) & 1):
                valid_swap = True
                break

        if not valid_swap:
            continue

        # Perform swap: remove edge (i,j), add edge (k,l)
        adj[i] &= ~(1 << j)
        adj[j] &= ~(1 << i)
        adj[k] |= (1 << l)
        adj[l] |= (1 << k)

        deg[i] -= 1
        deg[j] -= 1
        deg[k] += 1
        deg[l] += 1

        new_score = fast_score(adj, deg)
        delta = new_score - current_score

        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = adj[:]
        else:
            # Reject, revert
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            adj[k] &= ~(1 << l)
            adj[l] &= ~(1 << k)

            deg[i] += 1
            deg[j] += 1
            deg[k] -= 1
            deg[l] -= 1

        T *= cool_rate
        if T < min_T:
            T = min_T

        iter_count += 1

        # Periodic check
        if iter_count % 100000 == 0:
            elapsed = time.time() - start_time
            if elapsed > deadline * 0.8:
                break

    # Extract edges from best_adj
    edges_list = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges_list.append([i, j])

    return {'n': n, 'edges': edges_list}
