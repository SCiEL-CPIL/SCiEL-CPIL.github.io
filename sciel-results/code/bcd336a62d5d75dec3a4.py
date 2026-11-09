import time
import random
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for safety

    def check_validity_bit(n, adj_rows):
        """Check if no subset of size >= 6 induces a regular subgraph.
        adj_rows[i] is an integer where bit j is 1 iff edge (i,j) exists.
        Uses bit operations for speed.
        """
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                # Compute degrees using bit operations
                # For vertex i in comb, degree = popcount(adj_rows[i] & mask)
                # where mask has bits set for all vertices in comb except i
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                degrees = []
                for i in comb:
                    # mask without bit i
                    mi = mask & ~(1 << i)
                    d = (adj_rows[i] & mi).bit_count()
                    degrees.append(d)
                if all(d == degrees[0] for d in degrees):
                    return False
        return True

    def count_bad_6_bit(n, adj_rows, combs):
        """Count number of 6-subsets inducing regular subgraph using bit ops."""
        count = 0
        for comb in combs:
            mask = 0
            for v in comb:
                mask |= (1 << v)
            deg0 = (adj_rows[comb[0]] & (mask & ~(1 << comb[0]))).bit_count()
            # Quick check: if any degree differs from deg0, skip
            is_reg = True
            for i in range(1, 6):
                d = (adj_rows[comb[i]] & (mask & ~(1 << comb[i]))).bit_count()
                if d != deg0:
                    is_reg = False
                    break
            if is_reg:
                count += 1
        return count

    def greedy_build(n, max_time=None):
        """
        Greedy sequential vertex addition.
        Build graph on n vertices by adding one vertex at a time.
        For each new vertex, try all 2^(n-1) possible edge sets to existing vertices
        (or sample if too many), pick the one that minimizes new bad 6-subsets.
        """
        if max_time is None:
            max_time = deadline

        # Start with empty graph
        adj_rows = [0] * n

        for new_v in range(1, n):
            if time.time() > max_time:
                return None

            # Number of existing vertices
            m = new_v
            # We need to choose which of the m existing vertices to connect to new_v
            # Total possible edge sets: 2^m
            # For m >= 20, this is too many. But m grows from 1 to n-1.
            # For m <= 15, we can enumerate all 2^m possibilities.
            # For m > 15, we sample.

            # Precompute: for each existing vertex i, what 6-subsets would be affected?
            # A 6-subset containing new_v and 5 of the existing vertices.
            # When we add edges from new_v to some subset S of {0,...,m-1},
            # the induced subgraph on {new_v} ∪ T where T ⊂ {0,...,m-1}, |T|=5
            # has new_v's degree = |T ∩ S| and each vertex t in T has its degree
            # in the existing subgraph on T plus (1 if t in S else 0).

            # Strategy: enumerate or sample edge sets, compute bad_6 count for
            # subsets containing new_v, pick best.

            num_existing = m
            total_options = 1 << num_existing

            if total_options <= 200000:
                # Enumerate all
                candidates = range(total_options)
            else:
                # Sample
                n_samples = 50000
                candidates = [random.randrange(total_options) for _ in range(n_samples)]
                # Also include some structured ones
                # All zeros (no edges)
                # All ones (complete)
                # Random
                candidates = list(set(candidates + [0, total_options - 1]))

            best_mask = None
            best_score = float('inf')

            # Precompute: for each 5-subset of existing vertices, the degree pattern
            # Actually, let's just evaluate each candidate.

            # To speed up, precompute for each 5-subset T of {0,...,m-1}:
            # - the degrees of vertices in T in the current graph (restricted to T)
            # - this doesn't change with our choice of S
            # Then for a given S (edge mask for new_v):
            #   For each 5-subset T:
            #     new_v degree = popcount(S & T_mask)
            #     vertex t in T: degree = deg_T[t] + (1 if t in S else 0)
            #     Check if all 6 degrees are equal.

            # Precompute 5-subsets of existing vertices
            if num_existing < 5:
                # No 6-subsets possible yet (need 5 existing + 1 new = 6)
                # Just pick randomly
                best_mask = random.randrange(total_options)
            else:
                five_subsets = list(itertools.combinations(range(num_existing), 5))

                # For each 5-subset, precompute the degree of each vertex within that subset
                # deg_T[t] = number of neighbors of t in T (in current graph)
                five_deg_data = []
                for T in five_subsets:
                    T_mask = 0
                    for t in T:
                        T_mask |= (1 << t)
                    degs = []
                    for t in T:
                        # Count neighbors of t within T
                        d = (adj_rows[t] & T_mask & ~(1 << t)).bit_count()
                        degs.append(d)
                    five_deg_data.append((T, T_mask, degs))

                for mask in candidates:
                    # mask: bit i set means edge (new_v, i) exists
                    bad_count = 0
                    for T, T_mask, degs in five_deg_data:
                        # new_v's degree in {new_v} ∪ T
                        nv_deg = (mask & T_mask).bit_count()
                        # Each vertex t in T has deg t + (1 if edge to new_v)
                        # Check if all 6 are equal
                        # The 6 degrees: nv_deg, and for each t in T: degs[i] + (1 if (mask >> t) & 1 else 0)
                        # For this to be regular, all must equal nv_deg.
                        # So for each t: degs[i] + ((mask >> t) & 1) == nv_deg
                        ok = True
                        for i, t in enumerate(T):
                            edge_to_nv = (mask >> t) & 1
                            if degs[i] + edge_to_nv != nv_deg:
                                ok = False
                                break
                        if ok:
                            bad_count += 1

                    if bad_count < best_score:
                        best_score = bad_count
                        best_mask = mask
                        if bad_count == 0:
                            break

            # Apply best_mask
            for i in range(num_existing):
                if (best_mask >> i) & 1:
                    adj_rows[new_v] |= (1 << i)
                    adj_rows[i] |= (1 << new_v)

            # Check time
            if time.time() > max_time:
                return None

        # Verify full validity
        if check_validity_bit(n, adj_rows):
            return adj_rows
        return None

    def local_search_bit(n, max_iters=300, combs=None):
        """Local search minimizing bad 6-subsets using bit representation."""
        if combs is None:
            combs = list(itertools.combinations(range(n), 6))

        # Random init
        adj_rows = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_rows[i] |= (1 << j)
                    adj_rows[j] |= (1 << i)

        # Initial count
        cur_count = count_bad_6_bit(n, adj_rows, combs)
        if cur_count == 0:
            if check_validity_bit(n, adj_rows):
                return adj_rows

        best_adj = adj_rows[:]
        best_count = cur_count

        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

        # Precompute: for each edge (u,v), which 6-combs contain both
        edge_to_combs = {}
        for idx, comb in enumerate(combs):
            for i in range(6):
                for j in range(i + 1, 6):
                    u, v = comb[i], comb[j]
                    if u > v:
                        u, v = v, u
                    edge_to_combs.setdefault((u, v), []).append(idx)

        for it in range(max_iters):
            if time.time() > deadline:
                break

            u, v = random.choice(edges)
            idxs = edge_to_combs.get((u, v), [])
            if not idxs:
                continue

            # Compute delta
            delta = 0
            for idx in idxs:
                comb = combs[idx]
                mask = 0
                for x in comb:
                    mask |= (1 << x)
                # Current degrees
                degs = []
                for x in comb:
                    d = (adj_rows[x] & (mask & ~(1 << x))).bit_count()
                    degs.append(d)
                if all(d == degs[0] for d in degs):
                    # Was regular, flipping makes it non-regular
                    delta -= 1
                else:
                    # Was non-regular. Check if flip makes it regular.
                    # Flip changes degrees of u and v by ±1
                    if (adj_rows[u] >> v) & 1:
                        # Removing edge: u and v lose 1
                        new_degs = []
                        for x in comb:
                            d = (adj_rows[x] & (mask & ~(1 << x))).bit_count()
                            if x == u or x == v:
                                d -= 1
                            new_degs.append(d)
                    else:
                        # Adding edge: u and v gain 1
                        new_degs = []
                        for x in comb:
                            d = (adj_rows[x] & (mask & ~(1 << x))).bit_count()
                            if x == u or x == v:
                                d += 1
                            new_degs.append(d)
                    if all(d == new_degs[0] for d in new_degs):
                        delta += 1

            new_count = cur_count + delta
            if new_count <= cur_count:
                # Accept
                if (adj_rows[u] >> v) & 1:
                    adj_rows[u] &= ~(1 << v)
                    adj_rows[v] &= ~(1 << u)
                else:
                    adj_rows[u] |= (1 << v)
                    adj_rows[v] |= (1 << u)
                cur_count = new_count
                if cur_count < best_count:
                    best_count = cur_count
                    best_adj = adj_rows[:]
                if cur_count == 0:
                    if check_validity_bit(n, adj_rows):
                        return adj_rows

        if best_count == 0:
            if check_validity_bit(n, best_adj):
                return best_adj
        return None

    def adj_to_edges(n, adj_rows):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_rows[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    # Try n=21 with greedy
    for attempt in range(3):
        if time.time() > start_time + 40:
            break
        random.seed(attempt * 12345 + 7)
        adj = greedy_build(21, max_time=start_time + 40)
        if adj is not None:
            return {'n': 21, 'edges': adj_to_edges(21, adj)}

    # Try n=21 with local search
    for attempt in range(5):
        if time.time() > start_time + 70:
            break
        random.seed(attempt * 54321 + 13)
        adj = local_search_bit(21, max_iters=200)
        if adj is not None:
            return {'n': 21, 'edges': adj_to_edges(21, adj)}

    # Try n=20 with greedy
    for attempt in range(5):
        if time.time() > start_time + 100:
            break
        random.seed(attempt * 9999 + 42)
        adj = greedy_build(20, max_time=start_time + 100)
        if adj is not None:
            return {'n': 20, 'edges': adj_to_edges(20, adj)}

    # Try n=20 with local search
    for attempt in range(10):
        if time.time() > start_time + 125:
            break
        random.seed(attempt * 7777 + 99)
        adj = local_search_bit(20, max_iters=300)
        if adj is not None:
            return {'n': 20, 'edges': adj_to_edges(20, adj)}

    # Fallback: n=15
    for attempt in range(5):
        if time.time() > start_time + 135:
            break
        random.seed(attempt * 3333 + 55)
        adj = local_search_bit(15, max_iters=500)
        if adj is not None:
            return {'n': 15, 'edges': adj_to_edges(15, adj)}

    # Ultimate fallback
    return {'n': 5, 'edges': []}
