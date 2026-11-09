import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin under the 150s evaluator limit

    # Fallback: Andrásfai graph And(9) on 26 vertices (triangle-free, alpha = 9)
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed)
                  if (j - i) % n_seed % 3 == 1]
    best_result = {'n': n_seed, 'edges': edges_seed}

    def edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    def has_independent_set_10(adj, n):
        """Return True iff the graph has an independent set of size 10.

        Uses bitmask branch-and-bound.  Triangle-freeness is assumed to be
        checked separately by the caller; this only tests alpha < 10.
        """
        all_mask = (1 << n) - 1
        non_neighbors = [0] * n
        for i in range(n):
            non_neighbors[i] = (all_mask ^ (1 << i)) & ~adj[i]

        # Precompute degree within the full set for tie-breaking
        deg = [non_neighbors[i].bit_count() for i in range(n)]

        def dfs(mask, size):
            if size == 10:
                return True
            if mask == 0:
                return False
            if mask.bit_count() < 10 - size:
                return False

            # Pick the most constrained vertex (fewest non-neighbors in mask).
            # This maximizes the "include" branch's pruning power.
            min_count = n + 1
            best_v = -1
            temp_mask = mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                cnt = (non_neighbors[v] & mask).bit_count()
                if cnt < min_count:
                    min_count = cnt
                    best_v = v
                    if cnt == 0:
                        break
                temp_mask ^= lsb

            if best_v == -1:
                return False

            v_bit = 1 << best_v

            # Include best_v: candidates become its non-neighbors within mask.
            if dfs(non_neighbors[best_v] & mask, size + 1):
                return True

            # Exclude best_v.
            if dfs(mask & ~v_bit, size):
                return True

            return False

        return dfs(all_mask, 0)

    # --- Strategy 1: deterministic polarity graph of PG(2, 5) -> n = 31 ---
    def construct_polarity_graph(q):
        n = q * q + q + 1
        if n > 40 or n < 2:
            return None, None
        points = []
        seen = set()
        for x in range(q):
            for y in range(q):
                for z in range(q):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    if x != 0:
                        inv = pow(x, q - 2, q)
                        rep = (1, (y * inv) % q, (z * inv) % q)
                    elif y != 0:
                        inv = pow(y, q - 2, q)
                        rep = (0, 1, (z * inv) % q)
                    else:
                        rep = (0, 0, 1)
                    if rep not in seen:
                        seen.add(rep)
                        points.append(rep)
        n_actual = len(points)
        adj = [0] * n_actual
        for i in range(n_actual):
            for j in range(i + 1, n_actual):
                dot = (points[i][0] * points[j][0] +
                       points[i][1] * points[j][1] +
                       points[i][2] * points[j][2]) % q
                if dot == 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return n_actual, adj

    for q in (5, 4, 3, 2):
        if time.time() > deadline - 5:
            break
        n_pol, adj_pol = construct_polarity_graph(q)
        if n_pol is not None and n_pol > best_result['n']:
            if not has_independent_set_10(adj_pol, n_pol):
                best_result = {'n': n_pol, 'edges': edges_from_adj(adj_pol, n_pol)}

    # --- Strategy 2: random maximal triangle-free search, largest n first ---
    # A maximal triangle-free graph is complete multipartite, which minimizes
    # alpha for a given edge count, so it is the right target distribution.
    for n in range(40, best_result['n'], -1):
        if time.time() > deadline - 10:
            break
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        restarts = 0
        while time.time() < deadline - 5:
            restarts += 1
            if restarts > 4000:
                break

            adj = [0] * n
            order = pairs[:]
            random.shuffle(order)

            # Pass 1: greedily add non-triangle edges.
            for u, v in order:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

            # Pass 2: make it maximal (add every remaining safe edge).
            for u, v in order:
                if not (adj[u] & (1 << v)) and (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

            if not has_independent_set_10(adj, n):
                best_result = {'n': n, 'edges': edges_from_adj(adj, n)}
                break  # found a valid graph for this n; try the next smaller n

    return best_result
