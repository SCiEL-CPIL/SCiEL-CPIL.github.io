import time
import random

def construct():
    """
    Return a C4-free graph with independence number < 11, maximizing n.
    Strategy:
    1. Construct polarity graph of PG(2,5) (31 vertices, C4-free, alpha=6).
    2. Greedily add vertices using a conflict matrix to maintain C4-freeness.
    3. Use a fast bitmask alpha-check with branch-and-bound to verify alpha < 11.
    4. Return the largest valid n found.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def max_independent_set_size(n, adj, time_limit):
        """
        Determine if the graph (n vertices, adj as bitmask list) has an independent
        set of size >= 11.  Returns the max independent-set size found (capped at 11).
        Uses branch-and-bound on the complement graph (max clique).
        """
        target = 11
        best = [0]
        timed_out = [False]

        # Complement adjacency (for max-clique search)
        full = (1 << n) - 1
        comp = [0] * n
        for i in range(n):
            comp[i] = full ^ (1 << i) ^ adj[i]

        def expand(cand, cur_sz):
            if time.time() > time_limit:
                timed_out[0] = True
                return
            if cand == 0:
                if cur_sz > best[0]:
                    best[0] = cur_sz
                return
            if best[0] >= target:
                return
            if cur_sz + cand.bit_count() <= best[0]:
                return

            # Pick vertex of max degree in candidate subgraph of complement
            m = cand
            best_v = -1
            best_deg = -1
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                d = (cand & comp[v]).bit_count()
                if d > best_deg:
                    best_deg = d
                    best_v = v
                m ^= lsb

            if best_v == -1:
                return
            vb = 1 << best_v

            # Include best_v
            expand(cand & comp[best_v] & ~vb, cur_sz + 1)
            if best[0] >= target or timed_out[0]:
                return
            # Exclude best_v
            expand(cand & ~vb, cur_sz)

        expand((1 << n) - 1, 0)
        return best[0]

    # --- Step 1: Polarity graph of PG(2,5) ---
    # Build 31 points
    pts = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalise: first non-zero coordinate to 1
                if x != 0:
                    inv = pow(x, -1, 5)
                    pts.append((1, y * inv % 5, z * inv % 5))
                elif y != 0:
                    inv = pow(y, -1, 5)
                    pts.append((0, 1, z * inv % 5))
                else:
                    pts.append((0, 0, 1))
    pts = list(set(pts))
    n0 = len(pts)  # 31

    def bilinear(a, b):
        return (a[0]*b[2] + a[1]*b[1] + a[2]*b[0]) % 5

    adj = [0] * n0
    for i in range(n0):
        for j in range(i + 1, n0):
            if bilinear(pts[i], pts[j]) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Verify base graph
    if max_independent_set_size(n0, adj, deadline - 15) >= 11:
        return {'n': 1, 'edges': []}

    best_n = n0
    best_adj = adj[:]

    # --- Step 2: Greedy growth ---
    for new_n in range(n0 + 1, 44):
        if time.time() > deadline - 25:
            break

        n_prev = new_n - 1
        # Build conflict: conflict[i] bitmask of j (j != i) sharing a neighbour
        conflict = [0] * n_prev
        for i in range(n_prev):
            for j in range(i + 1, n_prev):
                if adj[i] & adj[j]:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)

        found = False
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            random.seed(12345 + attempt * 31 + new_n * 7)

            # Greedy independent set in conflict graph
            S = 0
            cand = (1 << n_prev) - 1
            while cand:
                # Pick vertex with minimum conflict-degree in remaining cand
                m = cand
                best_v = -1
                best_dc = 999
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    dc = (conflict[v] & cand).bit_count()
                    if dc < best_dc:
                        best_dc = dc
                        best_v = v
                    m ^= lsb
                if best_v == -1:
                    break
                S |= (1 << best_v)
                cand &= ~(1 << best_v)
                cand &= ~conflict[best_v]

            # Build new adjacency
            new_adj = adj[:] + [S]
            for i in range(n_prev):
                if S & (1 << i):
                    new_adj[i] |= (1 << (n_prev))

            # Check alpha
            alpha = max_independent_set_size(new_n, new_adj, deadline - 3)
            if alpha < 11:
                adj = new_adj
                best_n = new_n
                best_adj = adj[:]
                found = True
                break

        if not found:
            break

    # Build edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])

    return {'n': best_n, 'edges': edges}
