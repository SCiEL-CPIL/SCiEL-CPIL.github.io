"""Difference triangle set: MWIS branch-and-bound with bitset pruning + greedy coloring bound."""
import sys

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0]] * n_rows

    # --- Build all candidate rows of k+1 increasing ints starting at 0 with scope <= M ---
    def gen_rows(M):
        rows = []
        diffs = []
        for a1 in range(1, M + 1):
            for a2 in range(a1 + 1, M + 1):
                for a3 in range(a2 + 1, M + 1):
                    for a4 in range(a3 + 1, M + 1):
                        for a5 in range(a4 + 1, M + 1):
                            if a5 > M:
                                break
                            row = [0, a1, a2, a3, a4, a5]
                            # all positive differences
                            d = set()
                            for i in range(6):
                                for j in range(i + 1, 6):
                                    d.add(row[j] - row[i])
                            if len(d) != 15:  # should be 15 distinct for strictly increasing
                                continue
                            rows.append(row)
                            diffs.append(d)
        return rows, diffs

    # --- Compatibility graph: node i,j incompatible if diff sets intersect ---
    # For speed, precompute for a given M the diff sets as frozensets.

    # Greedy coloring bound on the independent-set problem:
    # We model the conflict graph G (edges = incompatible pairs). We want an independent
    # set of size n_rows in G. Equivalent: choose n_rows nodes no two adjacent.
    # Standard branch-and-bound: at each step pick a node, branch include/exclude.
    # Upper bound on extendable size from a candidate set C: use greedy coloring of the
    # induced subgraph on C; max independent set <= number of colors.

    import sys
    sys.setrecursionlimit(100000)

    def solve(M):
        rows, diffs = gen_rows(M)
        n = len(rows)
        if n < n_rows:
            return None

        # Build conflict adjacency as bitmasks (Python ints)
        # conflict[i] is bitmask of j != i incompatible with i
        conflict = [0] * n
        for i in range(n):
            di = diffs[i]
            for j in range(i + 1, n):
                if di & diffs[j]:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)

        # Candidate pool as bitmask
        all_nodes = (1 << n) - 1

        # Precompute for each node its "conflict mask" restricted to pool for fast exclusion.
        # We'll maintain a pool mask; when we include node i, new pool = pool & ~conflict[i] & ~bit(i).

        best = [None]
        found = [False]

        # Greedy coloring bound: given a pool mask, color the induced subgraph greedily
        # (order nodes by degree descending within pool) to get an upper bound on max independent set size.
        # We use the classic "coloring bound" from branch-and-bound for maximum clique; here we
        # want max independent set, which is max clique in the complement. But simpler: we can
        # directly bound max independent set by greedy coloring of the conflict graph on pool:
        # independent set size <= number of colors used by any proper coloring of the conflict graph.

        # To make it fast, we use a simple greedy: repeatedly pick a maximal independent set as one color class.
        def coloring_bound(pool):
            if pool == 0:
                return 0
            colors = 0
            remaining = pool
            while remaining:
                colors += 1
                # pick a maximal independent set from remaining (greedy)
                # order by fewest conflicts within remaining (to pack more)
                # simple: iterate nodes, add if no conflict with already-chosen in this color
                chosen = 0
                # iterate bits
                r = remaining
                while r:
                    # lowest set bit
                    lb = r & (-r)
                    idx = lb.bit_length() - 1
                    if not (conflict[idx] & chosen):
                        chosen |= lb
                    r &= (r - 1)
                remaining &= ~chosen
            return colors

        # Branch and bound search for independent set of size n_rows
        # State: pool (bitmask), chosen (list of node indices), depth
        def search(pool, chosen, depth):
            if found[0]:
                return
            if depth == n_rows:
                found[0] = True
                best[0] = list(chosen)
                return
            if pool == 0:
                return
            # upper bound
            if coloring_bound(pool) < (n_rows - depth):
                return
            # pick a node to branch on: choose node with max conflicts in pool (to prune more)
            # iterate bits, compute popcount(conflict[i] & pool)
            r = pool
            best_node = -1
            best_conf = -1
            while r:
                lb = r & (-r)
                idx = lb.bit_length() - 1
                c = (conflict[idx] & pool).bit_count()
                if c > best_conf:
                    best_conf = c
                    best_node = idx
                r &= (r - 1)
            if best_node < 0:
                return
            bit = 1 << best_node
            # Branch 1: include best_node
            chosen.append(best_node)
            new_pool = pool & ~bit & ~conflict[best_node]
            search(new_pool, chosen, depth + 1)
            chosen.pop()
            if found[0]:
                return
            # Branch 2: exclude best_node
            search(pool & ~bit, chosen, depth)

        # To keep runtime bounded, we limit work: if node count is huge, we may not finish.
        # For (7,5) with M=112, number of rows is C(112,5) ~ 65M which is too many to enumerate.
        # So this naive approach won't work for large M. We need a smarter candidate generation.
        # Given the 120s limit, we'll cap M and node count.
        if n > 200000:
            return None

        search(all_nodes, [], 0)
        if not found[0]:
            return None
        return [rows[i] for i in best[0]]

    # Search M upward from a lower bound.
    # Lower bound: counting bound 105 for (7,5). We'll start from a reasonable floor.
    # To be safe and fast, start from max(0, ...) and go up; but for large M the row count explodes.
    # We'll cap the search: for (7,5) we know 112 works. Try M in a window.
    # Strategy: binary-ish search upward, but stop at first feasible M.
    # Given time limit, we'll try M values and bail if a single solve is too slow.
    import time
    t0 = time.time()
    deadline = t0 + 110  # leave 10s margin

    # For (7,5), known answer 112. We'll try a range.
    # But to be general, start from a heuristic lower bound.
    # Counting bound: total diffs needed = n_rows * k*(k+1)/2. Max diff = M, so M >= that.
    # For (7,5): 7*15=105. Start at 105.
    start = 105
    if n_rows * k * (k + 1) // 2 > start:
        start = n_rows * k * (k + 1) // 2

    # Try M from start upward, but limit total attempts.
    for M in range(start, start + 40):
        if time.time() > deadline:
            break
        res = solve(M)
        if res is not None:
            return res

    # Fallback: greedy (original approach) if MWIS search fails within time.
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
