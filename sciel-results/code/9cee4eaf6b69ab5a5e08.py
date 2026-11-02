"""
MIS-based constructor for difference-triangle-sets.

Strategy:
1. Generate a pool of candidate rows using randomized greedy (varied starts/skips).
2. Build a conflict graph: two rows conflict if they share any positive difference.
3. Solve Maximum Independent Set on this graph using branch-and-bound with greedy coloring.
4. From the selected rows, apply right-to-left compaction to minimize the maximum entry.
5. Repeat with different pools until deadline; keep best scope.
"""

import time
import random
import sys
from array import array


def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 115.0
    m = k + 1  # entries per row
    needed = n_rows

    # ---------- helpers ----------

    def row_diffs(row):
        """Return the set of all positive differences in a row."""
        s = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                s.add(row[j] - row[i])
        return s

    def greedy_row(used, start=0, skip_prob=0.0, rng=None):
        """Build one row greedily avoiding differences in 'used'."""
        row = array('i', [0])
        x = start
        while len(row) < m:
            x += 1
            if skip_prob > 0 and rng is not None and rng.random() < skip_prob:
                continue
            ok = True
            for a in row:
                if (x - a) in used:
                    ok = False
                    break
            if ok:
                for a in row:
                    used.add(x - a)
                row.append(x)
        return list(row)

    def compact_rows(rows):
        """Right-to-left compaction: reduce each element to minimum feasible value."""
        all_diffs = set()
        for row in rows:
            all_diffs |= row_diffs(row)

        improved = True
        max_rounds = 50
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            for r_idx in range(len(rows)):
                row = rows[r_idx]
                for pos in range(len(row) - 1, 0, -1):
                    cur = row[pos]
                    if cur <= row[pos - 1] + 1:
                        continue
                    # Try values from row[pos-1]+1 upward to cur-1
                    for nv in range(row[pos - 1] + 1, cur):
                        old_d = set()
                        for i in range(pos):
                            old_d.add(cur - row[i])
                        for j in range(pos + 1, len(row)):
                            old_d.add(row[j] - cur)
                        new_d = set()
                        for i in range(pos):
                            new_d.add(nv - row[i])
                        for j in range(pos + 1, len(row)):
                            new_d.add(row[j] - nv)
                        # Check conflicts: new_d elements not in old_d must not be in all_diffs
                        conflict = False
                        for d in new_d:
                            if d in old_d:
                                continue
                            if d in all_diffs:
                                conflict = True
                                break
                        if not conflict:
                            # Also ensure new_d has no internal duplicates (shouldn't happen for increasing row)
                            for d in old_d:
                                all_diffs.discard(d)
                            for d in new_d:
                                all_diffs.add(d)
                            row[pos] = nv
                            improved = True
                            break
        return rows

    def greedy_coloring(adj, n):
        """Greedy coloring for branch-and-bound upper bound. Returns list of colors."""
        color = [0] * n
        for v in range(n):
            used_colors = set()
            for u in adj[v]:
                if color[u]:
                    used_colors.add(color[u])
            c = 1
            while c in used_colors:
                c += 1
            color[v] = c
        return color

    def mis_branch_and_bound(adj, n, time_limit):
        """
        Branch-and-bound MIS with greedy coloring.
        adj[v] is a set of neighbors of v.
        Returns the independent set (list of vertices) found.
        """
        start_time = time.time()
        best_set = []
        best_size = 0

        # Work on a mutable copy of adjacency for pruning
        # Use a recursive approach with a stack to avoid recursion depth issues
        # State: (P, X, adj_P) where P is candidate set, X is excluded set
        # We'll use a simpler iterative approach

        # Initial upper bound via greedy coloring
        colors = greedy_coloring(adj, n)
        max_color = max(colors) if n > 0 else 0

        # Simple recursive B&B with time check
        def solve(P, X, current_set, upper_bound):
            nonlocal best_set, best_size
            if time.time() - start_time > time_limit:
                return
            if len(P) == 0:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                return
            if len(P) + len(current_set) <= best_size:
                return
            # Upper bound: max color in P
            P_list = list(P)
            p_colors = [colors[v] for v in P_list]
            ub = len(current_set) + max(p_colors) if p_colors else len(current_set)
            if ub <= best_size:
                return

            # Pick vertex with max degree in P
            v = max(P_list, key=lambda x: len(adj[x] & P))
            # Branch 1: include v
            P_new = P - {v} - adj[v]
            X_new = X | {v}
            current_set.append(v)
            solve(P_new, X_new, current_set, ub)
            current_set.pop()

            # Branch 2: exclude v
            P_new2 = P - {v}
            X_new2 = X | {v}
            solve(P_new2, X_new2, current_set, ub)

        # To handle large graphs, use a simpler approach:
        # Iterative greedy + local search
        # For now, use a simple recursive version with time limit
        if n <= 60:
            solve(set(range(n)), set(), [], max_color)
        else:
            # Fallback: greedy independent set
            P = set(range(n))
            current_set = []
            while P:
                v = max(P, key=lambda x: -len(adj[x] & P))
                current_set.append(v)
                P -= adj[v]
                P -= {v}
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set

        return best_set

    def generate_pool(rng, pool_size, max_scope):
        """Generate a pool of candidate rows."""
        pool = []
        seen_diff_sets = set()
        attempts = 0
        max_attempts = pool_size * 20
        while len(pool) < pool_size and attempts < max_attempts:
            attempts += 1
            start = rng.randint(0, min(3 * m + 10, max_scope // max(needed, 1)))
            skip = rng.choice([0.0, 0.05, 0.1, 0.15, 0.2])
            used = set()
            row = greedy_row(used, start, skip, rng)
            if max(row) > max_scope:
                continue
            ds = frozenset(row_diffs(row))
            if ds in seen_diff_sets:
                continue
            seen_diff_sets.add(ds)
            pool.append(row)
        return pool

    # ---------- main ----------

    # Baseline: deterministic greedy
    u = set()
    base_rows = []
    for _ in range(needed):
        base_rows.append(greedy_row(u, 0, 0.0, None))
    base_rows = compact_rows(base_rows)
    best_scope = max(max(r) for r in base_rows)
    best_rows = [list(r) for r in base_rows]

    rng = random.Random(42)

    # Try multiple pools with MIS
    pool_sizes = [80, 120, 200, 300]
    for pool_size in pool_sizes:
        if time.time() > deadline:
            break
        max_scope = int(best_scope * 1.2) + 20
        pool = generate_pool(rng, pool_size, max_scope)
        if len(pool) < needed:
            continue

        # Build conflict graph
        n = len(pool)
        diff_sets = [row_diffs(r) for r in pool]
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if diff_sets[i] & diff_sets[j]:
                    adj[i].add(j)
                    adj[j].add(i)

        # Solve MIS
        time_left = deadline - time.time()
        if time_left < 2.0:
            break
        bnb_time = min(time_left - 1.0, 30.0)
        independent = mis_branch_and_bound(adj, n, bnb_time)

        if len(independent) >= needed:
            # Take the first 'needed' rows from the independent set
            # Sort by max entry to get smaller scopes
            selected = sorted(independent, key=lambda idx: max(pool[idx]))
            chosen = [list(pool[idx]) for idx in selected[:needed]]
            chosen = compact_rows(chosen)
            scope = max(max(r) for r in chosen)
            if scope < best_scope:
                best_scope = scope
                best_rows = chosen
                # Update max_scope for next pool
                max_scope = int(best_scope * 1.1) + 10

    # Final compaction on best
    best_rows = compact_rows(best_rows)
    best_scope = max(max(r) for r in best_rows)

    return best_rows


if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
