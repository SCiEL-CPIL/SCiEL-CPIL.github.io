"""
Strategy: Use the iterative backtracking structure from the strongest inspiration,
but replace the expensive path-checking constraint with a fast local C6 detection.
This allows exploring much deeper in the search tree within the time budget.

Key structural change:
- Instead of checking for any path of length 5 between u and v (which is expensive),
  we check if adding edge (u,v) to color c creates a C6 by looking for a path of length 5
  between u and v that does NOT use the edge (u,v) itself. But more efficiently,
  we can check if there exists a simple path of length 5 from u to v in the current
  adjacency of color c. However, even this is expensive.

A better structural change: Use a different ordering and constraint propagation.
We order edges by (min degree, max degree) to fail early. But more importantly,
we use a faster C6 check: when considering adding edge (u,v) in color c,
we need to check if there is a path u-a-b-d-e-v of length 5 where all vertices are distinct
and the path does not use edge (u,v). This is equivalent to checking if v is reachable from u
in exactly 5 steps in the graph of color c with edge (u,v) removed. But since we're adding (u,v),
we just need to check if there's a path of length 5 from u to v in the current graph.

Actually, the most efficient check is: for each pair (u,v) and color c, count the number of
paths of length 5 from u to v. If it's >= 1, then adding (u,v) creates a C6.

But computing this is still expensive. Let's use a different approach:
Instead of strict backtracking, use a local search (simulated annealing or greedy repair)
to minimize the number of C6s, then try to repair to zero.

However, the directive says to make a STRUCTURAL change. Let's use a hybrid:
1. Start with a random or heuristic coloring.
2. Use local search to minimize the total number of C6s across all colors.
3. If we reach zero, we're done.
4. If not, try to find a valid coloring via backtracking with the best heuristic ordering.

Actually, let's stick with backtracking but use a MUCH faster constraint check.
The key insight: when adding edge (u,v) in color c, a C6 is formed if and only if
there exists a path of length 5 from u to v in the current color-c graph.
We can precompute for each color c and each pair (u,v) the number of paths of length 5.
But this is expensive to update.

Alternative: Use a different representation. Represent each color class as a bitset.
When adding edge (u,v) to color c, check if there's a path u-a-b-d-e-v.
We can do this by:
  - For each neighbor a of u in color c (a != v):
    - For each neighbor b of a in color c (b != u, b != v):
      - For each neighbor d of b in color c (d != u, d != a, d != v):
        - For each neighbor e of d in color c (e != u, e != a, e != b, e != v):
          - If e is a neighbor of v in color c: C6 found.

This is O(d^4) where d is the max degree. For n=17, d can be up to 16, so 16^4 = 65536, which is too slow.

We need a faster check. Let's use matrix multiplication or bitset operations.
Actually, let's use a different strategy: order the edges in a way that fails early,
and use a time limit to return the best n found.

Let's combine: use the backtracking from inspiration 1, but with a faster C6 check
using precomputed path counts that are updated incrementally.

Actually, the simplest structural change that might help: use a different edge ordering.
Order edges by (i+j) or by some heuristic that groups edges that are likely to conflict.

Let's try: order edges by (min(i,j), max(i,j)) but with a twist: prioritize edges that
are in the "center" of the graph. This might lead to earlier conflicts.

Another idea: use symmetry breaking. Fix the first few edges to reduce the search space.

Let's implement a version that:
1. Uses symmetry breaking for the first few edges.
2. Uses a fast C6 check with bitsets.
3. Has a time limit and returns the best n found.
"""

import time

def construct():
    start_time = time.time()
    hard_deadline = start_time + 115.0

    def verify(n, colors):
        """Verify that no color class contains a C6."""
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        for c in range(4):
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[c][i] >> j) & 1:
                        continue
                    # Check for path of length 5 from i to j avoiding direct edge
                    # i-a-b-d-e-j
                    found = False
                    mask_i = adj[c][i] & ~(1 << j)
                    while mask_i and not found:
                        lsb_a = mask_i & -mask_i
                        a = lsb_a.bit_length() - 1
                        mask_i -= lsb_a
                        mask_a = adj[c][a] & ~(1 << i) & ~(1 << j)
                        while mask_a and not found:
                            lsb_b = mask_a & -mask_a
                            b = lsb_b.bit_length() - 1
                            mask_a -= lsb_b
                            mask_b = adj[c][b] & ~(1 << i) & ~(1 << a) & ~(1 << j)
                            while mask_b and not found:
                                lsb_d = mask_b & -mask_b
                                d = lsb_d.bit_length() - 1
                                mask_b -= lsb_d
                                mask_d = adj[c][d] & ~(1 << i) & ~(1 << a) & ~(1 << b) & ~(1 << j)
                                while mask_d and not found:
                                    lsb_e = mask_d & -mask_d
                                    e = lsb_e.bit_length() - 1
                                    mask_d -= lsb_e
                                    if (adj[c][e] >> j) & 1:
                                        found = True
                                        break
                    if found:
                        return False
        return True

    def attempt(n, time_budget):
        """
        Backtracking search with fast C6 check using bitsets.
        """
        local_deadline = time.time() + time_budget
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        colors = [-1] * num_edges
        adj = [[0] * n for _ in range(4)]

        def has_c6(c, u, v):
            """Check if adding edge (u,v) to color c creates a C6.
            Look for path u-a-b-d-e-v where all distinct and not using (u,v)."""
            # a in N(u) \ {v}
            mask_a = adj[c][u] & ~(1 << v)
            while mask_a:
                lsb_a = mask_a & -mask_a
                a = lsb_a.bit_length() - 1
                mask_a -= lsb_a
                # b in N(a) \ {u, v}
                mask_b = adj[c][a] & ~(1 << u) & ~(1 << v)
                while mask_b:
                    lsb_b = mask_b & -mask_b
                    b = lsb_b.bit_length() - 1
                    mask_b -= lsb_b
                    # d in N(b) \ {u, a, v}
                    mask_d = adj[c][b] & ~(1 << u) & ~(1 << a) & ~(1 << v)
                    while mask_d:
                        lsb_d = mask_d & -mask_d
                        d = lsb_d.bit_length() - 1
                        mask_d -= lsb_d
                        # e in N(d) \ {u, a, b, v}
                        mask_e = adj[c][d] & ~(1 << u) & ~(1 << a) & ~(1 << b) & ~(1 << v)
                        while mask_e:
                            lsb_e = mask_e & -mask_e
                            e = lsb_e.bit_length() - 1
                            mask_e -= lsb_e
                            if (adj[c][e] >> v) & 1:
                                return True
            return False

        # Iterative backtracking
        stack = [(0, 0)]  # (edge_index, colors_tried_mask)
        while stack:
            idx, tried_mask = stack[-1]

            if time.time() > local_deadline:
                return None

            if idx == num_edges:
                return colors[:]

            u, v = edges[idx]
            found = False
            for c in range(4):
                cbit = 1 << c
                if tried_mask & cbit:
                    continue
                if not has_c6(c, u, v):
                    colors[idx] = c
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    stack[-1] = (idx, tried_mask | cbit)
                    stack.append((idx + 1, 0))
                    found = True
                    break

            if not found:
                stack.pop()
                if idx > 0:
                    prev_idx = idx - 1
                    prev_u, prev_v = edges[prev_idx]
                    prev_c = colors[prev_idx]
                    colors[prev_idx] = -1
                    adj[prev_c][prev_u] &= ~(1 << prev_v)
                    adj[prev_c][prev_v] &= ~(1 << prev_u)

        return None

    # Try n=17 first
    res = attempt(17, 50)
    if res:
        if verify(17, res):
            return {'n': 17, 'colors': res}

    # Try n=16
    if time.time() < hard_deadline - 40:
        res = attempt(16, 30)
        if res:
            if verify(16, res):
                return {'n': 16, 'colors': res}

    # Try n=15
    if time.time() < hard_deadline - 15:
        res = attempt(15, 10)
        if res:
            if verify(15, res):
                return {'n': 15, 'colors': res}

    # Fallback to seed solution
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
