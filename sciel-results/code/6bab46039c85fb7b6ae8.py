import numpy as np
import time

# ---------------------------------------------------------------------------
# Lazebnik-Ustimenko-Woldar fallback (kept exactly as seed interface)
# ---------------------------------------------------------------------------
def _luw_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


# ---------------------------------------------------------------------------
# Fast girth certification via vectorized adjacency-matrix powers.
# A cycle of length L exists iff (A^L)[s,s] > 0 for some s.
# Returns (True, None) if girth >= g, else (False, (cycle_len, s)).
# ---------------------------------------------------------------------------
def _check_girth(n, A, g):
    # Precompute A^2 .. A^g  (boolean, vectorized)
    # A^2 = A @ A ; A^L = A^(L-1) @ A
    powers = [None] * (g + 1)
    powers[1] = A
    for L in range(2, g + 1):
        powers[L] = powers[L - 1] @ A

    # For each L from 2..g-1, check diagonal.
    # (A^L)[s,s] > 0  =>  a closed walk of length L.
    # A closed walk of even length L contains a simple cycle of length <= L.
    # A closed walk of odd length L contains a simple cycle of length <= L.
    # So if any (A^L)[s,s] > 0 for L < g, girth < g.
    for L in range(2, g):
        P = powers[L]
        diag = np.diagonal(P)
        if np.any(diag):
            s = int(np.argmax(diag))
            return (False, (L, s))
    return (True, None)


# ---------------------------------------------------------------------------
# Random simple 3-regular graph generator (config model, vectorized shuffle)
# ---------------------------------------------------------------------------
def _gen_random_3reg(n, rng):
    if n < 4 or (n % 2) != 0:
        return None
    stubs = np.repeat(np.arange(n), 3)
    for _ in range(200):
        rng.shuffle(stubs)
        ok = True
        edge_set = set()
        edges = []
        for i in range(0, n * 3, 2):
            u = int(stubs[i]); v = int(stubs[i + 1])
            if u == v:
                ok = False; break
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                ok = False; break
            edge_set.add((u, v))
            edges.append((u, v))
        if ok:
            return edges
    return None


def _edges_to_arrays(n, edges):
    A = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
    return A


# ---------------------------------------------------------------------------
# 2-switch local search: break the shortest cycle, keep 3-regular.
# edges: list of (u,v) tuples.  A: current adjacency matrix (int8).
# Returns (new_edges, new_A) or (None, None) on failure.
# ---------------------------------------------------------------------------
def _one_2switch(n, edges, A, g, rng):
    # Find a short cycle to break.
    valid, info = _check_girth(n, A, g)
    if valid:
        return edges, A  # already good
    L, s = info

    # Collect edges incident to s and to its neighbors, to find a 2-switch
    # that breaks the cycle.  Strategy: find an edge (a,b) on the cycle and
    # an edge (c,d) off the cycle such that swapping (a,b)+(c,d) -> (a,c)+(b,d)
    # is valid (no loops, no multi-edges, preserves degree).

    # Get neighbors of s (3 of them)
    nbrs_s = list(np.nonzero(A[s])[0])

    # Build adjacency list from A for quick lookup
    adj = [list(np.nonzero(A[i])[0]) for i in range(n)]

    # Try all pairs of edges (a,b) and (c,d) where a,b are adjacent and c,d are
    # adjacent, and the new edges (a,c),(b,d) don't already exist and a!=c, b!=d.
    # We only need to try a bounded number to stay fast.
    # Focus on edges near s to hit the cycle.
    candidate_edges = []
    for a in nbrs_s:
        for b in adj[a]:
            if b != s and b > a:
                candidate_edges.append((a, b))
    # Also add a random sample of all edges
    for _ in range(30):
        e = edges[rng.integers(len(edges))]
        candidate_edges.append(e)

    for (a, b) in candidate_edges:
        # find all c adjacent to a, c != b, c != s
        for c in adj[a]:
            if c == b or c == s:
                continue
            # find all d adjacent to c, d != a
            for d in adj[c]:
                if d == a or d == b:
                    continue
                # Check (a,c) and (b,d) don't exist
                if A[a, c] or A[b, d]:
                    continue
                # Also (a,d) and (b,c) must not create issues - we're adding (a,c),(b,d)
                # Remove (a,b) and (c,d), add (a,c) and (b,d)
                # But (c,d) must be an edge
                if not A[c, d]:
                    continue
                # Apply switch
                A[a, b] = 0; A[b, a] = 0
                A[c, d] = 0; A[d, c] = 0
                A[a, c] = 1; A[c, a] = 1
                A[b, d] = 1; A[d, b] = 1

                # Build new edge list
                new_edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j and A[i, j]:
                            new_edges.append((i, j))
                # Rebuild adj
                new_adj = [list(np.nonzero(A[i])[0]) for i in range(n)]

                # Check if this broke the cycle
                valid2, _ = _check_girth(n, A, g)
                if valid2:
                    return new_edges, A

                # If not better, revert
                A[a, b] = 1; A[b, a] = 1
                A[c, d] = 1; A[d, c] = 1
                A[a, c] = 0; A[c, a] = 0
                A[b, d] = 0; A[d, b] = 0

    return None, None


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin under 120s

    def fallback():
        return _luw_construct(k, g)

    if k != 3:
        return fallback()

    rng = np.random.default_rng(42)

    # Moore bound for k=3, g=16: 1 + 3*sum(2^i, i=0..7) = 1+3*(255) = 766
    # But prompt says lower bound 512. Start from a reasonable n.
    # For g=16, random 3-reg graphs need n >> 512 to have girth 16.
    # Expected girth of random 3-reg graph on n vertices ~ (1/2)log_2(n).
    # For girth 16: n ~ 2^32 ~ 4e9. That's way too big for random search.
    # So random search won't find girth-16 graphs at small n.
    # We need the L-U-W construction or a better explicit construction.

    # The L-U-W construction with q=3, d=11 gives a 3-regular graph with
    # girth >= d+5 = 16. Its size is q^d * 2 = 3^11 * 2 = 354294 * 2 = 708588.
    # That's huge but valid. The known best is 936, so L-U-W is far from optimal.

    # For this task, we should just return the L-U-W construction since
    # random search can't find girth-16 graphs at feasible n.
    # But let's try a small random search for girth >= g in case g is small.

    if g <= 8:
        # Small girth: random search can work
        n = 2 * (g // 2 + 1)
        while n < 200:
            if time.time() > deadline - 5:
                break
            for _ in range(200):
                if time.time() > deadline - 5:
                    break
                edges = _gen_random_3reg(n, rng)
                if edges is None:
                    continue
                A = _edges_to_arrays(n, edges)
                valid, _ = _check_girth(n, A, g)
                if valid:
                    return n, edges
            n += 2

    # For large girth (g >= 10), random search is infeasible.
    # Return L-U-W construction.
    return fallback()
