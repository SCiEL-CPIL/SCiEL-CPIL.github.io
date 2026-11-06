# EVOLVE-BLOCK-START
import time
import random
import sys

sys.setrecursionlimit(100000)

_DEADLINE = time.time() + 145.0  # hard wall-clock safety margin before the 150s kill

# ---------------------------------------------------------------------------
# Maximum Independent Set via branch-and-bound (Tomita style + greedy coloring)
# ---------------------------------------------------------------------------
def _max_independent_set(n, adj):
    """Return a maximum independent set (as a set of ints) of the graph.

    adj: list of sets (adjacency).  Uses a Tomita-style branch and bound with
    greedy coloring to bound the search.  Deterministic ordering.
    """
    # Precompute adjacency as bitmasks for fast set operations.
    mask = [0] * n
    for v in range(n):
        m = 0
        for u in adj[v]:
            m |= (1 << u)
        mask[v] = m

    best = [0]          # best size found
    best_set = [set()]  # best independent set found
    cur = [set()]       # current independent set (vertices chosen on the path)

    def color_sort(P):
        """Greedy coloring of P (ordered by degree).  Returns (order, colors)
        where colors[i] is the color (1-based) assigned to order[i]."""
        P_list = sorted(P, key=lambda v: (len(P & mask[v]), v))
        # available colors: a list of bitmasks; color c has bitmask 1<<c
        col_masks = []
        color = [0] * len(P_list)
        for idx, v in enumerate(P_list):
            nb = mask[v] & P
            # find the smallest color c such that no vertex of that color is adjacent to v
            c = 0
            for cm in col_masks:
                if cm & nb == 0:
                    break
                c += 1
            else:
                col_masks.append(0)
                c = len(col_masks) - 1
            col_masks[c] |= (1 << v)
            color[idx] = c + 1
        return P_list, color

    def expand(P):
        # P is a set of candidate vertices (all mutually reachable, independent of cur)
        if not P:
            return
        if len(P) + len(cur[0]) <= best[0]:
            return
        order, colors = color_sort(P)
        maxc = colors[-1]  # upper bound = number of colors
        if len(cur[0]) + maxc <= best[0]:
            return
        # iterate candidates in reverse color order (highest color first)
        for i in range(len(order) - 1, -1, -1):
            if time.time() > _DEADLINE:
                return
            v = order[i]
            # bound: cur + (vertices of color <= colors[i])
            if len(cur[0]) + colors[i] <= best[0]:
                return
            new_cur = cur[0] | {v}
            cur[0] = new_cur
            if len(new_cur) > best[0]:
                best[0] = len(new_cur)
                best_set[0] = set(new_cur)
            new_P = P & mask[v]
            expand(new_P)
            cur[0] = cur[0] - {v}
            P = P - {v}

    expand(set(range(n)))
    return best_set[0]


def _solve_instance(n, p, seed):
    """Build the random graph (matching the verifier's construction) and return
    the smallest vertex list meeting every maximal clique of size >= 2, i.e.
    the complement of a maximum independent set."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)

    # Independent set T must have no edge inside it -> S = V \ T meets every
    # maximal clique (every edge's endpoints are not both in T).
    T = _max_independent_set(n, adj)
    S = set(range(n)) - T
    return sorted(S)


def construct():
    start = time.time()
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    for (n, p, seed) in instances:
        if time.time() > _DEADLINE - 2.0:
            # Not enough time: fall back to a valid (if not minimal) answer.
            results.append(list(range(n)))
            continue
        results.append(_solve_instance(n, p, seed))
    return results
# EVOLVE-BLOCK-END
