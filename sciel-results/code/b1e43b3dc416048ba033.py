"""Rewritten: maximize edges of a triangle-free, non-3-colorable graph on 16 vertices."""
import time
import random
from itertools import combinations

N = 16
DEADLINE = time.time() + 115.0  # safety margin before 150s kill

# All potential edges (u < v)
ALL_EDGES = [(u, v) for u in range(N) for v in range(u + 1, N)]
EDGE_INDEX = {(u, v): i for i, (u, v) in enumerate(ALL_EDGES)}
M = len(ALL_EDGES)  # 120


def is_not_3_colorable(adj):
    """Bit-parallel exact 3-colorability check. Returns True if NOT 3-colorable."""
    # For each vertex, the set of colors forbidden by its already-colored neighbors.
    # We do backtracking over vertices in a heuristic order.
    # adj: list of int bitmasks of neighbors.

    # Precompute degree for ordering
    order = list(range(N))
    order.sort(key=lambda x: -adj[x].bit_count())

    # We assign colors 0,1,2. color[v] = -1 unassigned.
    color = [-1] * N
    # forbidden[v] bitmask: colors that are NOT allowed for v (due to colored neighbors)
    forbidden = [0] * N

    def backtrack(idx):
        if idx == N:
            return True
        v = order[idx]
        # Try each color not in forbidden[v]
        avail = 7 & ~forbidden[v]
        while avail:
            c = avail & (-avail)
            avail -= c
            color[v] = c
            # Propagate to uncolored neighbors
            nb = adj[v]
            affected = []
            while nb:
                w = (nb & (-nb)).bit_length() - 1
                nb -= nb & (-nb)
                if color[w] == -1:
                    forbidden[w] |= c
                    affected.append(w)
            if backtrack(idx + 1):
                return True
            # undo
            for w in affected:
                forbidden[w] &= ~c
        color[v] = -1
        return False

    return not backtrack(0)


def is_triangle_free(adj):
    for u in range(N):
        a = adj[u]
        while a:
            v = (a & (-a)).bit_length() - 1
            a -= a & (-a)
            if adj[u] & adj[v]:
                return False
    return True


def greedy_complete(adj, base_edges):
    """Add any edge that keeps triangle-free and non-3-colorable. Returns (adj, edges)."""
    edges = list(base_edges)
    added = True
    while added:
        added = False
        for (u, v) in ALL_EDGES:
            if u == v:
                continue
            if adj[u] >> v & 1:
                continue
            # triangle check
            if adj[u] & adj[v]:
                continue
            # add edge
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            if is_not_3_colorable(adj):
                edges.append([u, v])
                added = True
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
    return adj, edges


def random_base(seed):
    """Build a random triangle-free graph, then try to make it non-3-colorable via greedy add."""
    rng = random.Random(seed)
    # Start with a random sparse triangle-free graph
    adj = [0] * N
    edges = []
    # Random triangle-free generation: pick random edges, reject if creates triangle
    cand = ALL_EDGES[:]
    rng.shuffle(cand)
    for (u, v) in cand:
        if adj[u] & adj[v]:
            continue
        # adding u-v: need no common neighbor (triangle-free)
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        edges.append([u, v])
        if len(edges) >= 40:
            break
    # Now greedily complete: add edges preserving triangle-free AND non-3-colorable
    adj, edges = greedy_complete(adj, edges)
    return adj, edges


def construct():
    best_edges = None
    best_len = -1

    # First, a strong deterministic seed: Mycielski (Grötzsch) on 11 vertices + greedy
    # Build Grötzsch graph (triangle-free, non-3-colorable, 11 vertices)
    adj = [0] * N
    edges = []

    def add(u, v):
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        edges.append([u, v])

    # Pentagon 0-4
    for i in range(5):
        add(i, (i + 1) % 5)
    # Mycielski: vertex 5+i connects to i and (i+1)%5? Standard Mycielski of C5:
    # Original vertices 0..4. New vertices 5..9 where 5+i adjacent to i and (i-1)%5.
    # Then vertex 10 adjacent to all of 5..9.
    for i in range(5):
        add(5 + i, i)
        add(5 + i, (i - 1) % 5)
    for i in range(5):
        add(10, 5 + i)

    # Greedily complete
    adj, edges = greedy_complete(adj, edges)
    best_edges = edges
    best_len = len(edges)

    # Try random bases
    t0 = time.time()
    seed = 0
    while time.time() < DEADLINE:
        seed += 1
        adj, edges = random_base(seed)
        if len(edges) > best_len:
            best_len = len(edges)
            best_edges = edges
        # periodic check
        if seed % 200 == 0 and time.time() > DEADLINE:
            break

    return best_edges


if __name__ == "__main__":
    e = construct()
    print(len(e))
