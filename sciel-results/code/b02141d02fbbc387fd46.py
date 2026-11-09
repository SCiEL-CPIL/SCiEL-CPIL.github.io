"""Construct a dense C4-rainbow edge-coloured graph on 20 vertices."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    n = 20
    num_colours = 20

    def elapsed():
        return time.time() - start

    def time_ok():
        return elapsed() < deadline - 5.0

    # We'll build greedily. Represent edges as dict (u,v)->color with u<v.
    # adjacency for quick 4-cycle detection.
    def build_greedy(seed):
        rng = random.Random(seed)
        edge_color = {}  # (u,v) with u<v -> colour
        adj = {i: {} for i in range(n)}  # adj[u][v] = colour

        def add_edge(u, v, c):
            if u > v:
                u, v = v, u
            edge_color[(u, v)] = c
            adj[u][v] = c
            adj[v][u] = c

        def remove_edge(u, v):
            if u > v:
                u, v = v, u
            del edge_color[(u, v)]
            del adj[u][v]
            del adj[v][u]

        def is_valid_add(u, v, c):
            # Check if adding edge (u,v) with colour c creates a non-rainbow 4-cycle.
            # A 4-cycle containing (u,v) is u-v-w-x-u.
            # w must be neighbor of v, x must be neighbor of u.
            # (w,x) must be an edge.
            # u,v,w,x must be distinct.
            nv = adj[v]
            nu = adj[u]
            
            # Iterate over potential w (neighbor of v)
            for w, cw in nv.items():
                if w == u:
                    continue
                # Iterate over potential x (neighbor of u)
                for x, cxu in nu.items():
                    if x == v or x == w:
                        continue
                    # Check if w and x are connected
                    if x in adj[w]:
                        cwx = adj[w][x]
                        # Colours on cycle: c (u-v), cw (v-w), cwx (w-x), cxu (x-u)
                        # They must be all distinct
                        if len({c, cw, cwx, cxu}) != 4:
                            return False
            return True

        # Try to add all possible edges, choosing colour that works.
        # Order edges to encourage density.
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(all_pairs)

        for (u, v) in all_pairs:
            if not time_ok():
                break
            # Try each colour, pick first that works
            cols = list(range(num_colours))
            rng.shuffle(cols)
            placed = False
            for c in cols:
                if is_valid_add(u, v, c):
                    add_edge(u, v, c)
                    placed = True
                    break
            # if no colour works, skip this edge
        return edge_color

    # Try a few seeds, keep the densest valid result.
    best = None
    best_count = -1
    for seed in range(1000):
        if not time_ok():
            break
        ec = build_greedy(seed)
        cnt = len(ec)
        if cnt > best_count:
            best_count = cnt
            best = ec
        # Early stop if we reached max possible edges (190)
        if best_count >= 190:
            break

    if best is None:
        # fallback: star
        return [[0, v, 0] for v in range(1, n)]

    result = [[u, v, c] for (u, v), c in best.items()]
    return result
# EVOLVE-BLOCK-END
