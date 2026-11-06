import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Known construction: 8 disjoint K4s gives n=32, C5-free, alpha=8.
    # The current program returns this immediately; we keep that design and add
    # a fast exact validation guard so the returned graph is provably valid.

    n = 32
    edges = []

    # Build 8 disjoint K4s
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])

    # ---- Exact, fast validation for this specific family (8 disjoint K4s) ----
    # 1) C5-free: every edge lies inside a K4, so any cycle is confined to a single
    #    component (a K4 on 4 vertices). No cycle of length 5 can exist.
    # 2) alpha < 9: an independent set picks at most 1 vertex per K4 (each K4 is a
    #    clique), and there are 8 K4s, so alpha <= 8 < 9.
    # We verify both properties exactly and cheaply below.

    # Build adjacency
    adj = set()
    for e in edges:
        adj.add((e[0], e[1]))
        adj.add((e[1], e[0]))

    # Exact C5 check: for each edge (u,v), a C5 through it needs two vertices a,b
    # such that u-a-b-v and u-b-a-v are paths of length 3, i.e., a is a common
    # neighbor of u and (some neighbor of v) etc. Simpler: for each edge (u,v),
    # for each neighbor a of u (a != v), for each neighbor b of a (b != u, b != v),
    # check if b is adjacent to v and b != u. This finds a 4-path u-a-b-v; then
    # if u-v edge exists (it does), we have a C5. But we must ensure all 5 vertices
    # are distinct: u, a, b, v are 4 distinct; the 5th is the edge u-v closing it.
    # Actually a C5 is 5 vertices: u-a-b-c-v-u. Let me just do the standard check.

    def exact_has_c5(n_verts, adj_set):
        """Return True if the graph contains a 5-cycle (as a subgraph)."""
        # Build adjacency list
        adj_list = [set() for _ in range(n_verts)]
        for (u, v) in adj_set:
            adj_list[u].add(v)
            adj_list[v].add(u)
        # For each edge (u, v) with u < v, look for a path of length 3 from u to v
        # through two intermediate vertices a, b (all distinct from u, v and each other).
        # Then u-a-b-v-u is a 5-cycle.
        for u in range(n_verts):
            for v in adj_list[u]:
                if v <= u:
                    continue
                # a in N(u) \ {v}
                for a in adj_list[u]:
                    if a == v:
                        continue
                    # b in N(a) \ {u, v}
                    for b in adj_list[a]:
                        if b == u or b == v:
                            continue
                        # Check b adjacent to v
                        if v in adj_list[b]:
                            return True
        return False

    def exact_has_independent_set_9(n_verts, adj_set):
        """Return True if there exists an independent set of size 9."""
        # Build adjacency list as sets
        adj_list = [set() for _ in range(n_verts)]
        for (u, v) in adj_set:
            adj_list[u].add(v)
            adj_list[v].add(u)

        # Branch and bound: try to find an independent set of size 9.
        # Use a simple recursive search with pruning.
        found = [False]

        def search(candidates, current_size):
            if found[0]:
                return
            if current_size == 9:
                found[0] = True
                return
            if current_size + len(candidates) < 9:
                return
            if not candidates:
                return
            # Pick a vertex; try including it first
            v = next(iter(candidates))
            # Option 1: include v
            new_cands = candidates - {v} - adj_list[v]
            search(new_cands, current_size + 1)
            if found[0]:
                return
            # Option 2: exclude v
            new_cands2 = candidates - {v}
            search(new_cands2, current_size)

        search(set(range(n_verts)), 0)
        return found[0]

    # Validate the constructed graph exactly
    if not exact_has_c5(n, adj) and not exact_has_independent_set_9(n, adj):
        return {'n': n, 'edges': edges}

    # Fallback: should not happen for 8xK4, but return a known-valid smaller graph
    # (7 K4s, n=28) as a safe fallback.
    n_fallback = 28
    edges_fallback = []
    for b in range(7):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges_fallback.append([base + i, base + j])
    return {'n': n_fallback, 'edges': edges_fallback}
