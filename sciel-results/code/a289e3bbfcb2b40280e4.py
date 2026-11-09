import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 110  # safety margin before 120s limit
    N = 40

    # ---------- Fast C6 detection via adjacency matrix ----------
    def _init_adj():
        return np.zeros((N, N), dtype=np.uint8)

    def _creates_c6(adj, u, v):
        # Adding edge (u,v) creates a C6 iff there is a path of length 3 from u to v
        # (i.e., a path u-x-y-z-v with x,y,z distinct from u,v and from each other
        #  in the sense that the 4 intermediate vertices are distinct).
        # Path length 3: u -> x -> y -> z -> v where x,y,z are distinct and
        # none equals u or v.
        # We need: x in N(u), y in N(x)∩N(v) ... wait, path is u-x-y-z-v
        # so: x ~ u, y ~ x, z ~ y, z ~ v, all of x,y,z distinct and not u,v.
        # Equivalently: exists x in N(u), y in N(x)\{u}, z in N(y)\{x,u} with z in N(v)\{u,y}
        # and x != v, y != v, z != u.
        # Simpler: BFS/DFS from u up to depth 3, check if we can reach v.
        # But we must ensure the 4 vertices u,x,y,z,v are all distinct.
        # Since u != v (simple graph, no loops), we need x,y,z distinct from u,v and from each other.

        nu = np.nonzero(adj[u])[0]
        nv = np.nonzero(adj[v])[0]
        # x must be in N(u) and not v
        for x in nu:
            if x == v:
                continue
            # y in N(x), y != u, y != v
            nx = np.nonzero(adj[x])[0]
            for y in nx:
                if y == u or y == v:
                    continue
                # z in N(y) ∩ N(v), z != u, z != x, z != v
                ny = np.nonzero(adj[y])[0]
                for z in ny:
                    if z == u or z == x or z == v:
                        continue
                    if adj[z, v]:
                        return True
        return False

    def _creates_c6_fast(adj, u, v):
        # More efficient: precompute neighbor lists
        nu = adj[u].nonzero()[0]
        nv = adj[v].nonzero()[0]
        # We need x in N(u)\{v}, y in N(x)\{u,v}, z in N(y)∩N(v)\{u,x,v}
        # z must be in N(y) and N(v)
        # Let's iterate x in N(u)\{v}, then y in N(x)\{u,v}, then check if
        # (N(y) ∩ N(v)) has an element not in {u, x, v}
        nv_set = set(nv.tolist())
        for x in nu:
            if x == v:
                continue
            nx = adj[x].nonzero()[0]
            for y in nx:
                if y == u or y == v:
                    continue
                ny = adj[y].nonzero()[0]
                # Check if any z in ny is in nv_set and z not in {u,x,v}
                for z in ny:
                    if z in nv_set and z != u and z != x and z != v:
                        return True
        return False

    def _creates_c6_fastest(adj, u, v):
        # Use numpy for speed
        nu = adj[u]  # boolean/uint8 vector
        nv = adj[v]
        # For each x where nu[x]=1 and x!=v:
        #   for each y where adj[x,y]=1 and y!=u,v:
        #     check if (adj[y] & nv) has any 1 at position not in {u,x,v}
        # Vectorize the inner loop:
        # candidates_y = adj[x] & ~onehot(u) & ~onehot(v)
        # candidates_z = adj[y] & nv & ~onehot(u) & ~onehot(x) & ~onehot(v)
        # This is still a loop over x and y.
        # Let's just do the triple loop but with numpy masks.
        for x in range(N):
            if x == v or not adj[u, x]:
                continue
            # y in N(x), y != u, y != v
            y_mask = adj[x].copy()
            y_mask[u] = 0
            y_mask[v] = 0
            ys = y_mask.nonzero()[0]
            for y in ys:
                # z in N(y) ∩ N(v), z not in {u, x, v}
                z_mask = adj[y] & nv
                z_mask[u] = 0
                z_mask[x] = 0
                z_mask[v] = 0
                if z_mask.any():
                    return True
        return False

    def _add_edge(adj, u, v):
        adj[u, v] = 1
        adj[v, u] = 1

    def _remove_edge(adj, u, v):
        adj[u, v] = 0
        adj[v, u] = 0

    def _edge_count(adj):
        return int(adj.sum() // 2)

    def _greedy_build(adj, order):
        """Add edges in given order if they don't create C6."""
        for (u, v) in order:
            if not adj[u, v] and not _creates_c6_fastest(adj, u, v):
                _add_edge(adj, u, v)

    def _local_search(adj, max_steps=50000):
        """Try to add edges by removing one and adding two (or other moves)."""
        best_count = _edge_count(adj)
        no_improve = 0
        for step in range(max_steps):
            if time.time() > DEADLINE - 5:
                break
            # Try to add a random missing edge
            improved = False
            for attempt in range(20):
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v or adj[u, v]:
                    continue
                if not _creates_c6_fastest(adj, u, v):
                    _add_edge(adj, u, v)
                    improved = True
                    best_count = _edge_count(adj)
                    break
            if not improved:
                # Try swap: remove edge (a,b), add edges (c,d) and (e,f)
                # or remove one and add one
                # Try: remove (a,b), add (c,d)
                a, b = random.randint(0, N-1), random.randint(0, N-1)
                if a == b or not adj[a, b]:
                    continue
                c, d = random.randint(0, N-1), random.randint(0, N-1)
                if c == d or adj[c, d]:
                    continue
                # Remove (a,b)
                _remove_edge(adj, a, b)
                # Try to add (c,d)
                if not _creates_c6_fastest(adj, c, d):
                    _add_edge(adj, c, d)
                    # Try to also add another edge to gain net +1
                    e, f = random.randint(0, N-1), random.randint(0, N-1)
                    if e != f and not adj[e, f] and not _creates_c6_fastest(adj, e, f):
                        _add_edge(adj, e, f)
                        new_count = _edge_count(adj)
                        if new_count > best_count:
                            best_count = new_count
                            no_improve = 0
                        else:
                            # Revert the second add
                            _remove_edge(adj, e, f)
                            # Keep the swap if it doesn't decrease
                            if _edge_count(adj) >= best_count:
                                no_improve = 0
                            else:
                                # Revert
                                _remove_edge(adj, c, d)
                                _add_edge(adj, a, b)
                                no_improve += 1
                    else:
                        new_count = _edge_count(adj)
                        if new_count >= best_count:
                            no_improve = 0
                        else:
                            _remove_edge(adj, c, d)
                            _add_edge(adj, a, b)
                            no_improve += 1
                else:
                    # Revert
                    _add_edge(adj, a, b)
                    no_improve += 1
            if no_improve > 500:
                break
        return adj

    # ---------- Main: multi-start greedy + local search ----------
    best_adj = _init_adj()
    best_count = 0

    # Start with the 8*K5 baseline
    base_adj = _init_adj()
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                _add_edge(base_adj, 5 * k + a, 5 * k + b)
    best_adj = base_adj.copy()
    best_count = _edge_count(best_adj)  # 80

    # Try several random orderings for greedy completion
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    random.seed(42)

    for trial in range(30):
        if time.time() > DEADLINE - 10:
            break

        adj = best_adj.copy()
        # Shuffle remaining edges and greedily add
        remaining = [(u, v) for (u, v) in all_pairs if not adj[u, v]]
        random.shuffle(remaining)
        _greedy_build(adj, remaining)

        cnt = _edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj.copy()

        # Local search
        if time.time() > DEADLINE - 15:
            break
        adj = _local_search(adj)
        cnt = _edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj.copy()

    # Final attempt: try to add any missing edge to best_adj
    for (u, v) in all_pairs:
        if time.time() > DEADLINE - 3:
            break
        if not best_adj[u, v] and not _creates_c6_fastest(best_adj, u, v):
            _add_edge(best_adj, u, v)
            best_count = _edge_count(best_adj)

    # Extract edges
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_adj[u, v]:
                edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
