"""C6-free graph on 40 vertices: bitset greedy + local search + restarts."""
import time
import random

N = 40
DEADLINE = 110.0  # safety margin before 120s wall-clock kill


def _has_c6_bitset(adj, u, v):
    """Check if adding edge (u,v) creates a C6.
    Uses bitset intersection: for each a in N(u), b in N(v), check if N(a) & N(b) has
    a common neighbor (other than u/v endpoints that would form shorter cycles).
    A C6 is u-a-?-b-v-? ... actually we need a 6-cycle containing edge (u,v).
    A 6-cycle containing edge (u,v) has form u - x - y - z - w - v - u
    where x in N(u), w in N(v), and there is a path x-y-z-w of length 3
    (i.e., a 3-edge path from x to w) in G - {u,v}.
    So: exists x in N(u), w in N(v) such that dist_{G-{u,v}}(x,w) == 3.
    Equivalently: exists x in N(u), w in N(v), y in N(x)∩(G-{u,v}), z in N(w)∩(G-{u,v})
    with y-z an edge, y != x, z != w, and {x,y,z,w} disjoint from {u,v}, y != w, z != x.
    """
    adj_u = adj[u] & ~(1 << v)
    adj_v = adj[v] & ~(1 << u)
    if adj_u == 0 or adj_v == 0:
        return False
    # For each x in N(u) \ {v}, w in N(v) \ {u}:
    #   need a 3-path x-y-z-w in G - {u,v}
    #   i.e., exists y in N(x) \ {u,v,w} and z in N(w) \ {u,v,x} with y-z edge
    #   and y != z, y != x, z != w, y != w, z != x
    #   Simplify: y in N(x) & ~(1<<u) & ~(1<<v) & ~(1<<w)
    #             z in N(w) & ~(1<<u) & ~(1<<v) & ~(1<<x)
    #             need y in N(z) (edge y-z), y != z
    # To speed up: precompute for the pair (x,w) whether there's a length-3 path.
    # Use: N(x) & ~(1<<u) & ~(1<<v) & ~(1<<w)  =  Sx
    #      N(w) & ~(1<<u) & ~(1<<v) & ~(1<<x)  =  Sw
    #      need exists y in Sx, z in Sw: y in N(z) and y != z
    #      = (N(z) & Sx & ~(1<<z)) != 0 for some z in Sw
    #      = OR over z in Sw of (N(z) & Sx & ~(1<<z)) != 0
    #      = (Sx & (union over z in Sw of (N(z) & ~(1<<z)))) != 0
    # Precompute: for a given set Sw, U = union of N(z) & ~(1<<z) for z in Sw
    # But Sw depends on x. Let's just iterate.

    # Enumerate x in adj_u, w in adj_v
    xu = adj_u
    xv = adj_v
    mask_uv = (1 << u) | (1 << v)
    while xu:
        xb = xu & (-xu)
        x = xb.bit_length() - 1
        xu ^= xb
        # Sx = N(x) minus u, v, and we'll subtract w per iteration
        base_sx = adj[x] & ~mask_uv
        xv2 = xv
        while xv2:
            wb = xv2 & (-xv2)
            w = wb.bit_length() - 1
            xv2 ^= wb
            Sx = base_sx & ~(1 << w)
            if Sx == 0:
                continue
            Sw = adj[w] & ~mask_uv & ~(1 << x)
            if Sw == 0:
                continue
            # Need exists z in Sw such that N(z) & Sx & ~(1<<z) != 0
            # i.e., Sx & (OR_{z in Sw} (N(z) & ~(1<<z))) != 0
            # Compute union:
            union = 0
            sw2 = Sw
            while sw2:
                zb = sw2 & (-sw2)
                z = zb.bit_length() - 1
                sw2 ^= zb
                union |= adj[z] & ~(1 << z)
            if Sx & union:
                return True
    return False


def _is_c6_free(adj, edges):
    """Verify the entire graph is C6-free by checking every edge."""
    for (u, v) in edges:
        # Temporarily consider edge (u,v) as present; check if there's a 6-cycle through it
        # We need to check in the graph WITHOUT (u,v) whether there's a 3-path between
        # some x in N(u)\{v} and w in N(v)\{u}.
        # Actually simpler: build adj without (u,v) and check _has_c6_bitset on that.
        # But that's expensive. Instead, just check directly.
        adj_u = adj[u] & ~(1 << v)
        adj_v = adj[v] & ~(1 << u)
        if adj_u == 0 or adj_v == 0:
            continue
        mask_uv = (1 << u) | (1 << v)
        xu = adj_u
        while xu:
            xb = xu & (-xu)
            x = xb.bit_length() - 1
            xu ^= xb
            base_sx = adj[x] & ~mask_uv
            xv2 = adj_v
            while xv2:
                wb = xv2 & (-xv2)
                w = wb.bit_length() - 1
                xv2 ^= wb
                Sx = base_sx & ~(1 << w)
                if Sx == 0:
                    continue
                Sw = adj[w] & ~mask_uv & ~(1 << x)
                if Sw == 0:
                    continue
                union = 0
                sw2 = Sw
                while sw2:
                    zb = sw2 & (-sw2)
                    z = zb.bit_length() - 1
                    sw2 ^= zb
                    union |= adj[z] & ~(1 << z)
                if Sx & union:
                    return False
    return True


def _greedy_build(seed):
    """Greedy construction: add edges in order of some heuristic."""
    rng = random.Random(seed)
    adj = [0] * N
    edges = []
    # Random permutation of all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(all_edges)
    for (u, v) in all_edges:
        if _has_c6_bitset(adj, u, v):
            continue
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edges.append((u, v))
    return edges


def _local_search(edges, adj, max_iters=2000):
    """Try to improve via edge swaps: remove edge e, add edge f (not in graph)
    if the result is still C6-free and we gain an edge (net +1 or same count but
    different structure enabling more). Actually just try to add edges we missed.
    Better: try swapping: remove one edge, try adding two edges."""
    edge_set = set(edges)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # Try adding any missing edge
        for i in range(N):
            for j in range(i + 1, N):
                if (i, j) in edge_set or (j, i) in edge_set:
                    continue
                if _has_c6_bitset(adj, i, j):
                    continue
                # Can add
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append((i, j))
                edge_set.add((i, j))
                improved = True
        # Try swaps: remove edge (a,b), add (c,d) and (e,f)
        if not improved:
            for (a, b) in list(edges):
                # Remove (a,b)
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                edges.remove((a, b))
                edge_set.discard((a, b))
                # Try to add two edges
                added = []
                for i in range(N):
                    for j in range(i + 1, N):
                        if (i, j) in edge_set or (j, i) in edge_set:
                            continue
                        if _has_c6_bitset(adj, i, j):
                            continue
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        edges.append((i, j))
                        edge_set.add((i, j))
                        added.append((i, j))
                        break
                    if len(added) == 1:
                        break
                if len(added) >= 1:
                    # Net gain: removed 1, added len(added)
                    if len(added) >= 1:
                        improved = True
                        continue
                # If not improved, revert
                for (i, j) in added:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                    edges.remove((i, j))
                    edge_set.discard((i, j))
                # Re-add (a,b)
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                edges.append((a, b))
                edge_set.add((a, b))
    return edges


def _rebuild_adj(edges):
    adj = [0] * N
    for (u, v) in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj


def construct():
    t0 = time.time()
    best_edges = None
    best_count = 0

    # Phase 1: multiple greedy restarts
    n_restarts = 50
    for seed in range(n_restarts):
        if time.time() - t0 > DEADLINE:
            break
        edges = _greedy_build(seed)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]

    # Phase 2: local search on the best
    if best_edges is not None:
        adj = _rebuild_adj(best_edges)
        best_edges = _local_search(best_edges, adj, max_iters=1000)
        best_count = len(best_edges)

    # Phase 3: more restarts if time remains, targeting better seeds
    t_now = time.time()
    remaining = DEADLINE - t_now
    if remaining > 10:
        # Try more restarts with different strategies
        extra = min(int(remaining / 2), 100)
        for seed in range(1000, 1000 + extra):
            if time.time() - t0 > DEADLINE:
                break
            edges = _greedy_build(seed)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges[:]
                # Quick local search
                adj = _rebuild_adj(best_edges)
                best_edges = _local_search(best_edges, adj, max_iters=500)
                best_count = len(best_edges)

    # Convert to list of [u, v]
    result = [[u, v] for (u, v) in best_edges]
    return result


# Fallback: if something goes wrong, return the seed solution
def _fallback():
    edges = []
    for k in range(8):
        edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
    return edges


# Wrap construct with safety
_orig_construct = construct
def construct():
    try:
        result = _orig_construct()
        # Validate: simple graph, no C6
        if not result:
            return _fallback()
        # Check simplicity
        seen = set()
        for e in result:
            u, v = e[0], e[1]
            if u == v or not (0 <= u < 40 and 0 <= v < 40):
                return _fallback()
            key = (min(u, v), max(u, v))
            if key in seen:
                return _fallback()
            seen.add(key)
        # Check C6-free
        adj = [0] * 40
        for e in result:
            adj[e[0]] |= (1 << e[1])
            adj[e[1]] |= (1 << e[0])
        if not _is_c6_free(adj, result):
            return _fallback()
        return result
    except Exception:
        return _fallback()
