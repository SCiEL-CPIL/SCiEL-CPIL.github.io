"""Construct a 91-edge graph on 31 vertices minimizing the number of C4 subgraphs.

Strategy: start from the C4-free polarity graph of PG(2,5) (90 edges, 0 C4s),
then perform a 1-in/1-out edge swap (add one non-edge, remove one existing edge)
that minimizes the resulting C4 count, evaluated with fast bitset common-neighbor
intersections.
"""
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before the 150s kill

# --- Build the polarity graph of PG(2,5) (90 edges, C4-free) ---
# Points of PG(2,5): 31 = (5^3 - 1)/(5 - 1).  Use homogeneous coords mod 5
# with the canonical representative x0 != 0 or (x0==0 and x1 != 0) or
# (x0==x1==0 and x2==1).
def _build_polarity_graph():
    q = 5
    pts = []
    for x0 in range(q):
        for x1 in range(q):
            for x2 in range(q):
                if x0 == 0 and x1 == 0 and x2 == 0:
                    continue
                if x0 != 0:
                    pts.append((x0, x1, x2))
                elif x1 != 0:
                    pts.append((0, 1, x2))
                else:
                    pts.append((0, 0, 1))
    n = len(pts)  # 31
    assert n == 31, n

    # Orthogonality: <p, q> = p0*q0 + p1*q1 + p2*q2 == 0 (mod 5)
    pts_arr = np.array(pts, dtype=np.int64)
    # adjacency via dot product (symmetric polarity)
    dots = pts_arr @ pts_arr.T % q
    adj = (dots == 0)
    # remove self loops
    np.fill_diagonal(adj, False)
    # make symmetric (it already is, but be safe)
    adj = np.logical_or(adj, adj.T)
    # remove self again
    np.fill_diagonal(adj, False)

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append((i, j))
    return n, edges


def _count_c4(adj_bool, n):
    """Count C4 subgraphs = sum over pairs {u,v} of C(|N(u)&N(v)|, 2)."""
    # build uint64 neighbor masks
    masks = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        m = 0
        row = adj_bool[i]
        for j in range(n):
            if row[j]:
                m |= (1 << j)
        masks[i] = m
    total = 0
    for i in range(n):
        mi = masks[i]
        for j in range(i + 1, n):
            c = np.uint64(mi & masks[j]).bit_count()
            total += c * (c - 1) // 2
    return total


def _c4_delta_add(adj_bool, n, u, v):
    """Net C4 change when adding edge (u,v) (assuming not present)."""
    # build masks
    masks = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        m = 0
        row = adj_bool[i]
        for j in range(n):
            if row[j]:
                m |= (1 << j)
        masks[i] = m
    nu = np.uint64(1 << u)
    nv = np.uint64(1 << v)
    mi_u = masks[u] | nu
    mi_v = masks[v] | nv
    delta = 0
    for w in range(n):
        if w == u or w == v:
            continue
        mw = masks[w]
        cu = np.uint64(mi_u & mw).bit_count()
        cv = np.uint64(mi_v & mw).bit_count()
        delta += cu * (cu - 1) // 2 + cv * (cv - 1) // 2
    # pair (u,v) itself
    cuv = np.uint64(mi_u & mi_v).bit_count()
    delta += cuv * (cuv - 1) // 2
    return delta


def _c4_delta_remove(adj_bool, n, u, v):
    """Net C4 change when removing edge (u,v) (assuming present)."""
    masks = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        m = 0
        row = adj_bool[i]
        for j in range(n):
            if row[j]:
                m |= (1 << j)
        masks[i] = m
    nu = np.uint64(1 << u)
    nv = np.uint64(1 << v)
    mi_u = masks[u] & ~nu
    mi_v = masks[v] & ~nv
    delta = 0
    for w in range(n):
        if w == u or w == v:
            continue
        mw = masks[w]
        cu = np.uint64(mi_u & mw).bit_count()
        cv = np.uint64(mi_v & mw).bit_count()
        delta += cu * (cu - 1) // 2 + cv * (cv - 1) // 2
    cuv = np.uint64(mi_u & mi_v).bit_count()
    delta += cuv * (cuv - 1) // 2
    return delta


def construct():
    n, edges = _build_polarity_graph()
    assert len(edges) == 90, len(edges)

    adj = np.zeros((n, n), dtype=bool)
    for (u, v) in edges:
        adj[u, v] = True
        adj[v, u] = True

    base_c4 = _count_c4(adj, n)

    edge_set = set(edges)
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i, j]:
                non_edges.append((i, j))

    # Explore 1-in/1-out swaps.  Precompute add-deltas for all non-edges,
    # remove-deltas for all edges, then pick the swap minimizing the sum.
    # To keep runtime bounded, sample a manageable subset of pairs.

    best_swap = None
    best_total = base_c4  # no swap baseline

    # Precompute masks once for reuse
    masks = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        m = 0
        row = adj[i]
        for j in range(n):
            if row[j]:
                m |= (1 << j)
        masks[i] = m

    def _delta_add_fast(u, v):
        nu = np.uint64(1 << u)
        nv = np.uint64(1 << v)
        mi_u = masks[u] | nu
        mi_v = masks[v] | nv
        delta = 0
        for w in range(n):
            if w == u or w == v:
                continue
            mw = masks[w]
            cu = np.uint64(mi_u & mw).bit_count()
            cv = np.uint64(mi_v & mw).bit_count()
            delta += cu * (cu - 1) // 2 + cv * (cv - 1) // 2
        cuv = np.uint64(mi_u & mi_v).bit_count()
        delta += cuv * (cuv - 1) // 2
        return delta

    def _delta_remove_fast(u, v):
        nu = np.uint64(1 << u)
        nv = np.uint64(1 << v)
        mi_u = masks[u] & ~nu
        mi_v = masks[v] & ~nv
        delta = 0
        for w in range(n):
            if w == u or w == v:
                continue
            mw = masks[w]
            cu = np.uint64(mi_u & mw).bit_count()
            cv = np.uint64(mi_v & mw).bit_count()
            delta += cu * (cu - 1) // 2 + cv * (cv - 1) // 2
        cuv = np.uint64(mi_u & mi_v).bit_count()
        delta += cuv * (cuv - 1) // 2
        return delta

    # Evaluate all non-edges for add-delta (cheap enough: ~465 non-edges * 31)
    add_deltas = {}
    for (u, v) in non_edges:
        add_deltas[(u, v)] = _delta_add_fast(u, v)
        if time.time() > _DEADLINE - 30:
            break

    # Evaluate all edges for remove-delta
    rem_deltas = {}
    for (u, v) in edges:
        rem_deltas[(u, v)] = _delta_remove_fast(u, v)
        if time.time() > _DEADLINE - 20:
            break

    # For each add, find the best remove (min add+rem).  Since remove-deltas
    # are independent of the add, we can just take the global min remove.
    if add_deltas and rem_deltas:
        min_rem = min(rem_deltas.values())
        # best add
        best_add = min(add_deltas, key=lambda k: add_deltas[k])
        best_total = base_c4 + add_deltas[best_add] + min_rem
        best_swap = (best_add, min_rem)

    # Apply the best swap
    if best_swap is not None:
        add_edge, rem_val = best_swap
        # find which edge to remove
        rem_edge = min(rem_deltas, key=lambda k: rem_deltas[k])
        new_edges = [e for e in edges if e != rem_edge]
        new_edges.append(add_edge)
    else:
        # fallback: just add the non-edge with smallest add-delta
        if add_deltas:
            best_add = min(add_deltas, key=lambda k: add_deltas[k])
            new_edges = list(edges) + [best_add]
        else:
            new_edges = list(edges) + [non_edges[0]]

    # Ensure exactly 91 distinct edges
    seen = set()
    out = []
    for (u, v) in new_edges:
        a, b = (u, v) if u < v else (v, u)
        key = (a, b)
        if key not in seen:
            seen.add(key)
            out.append([a, b])
    # if somehow short, pad with any non-edge
    while len(out) < 91:
        for (u, v) in non_edges:
            a, b = (u, v) if u < v else (v, u)
            if (a, b) not in seen:
                seen.add((a, b))
                out.append([a, b])
                break
    return out[:91]
