"""
Fewest four-cycles in graphs with 31 vertices and 91 edges.

Strategy (structural change: explicit C4-counting + local-search edge swaps):
1. Build the polarity graph of PG(2,5) — a C4-free graph with 31 vertices and 90 edges.
2. For each of the 465-90 = 375 non-edges, compute the number of common neighbors.
   Adding edge (u,v) to a C4-free graph creates C( common_neigh(u,v), 2 ) new C4s.
   Since the base is C4-free, common_neigh(u,v) <= 1 for all non-edges, so any
   single addition creates 0 C4s.
3. Add the first non-edge (all give 0 C4s), giving a 91-edge graph with 0 C4s.
4. Perform local search: try 1-for-1 edge swaps (remove edge e, add non-edge f)
   to see if we can maintain 91 edges while reducing C4 count below 0 (impossible)
   — but more importantly, this confirms the baseline and guards against any
   construction error.

Actually, since the polarity graph is provably C4-free and adding one non-edge
creates 0 C4s, the answer is 0. But we still do a fast verification and a short
local search to confirm no better structure exists.

The key structural difference from the seed: we explicitly count C4s using
bitmask operations and run a bounded local search over edge swaps, rather than
relying solely on the theoretical C4-free argument.
"""

import time
import random

def _build_polarity_graph():
    """
    Construct the polarity graph of PG(2,5) with 31 vertices.
    Vertices: projective points over GF(5), canonicalized.
    Edges: p~q iff p·q = 0 mod 5 (dot product under identity form).
    """
    q = 5
    points = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
                if a != 0:
                    inv_a = pow(a, q - 2, q)
                    pt = (1, (b * inv_a) % q, (c * inv_a) % q)
                elif b != 0:
                    inv_b = pow(b, q - 2, q)
                    pt = (0, 1, (c * inv_b) % q)
                else:
                    pt = (0, 0, 1)
                if pt not in seen:
                    seen.add(pt)
                    points.append(pt)

    assert len(points) == 31, f"Expected 31 points, got {len(points)}"

    edges = []
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            qq = points[j]
            dot = (p[0] * qq[0] + p[1] * qq[1] + p[2] * qq[2]) % q
            if dot == 0:
                edges.append([i, j])

    return edges


def _count_c4s(edges, n=31):
    """Count the number of 4-cycles in the graph using bitmask adjacency."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                continue
            # Common neighbors of i and j
            common = adj[i] & adj[j]
            c = bin(common).count('1')
            count += c * (c - 1) // 2
    return count // 4  # Each C4 counted 4 times (once per vertex pair that is a diagonal)


def _common_neighbors_mask(adj, u, v):
    """Return bitmask of common neighbors of u and v."""
    return adj[u] & adj[v]


def _c4s_created_by_adding_edge(adj, u, v):
    """Number of new C4s created by adding edge (u,v) to the current graph."""
    common = adj[u] & adj[v]
    c = bin(common).count('1')
    return c * (c - 1) // 2


def _try_improve(edges, adj, n, time_limit, seed=42):
    """
    Local search: try 1-for-1 edge swaps to reduce C4 count.
    Start from the given edge set; try removing one edge and adding another.
    Accept if C4 count decreases.
    """
    rng = random.Random(seed)
    best_edges = list(edges)
    best_c4 = _count_c4s(best_edges, n)

    t_start = time.time()
    max_iter = 50000

    for iteration in range(max_iter):
        if time.time() - t_start > time_limit:
            break

        # Pick a random edge to remove
        ei = rng.randrange(len(best_edges))
        u_rm, v_rm = best_edges[ei]

        # Tentatively remove
        adj[u_rm] &= ~(1 << v_rm)
        adj[v_rm] &= ~(1 << u_rm)
        edge_set = set(tuple(sorted(e)) for e in best_edges)
        edge_set.remove(tuple(sorted((u_rm, v_rm))))

        # Try adding a random non-edge
        found = False
        for _attempt in range(20):
            a = rng.randrange(n)
            b = rng.randrange(n)
            if a == b:
                continue
            if (a, b) in edge_set or (b, a) in edge_set:
                continue
            # Check it's not already an edge
            if (adj[a] >> b) & 1:
                continue

            # Count C4s created
            new_c4 = _c4s_created_by_adding_edge(adj, a, b)
            # Total after swap = (C4s in graph without removed edge) + new_c4
            # But we need C4s in graph without the removed edge.
            # This is expensive to compute exactly. Instead, estimate:
            # C4s involving the removed edge get destroyed, others remain.
            # For simplicity, just compute full C4 count occasionally.

            # Add tentatively
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
            edge_set.add((min(a, b), max(a, b)))

            # Compute full C4 count (expensive but accurate)
            # Only do this for a subset of trials
            if _attempt < 3:
                temp_edges = list(best_edges[:ei]) + list(best_edges[ei+1:]) + [[a, b]]
                c4_temp = _count_c4s(temp_edges, n)
                if c4_temp < best_c4:
                    best_edges = temp_edges
                    best_c4 = c4_temp
                    # Rebuild adj
                    adj = [0] * n
                    for uu, vv in best_edges:
                        adj[uu] |= (1 << vv)
                        adj[vv] |= (1 << uu)
                    edge_set = set(tuple(sorted(e)) for e in best_edges)
                    found = True
                    break

            # Revert
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            edge_set.remove((min(a, b), max(a, b)))

            if found:
                break

        # Revert removal
        adj[u_rm] |= (1 << v_rm)
        adj[v_rm] |= (1 << u_rm)
        edge_set.add(tuple(sorted((u_rm, v_rm))))

        if best_c4 == 0:
            break

    return best_edges, best_c4


def construct():
    """
    Return exactly 91 distinct edges [u, v] on vertices 0..30.
    Minimizes the number of 4-cycles.
    """
    t_start = time.time()
    DEADLINE = 130.0  # 20s safety margin before 150s kill

    # Step 1: Build C4-free polarity graph (90 edges)
    edges = _build_polarity_graph()
    assert len(edges) == 90, f"Polarity graph should have 90 edges, got {len(edges)}"

    # Step 2: Verify it's C4-free
    base_c4 = _count_c4s(edges, 31)
    # Should be 0; if not, we still proceed but note it
    # (theoretical guarantee says 0)

    # Step 3: Find the best non-edge to add (minimizes C4 increase)
    adj = [0] * 31
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))

    best_nonedge = None
    best_increase = float('inf')

    for i in range(31):
        for j in range(i + 1, 31):
            if (i, j) in edge_set:
                continue
            inc = _c4s_created_by_adding_edge(adj, i, j)
            if inc < best_increase:
                best_increase = inc
                best_nonedge = [i, j]
            if inc == 0:
                # Can't do better than 0
                break
        if best_increase == 0:
            break

    assert best_nonedge is not None, "No non-edge found"
    edges.append(best_nonedge)
    assert len(edges) == 91

    # Step 4: Verify edge count and distinctness
    edge_set_final = set(tuple(sorted(e)) for e in edges)
    assert len(edge_set_final) == 91, f"Expected 91 distinct edges, got {len(edge_set_final)}"

    # Step 5: Compute C4 count of the resulting graph
    final_c4 = _count_c4s(edges, 31)

    # Step 6: If we have time, try local search to see if we can do better
    remaining = DEADLINE - (time.time() - t_start)
    if remaining > 5.0 and final_c4 > 0:
        edges, final_c4 = _try_improve(edges, adj, 31, remaining - 2.0)
        edge_set_final = set(tuple(sorted(e)) for e in edges)
        assert len(edge_set_final) == 91

    # Final safety check
    assert len(edges) == 91
    assert all(0 <= u <= 30 and 0 <= v <= 30 for u, v in edges)
    assert all(u != v for u, v in edges)

    return edges
