"""Fewest four-cycles in graphs with 31 vertices and 91 edges.

Strategy: build a C4-free polarity graph of PG(2,5) with 90 edges, then add
the single edge that creates the fewest C4s.  The polarity graph of PG(2,5)
is known to be C4-free.  Adding one edge creates C4s only through 3-paths
between the new edge's endpoints.  We try all 8 bilinear-form variants to
ensure we get exactly 90 edges, then do a local-search polish (edge swaps
that keep 91 edges) to reduce C4 count further.

The C4 count is computed exactly via:
  C4 = (1/2) * sum_{i<j} C(common_neighbors(i,j), 2)
"""
import time
import random
import numpy as np


def _pg2_points(q):
    """Generate all points of PG(2, q-1) over GF(q) as normalized tuples."""
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                idx = 0
                while idx < 3 and (x, y, z)[idx] == 0:
                    idx += 1
                factor = (x, y, z)[idx]
                inv_factor = pow(factor, q - 2, q)
                norm = tuple((x, y, z)[i] * inv_factor % q for i in range(3))
                if norm not in seen:
                    seen.add(norm)
                    points.append(norm)
    return points


def _bilinear(p, q_pt, q_mod=5, variant=0):
    """Symmetric bilinear form for polarity graph. variant selects coefficients."""
    x1, y1, z1 = p
    x2, y2, z2 = q_pt
    m = q_mod
    if variant == 0:
        return (x1 * x2 + y1 * y2 + z1 * z2 + x1 * y2 + y1 * x2) % m
    elif variant == 1:
        return (x1 * x2 + y1 * y2 + z1 * z2 + x1 * y2 + y1 * x2 + z1 * x2 + z2 * x1) % m
    elif variant == 2:
        return (x1 * x2 + y1 * y2 + z1 * z2 + x1 * y2 + y1 * x2 + z1 * y2 + z2 * y1) % m
    elif variant == 3:
        return (x1 * x2 + y1 * y2 + z1 * z2 + 2 * x1 * y2 + 2 * y1 * x2) % m
    elif variant == 4:
        return (x1 * x2 + y1 * y2 + z1 * z2 + x1 * z2 + z1 * x2) % m
    elif variant == 5:
        return (x1 * x2 + y1 * y2 + z1 * z2 + y1 * z2 + z1 * y2) % m
    elif variant == 6:
        return (x1 * x2 + y1 * y2 + z1 * z2 + 3 * x1 * y2 + 3 * y1 * x2) % m
    elif variant == 7:
        return (x1 * x2 + y1 * y2 + z1 * z2 + x1 * y2 + y1 * x2 + x1 * z2 + z1 * x2) % m
    return (x1 * x2 + y1 * y2 + z1 * z2) % m


def _count_c4(adj, n):
    """Exact C4 count: (1/2) * sum_{i<j} C(cn(i,j), 2) where cn = (A^2)[i,j]."""
    A = adj.astype(np.int32)
    A2 = A @ A
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            cn = A2[i, j]
            if cn >= 2:
                total += cn * (cn - 1) // 2
    return total // 2


def _c4s_containing_edge(adj, i, j, n):
    """Count C4s that contain the specific edge (i,j).
    
    A C4 containing edge (i,j) has the form i-j-a-b-i where:
    - a is a neighbor of j (a != i)
    - b is a neighbor of i (b != j)
    - a-b is an edge
    - a != b
    """
    count = 0
    for a in range(n):
        if a == i or a == j:
            continue
        if adj[j, a]:
            for b in range(n):
                if b == i or b == j or b == a:
                    continue
                if adj[i, b] and adj[a, b]:
                    count += 1
    return count


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    n = 31
    target_edges = 91

    points = _pg2_points(5)
    assert len(points) == 31

    # Find a polarity graph with exactly 90 edges (C4-free)
    best_adj = None
    for variant in range(8):
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if _bilinear(points[i], points[j], 5, variant) == 0:
                    adj[i, j] = 1
                    adj[j, i] = 1
        ec = int(adj.sum() // 2)
        if ec == 90:
            best_adj = adj
            break
        if best_adj is None:
            best_adj = adj

    if best_adj is None:
        # Fallback: random graph
        random.seed(42)
        best_adj = np.zeros((n, n), dtype=np.int8)
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        for i, j in all_pairs[:target_edges]:
            best_adj[i, j] = 1
            best_adj[j, i] = 1

    adj = best_adj.copy()
    current_edges = int(adj.sum() // 2)

    # If we have 90 edges, add the best non-edge
    if current_edges < target_edges:
        need = target_edges - current_edges
        # Score each non-edge by C4s it would create
        # A new edge (i,j) creates C4s = number of 3-paths from i to j not using (i,j)
        # = sum over a in N(j)\{i} of |N(a) ∩ N(i) \ {j}|
        # But more simply: just compute C4 count after adding
        non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] == 0]
        # For speed, estimate C4s created by adding (i,j):
        # new C4s = number of pairs (a,b) such that a~j, a~b, b~i, and a!=i, b!=j, a!=b
        # which is exactly _c4s_containing_edge on the current graph (before adding)
        # Wait: _c4s_containing_edge counts C4s that would contain edge (i,j)
        # if it were present. But we're adding it, so the C4s it creates are exactly
        # the number of 3-paths from i to j in the current graph.
        # A 3-path from i to j: i-b-a-j where b~i, a~j, b~a, b!=j, a!=i, b!=a
        # This is the same as _c4s_containing_edge(adj, i, j, n)
        scores = []
        for (i, j) in non_edges:
            s = _c4s_containing_edge(adj, i, j, n)
            scores.append((s, i, j))
        scores.sort()
        # Add the best 'need' edges
        for s, i, j in scores[:need]:
            adj[i, j] = 1
            adj[j, i] = 1

    elif current_edges > target_edges:
        # Remove edges to get to target
        while int(adj.sum() // 2) > target_edges:
            edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            # Remove the edge in the most C4s
            best_remove = None
            best_count = -1
            for (i, j) in edges:
                c = _c4s_containing_edge(adj, i, j, n)
                if c > best_count:
                    best_count = c
                    best_remove = (i, j)
            if best_remove:
                i, j = best_remove
                adj[i, j] = 0
                adj[j, i] = 0

    current_c4 = _count_c4(adj, n)
    best_adj = adj.copy()
    best_c4 = current_c4

    # Local search: swap edges to reduce C4 count, keeping exactly 91 edges
    random.seed(2024)
    no_improve = 0
    max_no_improve = 200000

    while time.time() < deadline and no_improve < max_no_improve:
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]]
        non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not adj[i, j]]

        if not edges or not non_edges:
            break

        improved = False
        # Try random swaps
        for _ in range(30):
            if time.time() > deadline:
                break
            i, j = random.choice(edges)
            k, l = random.choice(non_edges)

            # Check validity: removing (i,j) and adding (k,l) must keep edge count at 91
            # and not create a duplicate
            if (k, l) in [(a, b) for a, b in edges] or (l, k) in [(a, b) for a, b in edges]:
                continue
            # Make sure (k,l) is not the same as (i,j)
            if {k, l} == {i, j}:
                continue

            adj_temp = adj.copy()
            adj_temp[i, j] = 0
            adj_temp[j, i] = 0
            adj_temp[k, l] = 1
            adj_temp[l, k] = 1

            if int(adj_temp.sum() // 2) != target_edges:
                continue

            c4_temp = _count_c4(adj_temp, n)
            if c4_temp < best_c4:
                adj = adj_temp
                best_adj = adj.copy()
                best_c4 = c4_temp
                no_improve = 0
                improved = True
                break
            else:
                no_improve += 1

        if improved:
            continue

        # Structured move: remove worst edge, add best non-edge
        if no_improve % 200 == 0 and time.time() < deadline:
            edge_c4 = {}
            for (i, j) in edges:
                if time.time() > deadline:
                    break
                edge_c4[(i, j)] = _c4s_containing_edge(adj, i, j, n)

            if edge_c4:
                worst = max(edge_c4, key=edge_c4.get)

                ne_c4 = {}
                for (k, l) in non_edges[:300]:
                    if time.time() > deadline:
                        break
                    ne_c4[(k, l)] = _c4s_containing_edge(adj, k, l, n)

                if ne_c4:
                    best_ne = min(ne_c4, key=ne_c4.get)
                    i, j = worst
                    k, l = best_ne

                    adj_temp = adj.copy()
                    adj_temp[i, j] = 0
                    adj_temp[j, i] = 0
                    adj_temp[k, l] = 1
                    adj_temp[l, k] = 1

                    if int(adj_temp.sum() // 2) == target_edges:
                        c4_temp = _count_c4(adj_temp, n)
                        if c4_temp < best_c4:
                            adj = adj_temp
                            best_adj = adj.copy()
                            best_c4 = c4_temp
                            no_improve = 0

    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j]:
                result.append([i, j])

    assert len(result) == target_edges, f"Expected {target_edges} edges, got {len(result)}"

    return result
