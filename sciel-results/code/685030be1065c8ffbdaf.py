import time
import sys

def construct():
    n = 15
    start_time = time.time()
    deadline = 75.0  # Safety margin before 80s limit

    def is_valid(edges):
        """Check if no simple 3-edge path has endpoints of equal degree."""
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)

        # Group vertices by degree
        from collections import defaultdict
        deg_groups = defaultdict(list)
        for i in range(n):
            deg_groups[deg[i]].append(i)

        for d, verts in deg_groups.items():
            if len(verts) < 2:
                continue
            # Check all pairs in same degree group
            for i in range(len(verts)):
                s = verts[i]
                for j in range(i + 1, len(verts)):
                    t = verts[j]
                    # Check if path s-u-v-t exists (simple, s != t)
                    found = False
                    for u in adj[s]:
                        if u == t:
                            continue
                        # v must be common neighbor of u and t, v != s, v != t
                        common = adj[u] & adj[t]
                        common.discard(s)
                        common.discard(t)
                        if common:
                            found = True
                            break
                    if found:
                        return False
        return True

    # Start with empty graph
    edges = set()

    # Generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

    # Greedy addition: try to add each edge if valid
    # Process in order that tends to maximize density
    # Strategy: try all edges, keep if valid
    added = 0
    for u, v in all_edges:
        if time.time() > deadline:
            break
        test_edges = edges | {(u, v)}
        if is_valid(test_edges):
            edges.add((u, v))
            added += 1

    # Try to add more edges with different orderings
    # Sort by sum of degrees (prefer adding edges between high-degree vertices first)
    def greedy_order(order_key):
        nonlocal edges
        # Reset and try again with different ordering
        pass

    # Second pass: try adding edges in reverse order
    edges2 = set()
    for u, v in reversed(all_edges):
        if time.time() > deadline:
            break
        test_edges = edges2 | {(u, v)}
        if is_valid(test_edges):
            edges2.add((u, v))

    if len(edges2) > len(edges):
        edges = edges2

    # Third pass: try adding edges sorted by vertex index difference
    edges3 = set()
    sorted_edges = sorted(all_edges, key=lambda e: abs(e[0] - e[1]))
    for u, v in sorted_edges:
        if time.time() > deadline:
            break
        test_edges = edges3 | {(u, v)}
        if is_valid(test_edges):
            edges3.add((u, v))

    if len(edges3) > len(edges):
        edges = edges3

    # Try local search: remove an edge, try to add multiple others
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try removing each edge and see if we can add more
        for u, v in list(edges):
            if time.time() > deadline - 5:
                break
            test_edges = edges - {(u, v)}
            # Try to add back other edges
            for u2, v2 in all_edges:
                if (u2, v2) == (u, v):
                    continue
                if (u2, v2) in test_edges:
                    continue
                test2 = test_edges | {(u2, v2)}
                if is_valid(test2):
                    test_edges.add((u2, v2))
            if len(test_edges) > len(edges):
                edges = test_edges
                improved = True
                break

    # Final validation
    if not is_valid(edges):
        # Fallback: return empty graph
        return []

    final_edges = [[int(u), int(v)] for u, v in edges]
    return final_edges
