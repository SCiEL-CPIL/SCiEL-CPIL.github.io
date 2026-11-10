"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def _cycle_lengths(edges):
    """Return the set of lengths of all simple cycles in a simple undirected graph."""
    n = len(edges)
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    lengths = set()
    for start in range(n):
        # canonical DFS: only explore simple paths that start at the smallest vertex
        # and whose every other vertex is greater than start.
        stack = [(start, -1, [start])]
        while stack:
            node, parent, path = stack.pop()
            for nb in adj[node]:
                if nb == parent:
                    continue
                if nb == start and len(path) >= 3:
                    lengths.add(len(path))
                elif nb > start and nb not in path:
                    stack.append((nb, node, path + [nb]))
    return lengths

def construct():
    # 6-prism (hexagonal prism): two 6-cycles joined by matching spokes.
    # 3-regular, bipartite, girth 6.
    edges = []
    for i in range(6):
        edges.append([i, (i + 1) % 6])
        edges.append([6 + i, 6 + (i + 1) % 6])
        edges.append([i, 6 + i])
    # sanity: verify minimum degree 3, triangle-free, and a cycle exists
    deg = [0] * 12
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    assert all(d >= 3 for d in deg)
    lens = _cycle_lengths(edges)
    assert 4 not in lens and len(lens) > 0
    return [list(e) for e in edges]

# EVOLVE-BLOCK-END
