"""New approach: maximal-clique search + greedy independent-set fallback."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    n = int(P['n'])
    edges = P['edges']

    # Build adjacency structures (sets for O(1) membership, lists for iteration).
    adj_sets = [set() for _ in range(n)]
    adj_list = [set() for _ in range(n)]
    for u, v in edges:
        adj_sets[u].add(v)
        adj_sets[v].add(u)
    for i in range(n):
        adj_list[i] = list(adj_sets[i])

    def bronk(R, Pset, X):
        # Pset and X are sets of candidate vertices.
        if not Pset and not X:
            return list(R)
        # Pivot: pick vertex in Pset ∪ X with max degree within Pset.
        U = Pset | X
        pivot = max(U, key=lambda v: len(adj_sets[v] & Pset)) if U else None
        candidates = Pset
        if pivot is not None:
            candidates = Pset - adj_sets[pivot]
        for v in list(candidates):
            newR = R | {v}
            newP = Pset & adj_sets[v]
            newX = X & adj_sets[v]
            clique = bronk(newR, newP, newX)
            if clique is not None and len(clique) > len(best[0]):
                best[0] = clique
            Pset.discard(v)
            X.add(v)
        return None

    best = [[0, 1]] if n >= 2 else []
    if n >= 2:
        bronk(set(), set(range(n)), set())

    branches = [int(x) for x in sorted(best[0])]
    if len(branches) >= 2:
        # A clique: every unordered pair is an edge, so path is just [u, v].
        paths = []
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                paths.append([branches[i], branches[j]])
        return {'branches': branches, 'paths': paths}

    # Fallback 1: maximal independent set (no two are adjacent => paths exist
    # because the graph is connected; endpoints are non-adjacent branch vertices).
    chosen = []
    remaining = set(range(n))
    while remaining:
        # Pick vertex with fewest neighbours in remaining (greedy).
        v = min(remaining, key=lambda x: len(adj_sets[x] & remaining))
        chosen.append(v)
        remaining.difference_update(adj_sets[v])
        remaining.discard(v)
    if len(chosen) >= 2:
        branches = [int(x) for x in chosen]
        paths = []
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                paths.append([branches[i], branches[j]])
        return {'branches': branches, 'paths': paths}

    # Fallback 2: first edge.
    u, v = edges[0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
# EVOLVE-BLOCK-END
