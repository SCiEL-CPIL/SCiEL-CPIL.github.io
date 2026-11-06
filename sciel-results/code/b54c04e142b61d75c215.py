def construct():
    import time
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    P = 101
    inv = [0] + [pow(x, -1, P) for x in range(1, 11)]

    # Enumerate all non-empty subsets
    subsets = []  # list of (residue, size, terms)
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                terms.append(x)
                s = (s + inv[x]) % P
        subsets.append((s, len(terms), terms))

    n_subsets = len(subsets)

    # For each L, try to find a perfect bipartite matching
    # Left: residues 0..100, Right: subsets with size <= L
    # Edge: residue r to subset i if subsets[i][0] == r and subsets[i][1] <= L
    for L in range(1, 11):
        if time.time() - start_time > DEADLINE:
            break

        # Build adjacency: adj[r] = list of subset indices that produce residue r with size <= L
        adj = [[] for _ in range(P)]
        for i in range(n_subsets):
            res, sz, terms = subsets[i]
            if sz <= L:
                adj[res].append(i)

        # Kuhn's augmenting path algorithm
        match_right = [-1] * n_subsets  # match_right[j] = residue matched to subset j
        match_left = [-1] * P           # match_left[r] = subset matched to residue r

        import sys
        sys.setrecursionlimit(10000)

        def try_kuhn(r, visited):
            for j in adj[r]:
                if visited[j]:
                    continue
                visited[j] = True
                if match_right[j] == -1 or try_kuhn(match_right[j], visited):
                    match_right[j] = r
                    match_left[r] = j
                    return True
            return False

        matched = 0
        for r in range(P):
            if time.time() - start_time > DEADLINE:
                break
            visited = [False] * n_subsets
            if try_kuhn(r, visited):
                matched += 1

        if matched == P:
            result = [subsets[match_left[r]][2] for r in range(P)]
            return result

    # Fallback: greedy
    assignment = {}
    for L in range(1, 11):
        for res, sz, terms in subsets:
            if sz == L and res not in assignment:
                assignment[res] = terms
        if len(assignment) == P:
            return [assignment[r] for r in range(P)]

    return [assignment.get(r, [1]) for r in range(P)]
