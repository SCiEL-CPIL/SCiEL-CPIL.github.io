def sign_edges(graph, budget):
    import numpy as np
    import time
    from collections import deque

    deadline = time.time() + 75  # 75s safety margin within 90s limit
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)

    # Build adjacency list for balance checking
    adj = [[] for _ in range(n)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Map edge to index
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        edge_index[frozenset((u, v))] = i

    # Build prism edge types
    # bottom cycle: (i, i+1 mod m)
    # top cycle: (m+i, m+i+1 mod m)
    # matching: (i, m+i)

    # Choose spanning tree: matching (all m) + bottom path (i, i+1) for i in 0..m-2
    tree_edges = set()
    for i in range(m):
        tree_edges.add(frozenset((i, m + i)))
    for i in range(m - 1):
        tree_edges.add(frozenset((i, i + 1)))
    # chord edges: bottom (m-1, 0) + top cycle all m
    chord_edges = []
    chord_edges.append(frozenset((m - 1, 0)))
    for i in range(m):
        chord_edges.append(frozenset((m + i, m + (i + 1) % m)))

    num_chords = len(chord_edges)

    def compute_radius(signs_arr):
        """Compute spectral radius from a sign array."""
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edges):
            s = signs_arr[i]
            A[u, v] = s
            A[v, u] = s
        w = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(w)))

    def is_unbalanced(signs_dict):
        """Check if the signing is unbalanced (lift is connected)."""
        pot = [0] * n
        pot[0] = 1
        dq = deque([0])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                s = signs_dict[frozenset((u, v))]
                if pot[v] == 0:
                    pot[v] = pot[u] * s
                    dq.append(v)
                else:
                    if pot[v] != pot[u] * s:
                        return True
        return False

    best_signs_arr = None
    best_radius = float('inf')

    # Phase 1: Enumerate chord patterns
    for pattern in range(2 ** num_chords):
        if time.time() > deadline - 10:
            break

        signs_dict = {}
        for e in tree_edges:
            signs_dict[e] = 1
        for j, e in enumerate(chord_edges):
            signs_dict[e] = 1 if (pattern >> j) & 1 else -1

        if not is_unbalanced(signs_dict):
            continue

        # Build sign array
        signs_arr = np.zeros(E, dtype=np.int8)
        for i, (u, v) in enumerate(edges):
            signs_arr[i] = signs_dict[frozenset((u, v))]

        r = compute_radius(signs_arr)
        if r < best_radius:
            best_radius = r
            best_signs_arr = signs_arr.copy()

    if best_signs_arr is None:
        # Fallback: all -1 (should be unbalanced for m>=4)
        best_signs_arr = np.full(E, -1, dtype=np.int8)
        best_radius = compute_radius(best_signs_arr)

    # Phase 2: Local search polish - flip edges to improve spectral radius
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(E):
                if time.time() > deadline - 3:
                    break
                # Try flipping edge i
                old_sign = best_signs_arr[i]
                best_signs_arr[i] = -old_sign

                # Check unbalanced
                # Quick check: build dict temporarily
                signs_dict = {}
                for j, (u, v) in enumerate(edges):
                    signs_dict[frozenset((u, v))] = int(best_signs_arr[j])

                if not is_unbalanced(signs_dict):
                    best_signs_arr[i] = old_sign
                    continue

                r = compute_radius(best_signs_arr)
                if r < best_radius - 1e-10:
                    best_radius = r
                    improved = True
                else:
                    best_signs_arr[i] = old_sign

    return best_signs_arr
