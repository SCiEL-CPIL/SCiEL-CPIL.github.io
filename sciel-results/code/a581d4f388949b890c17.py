def sign_edges(graph, budget):
    import numpy as np
    import time
    from collections import deque

    deadline = time.time() + 55.0

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)

    # Classify edges
    bot_edges = []
    top_edges = []
    match_edges = []
    for i, (u, v) in enumerate(edges):
        a, b = min(u, v), max(u, v)
        if 0 <= a < m and 0 <= b < m:
            bot_edges.append(i)
        elif a >= m and b >= m:
            top_edges.append(i)
        else:
            match_edges.append(i)

    # Build adjacency
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    def is_connected_lift(sgn):
        """Check if the 2-lift is connected."""
        # In the lift, vertices are (v, 0) and (v, 1)
        # Edge (u,v) with sign +1: (u,0)-(v,0) and (u,1)-(v,1)
        # Edge (u,v) with sign -1: (u,0)-(v,1) and (u,1)-(v,0)
        # Build adjacency in the lift
        lift_n = 2 * n
        lift_adj = [[] for _ in range(lift_n)]
        for i, (u, v) in enumerate(edges):
            s = sgn[i]
            if s == 1:
                lift_adj[u * 2].append(v * 2)
                lift_adj[v * 2].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2 + 1)
            else:
                lift_adj[u * 2].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2)
                lift_adj[v * 2].append(u * 2 + 1)
        
        # BFS from (0, 0)
        visited = [False] * lift_n
        visited[0] = True
        dq = deque([0])
        count = 1
        while dq:
            node = dq.popleft()
            for nb in lift_adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    dq.append(nb)
        return count == lift_n

    def spectral_radius(sgn):
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edges):
            A[u, v] = sgn[i]
            A[v, u] = sgn[i]
        w = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(w)))

    best_signs = None
    best_radius = float('inf')

    def consider(sgn):
        nonlocal best_signs, best_radius
        if not is_connected_lift(sgn):
            return
        r = spectral_radius(sgn)
        if r < best_radius:
            best_radius = r
            best_signs = sgn.copy()

    # Phase 1: Random sampling with gauge fixing
    # Fix bottom cycle to all +1, randomize top and matching
    rng = np.random.default_rng(3)
    n_samples = 50000 if time.time() < deadline - 20 else 10000
    for _ in range(n_samples):
        if time.time() > deadline - 5:
            break
        sgn = np.ones(E, dtype=np.int8)
        for j in range(m):
            sgn[top_edges[j]] = rng.choice([-1, 1])
        for j in range(m):
            sgn[match_edges[j]] = rng.choice([-1, 1])
        consider(sgn)

    if best_signs is None:
        best_signs = np.ones(E, dtype=np.int8)
        best_radius = spectral_radius(best_signs) if is_connected_lift(best_signs) else float('inf')

    # Phase 2: Local search - single flips
    cur = best_signs.copy()
    cur_r = best_radius
    flip_order = np.arange(E, dtype=np.int64)
    np.random.default_rng(42).shuffle(flip_order)

    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for ei in flip_order:
            if time.time() > deadline - 3:
                break
            cur[ei] = -cur[ei]
            if not is_connected_lift(cur):
                cur[ei] = -cur[ei]
                continue
            r = spectral_radius(cur)
            if r < cur_r - 1e-9:
                cur_r = r
                improved = True
                if r < best_radius:
                    best_radius = r
                    best_signs = cur.copy()
            else:
                cur[ei] = -cur[ei]

    # Phase 3: Multi-flip search
    if time.time() < deadline - 3:
        rng2 = np.random.default_rng(7)
        cur2 = best_signs.copy()
        cur2_r = best_radius

        improved2 = True
        while improved2 and time.time() < deadline - 3:
            improved2 = False
            for _ in range(200):
                if time.time() > deadline - 3:
                    break
                i, j = rng2.integers(0, E, size=2)
                if i == j:
                    continue
                cur2[i] = -cur2[i]
                cur2[j] = -cur2[j]
                if not is_connected_lift(cur2):
                    cur2[i] = -cur2[i]
                    cur2[j] = -cur2[j]
                    continue
                r = spectral_radius(cur2)
                if r < cur2_r - 1e-9:
                    cur2_r = r
                    improved2 = True
                    if r < best_radius:
                        best_radius = r
                        best_signs = cur2.copy()
                else:
                    cur2[i] = -cur2[i]
                    cur2[j] = -cur2[j]

    # Phase 4: High-precision polish
    if time.time() < deadline - 1:
        cur5 = best_signs.copy()
        cur5_r = best_radius
        for _ in range(100):
            if time.time() > deadline - 1:
                break
            best_flip = None
            best_new_r = cur5_r
            for ei in range(E):
                if time.time() > deadline - 1:
                    break
                cur5[ei] = -cur5[ei]
                if not is_connected_lift(cur5):
                    cur5[ei] = -cur5[ei]
                    continue
                r = spectral_radius(cur5)
                if r < best_new_r - 1e-12:
                    best_new_r = r
                    best_flip = ei
                cur5[ei] = -cur5[ei]
            
            if best_flip is not None:
                cur5[best_flip] = -cur5[best_flip]
                cur5_r = best_new_r
                if cur5_r < best_radius:
                    best_radius = cur5_r
                    best_signs = cur5.copy()
            else:
                break

    return np.asarray(best_signs, dtype=np.int8)
