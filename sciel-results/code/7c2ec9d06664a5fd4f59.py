def sign_edges(graph, budget):
    import numpy as np
    import time
    from collections import deque

    start_time = time.time()
    DEADLINE = 82  # Safety margin before 90s kill

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)

    # Precompute edge structure
    edge_u = np.array([e[0] for e in edges], dtype=int)
    edge_v = np.array([e[1] for e in edges], dtype=int)

    # Precompute adjacency: for each vertex, list of (edge_idx, other_vertex)
    adj = [[] for _ in range(2 * m)]
    for idx in range(n_edges):
        u, v = edge_u[idx], edge_v[idx]
        adj[u].append((idx, v))
        adj[v].append((idx, u))

    # Build edge type mapping
    top_edges = []
    bot_edges = []
    match_edges = []
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edges.append(idx)
        elif u >= m and v >= m:
            bot_edges.append(idx)
        else:
            match_edges.append(idx)

    def build_signed_adj(signs):
        n = 2 * m
        A = np.zeros((n, n))
        for idx in range(n_edges):
            s = signs[idx]
            u, v = edge_u[idx], edge_v[idx]
            A[u, v] = s
            A[v, u] = s
        return A

    def spectral_radius(signs):
        A = build_signed_adj(signs)
        vals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(vals)))

    def is_valid_lift(signs):
        """Check if the 2-lift is connected using balance test.
        The lift is disconnected iff the signed graph is balanced,
        i.e., there exists sigma: V->{+1,-1} with s(u,v) = sigma(u)*sigma(v) for all edges.
        """
        n = 2 * m
        sigma = np.zeros(n, dtype=int)
        sigma[0] = 1
        visited = [False] * n
        visited[0] = True
        q = deque([0])

        while q:
            u = q.popleft()
            for edge_idx, v in adj[u]:
                s = signs[edge_idx]
                expected_sigma_v = s * sigma[u]
                if not visited[v]:
                    sigma[v] = expected_sigma_v
                    visited[v] = True
                    q.append(v)
                else:
                    if sigma[v] != expected_sigma_v:
                        return True  # Contradiction: not balanced, lift is connected

        return False  # Balanced: lift is disconnected

    def local_search(initial_signs, max_iter=200):
        signs = initial_signs.copy()
        current_radius = spectral_radius(signs)
        if not is_valid_lift(signs):
            return None, current_radius

        improved = True
        iteration = 0
        while improved and iteration < max_iter and (time.time() - start_time) < DEADLINE - 3:
            improved = False
            iteration += 1

            # Single flip moves
            for i in range(n_edges):
                if (time.time() - start_time) > DEADLINE - 3:
                    break
                old_sign = signs[i]
                signs[i] = -old_sign
                if is_valid_lift(signs):
                    new_radius = spectral_radius(signs)
                    if new_radius < current_radius - 1e-12:
                        current_radius = new_radius
                        improved = True
                    else:
                        signs[i] = old_sign
                else:
                    signs[i] = old_sign

            # Double flip moves
            if (time.time() - start_time) > DEADLINE - 5:
                break
            for i in range(n_edges):
                if (time.time() - start_time) > DEADLINE - 3:
                    break
                for j in range(i + 1, n_edges):
                    if (time.time() - start_time) > DEADLINE - 3:
                        break
                    si, sj = signs[i], signs[j]
                    signs[i] = -si
                    signs[j] = -sj
                    if is_valid_lift(signs):
                        new_radius = spectral_radius(signs)
                        if new_radius < current_radius - 1e-12:
                            current_radius = new_radius
                            improved = True
                        else:
                            signs[i] = si
                            signs[j] = sj
                    else:
                        signs[i] = si
                        signs[j] = sj

        return signs, current_radius

    def polish(signs):
        """Final high-precision polish pass with tighter tolerance and more iterations."""
        if signs is None:
            return None
        result, radius = local_search(signs, max_iter=500)
        return result

    rng = np.random.default_rng(42)
    best_signs = None
    best_radius = float('inf')

    # Generate diverse initial configurations
    def generate_initial_configs():
        configs = []

        # All +1 with one -1 on matching
        s = np.ones(n_edges, dtype=int)
        if len(match_edges) > 0:
            s[match_edges[0]] = -1
        configs.append(s)

        # Alternating on top cycle
        s = np.ones(n_edges, dtype=int)
        for i, eidx in enumerate(top_edges):
            if i % 2 == 1:
                s[eidx] = -1
        configs.append(s)

        # Alternating on bottom cycle
        s = np.ones(n_edges, dtype=int)
        for i, eidx in enumerate(bot_edges):
            if i % 2 == 1:
                s[eidx] = -1
        configs.append(s)

        # Alternating on matching
        s = np.ones(n_edges, dtype=int)
        for i, eidx in enumerate(match_edges):
            if i % 2 == 1:
                s[eidx] = -1
        configs.append(s)

        # Mixed patterns
        s = np.ones(n_edges, dtype=int)
        for i, eidx in enumerate(top_edges):
            if i % 3 == 2:
                s[eidx] = -1
        for i, eidx in enumerate(bot_edges):
            if i % 3 == 1:
                s[eidx] = -1
        configs.append(s)

        # Random configurations
        for _ in range(10):
            s = rng.choice([-1, 1], size=n_edges).astype(int)
            configs.append(s)

        return configs

    # Phase 1: Explore diverse configurations
    configs = generate_initial_configs()
    for i, config in enumerate(configs):
        if (time.time() - start_time) > DEADLINE - 10:
            break

        s, r = local_search(config, max_iter=100)
        if s is not None and r < best_radius:
            best_signs = s.copy()
            best_radius = r

    # Phase 2: More random restarts from best
    for restart in range(200):
        if (time.time() - start_time) > DEADLINE - 8:
            break

        if restart < 50:
            # Perturb best solution
            if best_signs is not None:
                signs = best_signs.copy()
                k = int(rng.integers(1, max(2, n_edges // 4)))
                flip_idx = rng.choice(n_edges, size=k, replace=False)
                signs[flip_idx] *= -1
            else:
                signs = rng.choice([-1, 1], size=n_edges).astype(int)
        else:
            # Fully random
            signs = rng.choice([-1, 1], size=n_edges).astype(int)

        s, r = local_search(signs, max_iter=80)
        if s is not None and r < best_radius:
            best_signs = s.copy()
            best_radius = r

    # Phase 3: Polish the best solution
    if best_signs is not None and (time.time() - start_time) < DEADLINE - 2:
        polished = polish(best_signs)
        if polished is not None:
            best_signs = polished

    if best_signs is None:
        # Fallback: find a valid configuration
        best_signs = np.ones(n_edges, dtype=int)
        if len(match_edges) > 0:
            best_signs[match_edges[0]] = -1

    return best_signs
