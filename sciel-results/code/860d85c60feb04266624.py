def sign_edges(graph, budget):
    import numpy as np
    import time
    from collections import deque

    deadline = time.time() + 50  # 50s deadline, 10s safety margin

    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2

    # Identify edge types: top cycle (0..m-1), bottom cycle (m..2m-1), matching
    # Edges are ordered: let's identify them
    edge_types = []  # 0=top cycle, 1=bottom cycle, 2=matching
    for (u, v) in edges:
        if u < m and v < m:
            edge_types.append(0)
        elif u >= m and v >= m:
            edge_types.append(1)
        else:
            edge_types.append(2)

    def build_matrix(signs):
        A = np.zeros((n, n), dtype=np.float64)
        for idx, (u, v) in enumerate(edges):
            A[u, v] = signs[idx]
            A[v, u] = signs[idx]
        return A

    def spectral_radius_power(A, tol=1e-10, max_iter=200):
        """Power iteration for spectral radius of symmetric matrix."""
        n = A.shape[0]
        x = np.ones(n) / np.sqrt(n)
        x /= np.linalg.norm(x)
        for _ in range(max_iter):
            y = A @ x
            norm = np.linalg.norm(y)
            if norm < 1e-15:
                return 0.0
            x_new = y / norm
            if np.linalg.norm(x_new - x) < tol:
                return norm
            x = x_new
        return np.linalg.norm(A @ x)

    def is_valid_lift_fast(signs):
        """
        Check if 2-lift is connected.
        For a signed graph, the 2-lift is connected iff the sign product
        around every cycle is not all +1 (i.e., there's at least one negative cycle).
        Actually, the lift is disconnected iff all cycle sign products are +1.
        
        More precisely: the 2-lift has two components iff the signing is a coboundary,
        i.e., there exists vertex labeling s_v in {+1,-1} such that sign(e) = s_u * s_v
        for every edge (u,v). This means all cycle products are +1.
        
        For the prism graph, we can check: pick a spanning tree, assign s_0=+1,
        propagate s_v = s_u * sign(u,v) along tree edges. Then check if all non-tree
        edges satisfy sign(u,v) = s_u * s_v. If yes, disconnected; else connected.
        """
        # Build adjacency with edge indices
        adj = [[] for _ in range(n)]
        for idx, (u, v) in enumerate(edges):
            adj[u].append((v, idx))
            adj[v].append((u, idx))

        # BFS to find spanning tree and propagate signs
        s = [0] * n  # 0=unvisited, 1=+1, -1=-1
        s[0] = 1
        parent_edge = [-1] * n
        queue = deque([0])
        tree_edges = set()

        while queue:
            u = queue.popleft()
            for v, idx in adj[u]:
                if s[v] == 0:
                    s[v] = s[u] * signs[idx]
                    parent_edge[v] = idx
                    tree_edges.add(idx)
                    queue.append(v)

        # Check if all non-tree edges are consistent
        for idx, (u, v) in enumerate(edges):
            if idx not in tree_edges:
                if s[u] * s[v] != signs[idx]:
                    return True  # Not a coboundary, lift is connected
        return False  # All cycles have product +1, lift is disconnected

    # Generate structured initial candidates
    def make_candidate(pattern_type, seed_val):
        signs = np.ones(num_edges, dtype=np.int8)
        rng_local = np.random.default_rng(seed_val)

        if pattern_type == 0:
            # All +1 (will be disconnected, but useful as baseline)
            pass
        elif pattern_type == 1:
            # Alternate signs on top cycle only
            for idx, et in enumerate(edge_types):
                if et == 0:
                    signs[idx] = 1 if (idx % 2 == 0) else -1
        elif pattern_type == 2:
            # Alternate signs on bottom cycle only
            for idx, et in enumerate(edge_types):
                if et == 1:
                    signs[idx] = 1 if (idx % 2 == 0) else -1
        elif pattern_type == 3:
            # Random with bias toward -1 on matching
            for idx, et in enumerate(edge_types):
                if et == 2:
                    signs[idx] = -1 if rng_local.random() < 0.7 else 1
                else:
                    signs[idx] = 1 if rng_local.random() < 0.5 else -1
        elif pattern_type == 4:
            # Checkerboard-like: alternate on both cycles
            for idx, et in enumerate(edge_types):
                if et in (0, 1):
                    signs[idx] = 1 if (idx % 2 == 0) else -1
                else:
                    signs[idx] = -1 if rng_local.random() < 0.5 else 1
        elif pattern_type == 5:
            # Pure random
            signs = rng_local.choice([-1, 1], size=num_edges).astype(np.int8)
        elif pattern_type == 6:
            # All -1 on matching, random on cycles
            for idx, et in enumerate(edge_types):
                if et == 2:
                    signs[idx] = -1
                else:
                    signs[idx] = 1 if rng_local.random() < 0.5 else -1
        elif pattern_type == 7:
            # Spiral pattern: -1 on every 3rd edge
            for idx in range(num_edges):
                signs[idx] = -1 if (idx % 3 == 0) else 1
        elif pattern_type == 8:
            # -1 on matching, alternate on cycles
            for idx, et in enumerate(edge_types):
                if et == 2:
                    signs[idx] = -1
                else:
                    signs[idx] = 1 if (idx % 2 == 0) else -1
        elif pattern_type == 9:
            # Random but ensure at least one -1
            signs = rng_local.choice([-1, 1], size=num_edges).astype(np.int8)
            if np.all(signs == 1):
                signs[rng_local.integers(0, num_edges)] = -1

        return signs

    best_signs = None
    best_radius = float('inf')

    # Evaluate a set of initial candidates
    initial_candidates = []
    for pt in range(10):
        for sv in range(3):
            initial_candidates.append(make_candidate(pt, pt * 100 + sv))

    # Also add the seed's random candidates
    rng = np.random.default_rng(3)
    for _ in range(budget):
        initial_candidates.append(rng.choice([-1, 1], size=num_edges).astype(np.int8))

    for signs in initial_candidates:
        if not is_valid_lift_fast(signs):
            continue
        A = build_matrix(signs)
        r = spectral_radius_power(A)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()

    if best_signs is None:
        # Fallback: find any valid signing
        for pt in range(10):
            signs = make_candidate(pt, 42)
            if is_valid_lift_fast(signs):
                best_signs = signs
                break
        if best_signs is None:
            # Last resort: set one edge to -1
            signs = np.ones(num_edges, dtype=np.int8)
            for i in range(num_edges):
                signs[i] = -1
                if is_valid_lift_fast(signs):
                    best_signs = signs
                    break
                signs[i] = 1
            if best_signs is None:
                best_signs = np.ones(num_edges, dtype=np.int8)
        A = build_matrix(best_signs)
        best_radius = spectral_radius_power(A)

    # Local search with multiple strategies
    A = build_matrix(best_signs)
    current_radius = best_radius

    def try_flip(idx):
        nonlocal current_radius, best_radius, best_signs, A
        old_sign = best_signs[idx]
        best_signs[idx] = -old_sign

        if not is_valid_lift_fast(best_signs):
            best_signs[idx] = old_sign
            return False, current_radius

        A_new = build_matrix(best_signs)
        r_new = spectral_radius_power(A_new)

        if r_new < current_radius - 1e-10:
            current_radius = r_new
            best_radius = r_new
            A = A_new
            return True, r_new
        else:
            best_signs[idx] = old_sign
            return False, current_radius

    # Iterated local search
    for iteration in range(1000):
        if time.time() >= deadline:
            break

        improved = False
        # Single-pass local search
        for idx in range(num_edges):
            if time.time() >= deadline:
                break
            ok, r = try_flip(idx)
            if ok:
                improved = True

        if not improved:
            # Perturbation: flip k random edges
            k = rng.integers(2, 5)
            for _ in range(k):
                idx = rng.integers(0, num_edges)
                old_sign = best_signs[idx]
                best_signs[idx] = -old_sign

            if is_valid_lift_fast(best_signs):
                A = build_matrix(best_signs)
                current_radius = spectral_radius_power(A)
                # Local search from perturbed state
                for idx in range(num_edges):
                    if time.time() >= deadline:
                        break
                    ok, r = try_flip(idx)
                    if ok:
                        improved = True

                if current_radius < best_radius:
                    best_radius = current_radius
                    best_signs = best_signs.copy()
            else:
                # Revert perturbation
                for _ in range(k):
                    idx = rng.integers(0, num_edges)
                    # We don't know which edges were flipped, so just restart
                    break
                # Reset to best known
                best_signs = best_signs.copy() if best_signs is not None else np.ones(num_edges, dtype=np.int8)
                A = build_matrix(best_signs)
                current_radius = best_radius

            if time.time() >= deadline:
                break

            # If perturbation didn't help, try a different strategy
            if not improved and iteration % 10 == 0:
                # Try a completely new random restart
                new_signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
                if is_valid_lift_fast(new_signs):
                    new_A = build_matrix(new_signs)
                    new_r = spectral_radius_power(new_A)
                    if new_r < current_radius:
                        best_signs = new_signs
                        A = new_A
                        current_radius = new_r
                        best_radius = new_r
                        improved = True

        if not improved and iteration > 50:
            # Try more aggressive perturbation
            k = rng.integers(3, 8)
            for _ in range(k):
                idx = rng.integers(0, num_edges)
                best_signs[idx] = -best_signs[idx]

            if is_valid_lift_fast(best_signs):
                A = build_matrix(best_signs)
                current_radius = spectral_radius_power(A)
                for idx in range(num_edges):
                    if time.time() >= deadline:
                        break
                    ok, r = try_flip(idx)
                    if ok:
                        improved = True
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_signs = best_signs.copy()
            else:
                # Revert and reset
                best_signs = best_signs.copy() if best_signs is not None else np.ones(num_edges, dtype=np.int8)
                A = build_matrix(best_signs)
                current_radius = best_radius

    return best_signs.astype(np.int8)
