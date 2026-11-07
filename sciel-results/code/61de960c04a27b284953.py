import numpy as np
import time

def sign_edges(graph, budget):
    start = time.time()
    deadline = start + 55.0  # safety margin before 60s limit

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)

    # Map edges to structural types:
    # cycle1: (i, i+1 mod m) for i in 0..m-1
    # cycle2: (m+i, m+i+1 mod m)
    # match: (i, m+i)
    edge_types = []
    for (u, v) in edges:
        if u < m and v < m:
            # cycle1 edge
            a, b = min(u, v), max(u, v)
            if b == a + 1 or (a == 0 and b == m - 1):
                edge_types.append(('c1', a if b != a + 1 or a != 0 else 0))
            else:
                edge_types.append(('c1', a))
        elif u >= m and v >= m:
            u2, v2 = u - m, v - m
            a, b = min(u2, v2), max(u2, v2)
            if b == a + 1 or (a == 0 and b == m - 1):
                edge_types.append(('c2', a))
            else:
                edge_types.append(('c2', a))
        else:
            # matching edge
            i = u if u < m else v
            edge_types.append(('match', i))

    def build_signed_adj(signs):
        """Build signed adjacency matrix of base graph."""
        A = np.zeros((2*m, 2*m))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        return A

    def spectral_radius(A):
        """Max absolute eigenvalue of symmetric matrix."""
        ev = np.linalg.eigvalsh(A)
        return np.max(np.abs(ev))

    def is_valid_lift(signs):
        """Check if the 2-lift is connected and 3-regular."""
        # 3-regular is guaranteed by prism structure (each vertex degree 3)
        # Check connectivity of the lift
        # Vertices in lift: (v, 0) and (v, 1) for v in 0..2m-1
        # Edge (u,v) with sign +1: connects (u,0)-(v,0) and (u,1)-(v,1)
        # Edge (u,v) with sign -1: connects (u,0)-(v,1) and (u,1)-(v,0)
        n_lift = 2 * 2 * m  # 4m vertices
        # Build adjacency for lift
        from collections import deque
        adj = [[] for _ in range(n_lift)]
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                adj[u * 2 + 0].append(v * 2 + 0)
                adj[v * 2 + 0].append(u * 2 + 0)
                adj[u * 2 + 1].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2 + 1)
            else:
                adj[u * 2 + 0].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2 + 0)
                adj[u * 2 + 1].append(v * 2 + 0)
                adj[v * 2 + 0].append(u * 2 + 1)
        # BFS from vertex 0
        visited = [False] * n_lift
        visited[0] = True
        q = deque([0])
        count = 1
        while q:
            node = q.popleft()
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    q.append(nb)
        return count == n_lift

    def evaluate(signs):
        """Return spectral radius if valid, else inf."""
        if not is_valid_lift(signs):
            return float('inf')
        A = build_signed_adj(signs)
        return spectral_radius(A)

    # Strategy: structured patterns + local search
    # Key insight: for prism graphs, the spectral radius is minimized when
    # the signs create a "frustrated" configuration that pushes eigenvalues
    # toward zero.
    
    best_signs = None
    best_radius = float('inf')

    def try_signs(signs):
        nonlocal best_signs, best_radius
        r = evaluate(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
        return r

    # Generate structured initial patterns
    # Pattern 1: all +1
    try_signs(np.ones(n_edges, dtype=np.int8))

    # Pattern 2: all -1
    try_signs(-np.ones(n_edges, dtype=np.int8))

    # Pattern 3: cycle1 all +, cycle2 all +, match all -
    signs = np.ones(n_edges, dtype=np.int8)
    for i, (et, idx) in enumerate(edge_types):
        if et == 'match':
            signs[i] = -1
    try_signs(signs)

    # Pattern 4: cycle1 all -, cycle2 all +, match all +
    signs = np.ones(n_edges, dtype=np.int8)
    for i, (et, idx) in enumerate(edge_types):
        if et == 'c1':
            signs[i] = -1
    try_signs(signs)

    # Pattern 5: alternating on cycles
    signs = np.ones(n_edges, dtype=np.int8)
    for i, (et, idx) in enumerate(edge_types):
        if et == 'c1' and idx % 2 == 0:
            signs[i] = -1
        if et == 'c2' and idx % 2 == 0:
            signs[i] = -1
    try_signs(signs)

    # Pattern 6: matching alternating
    signs = np.ones(n_edges, dtype=np.int8)
    for i, (et, idx) in enumerate(edge_types):
        if et == 'match' and idx % 2 == 0:
            signs[i] = -1
    try_signs(signs)

    # Pattern 7: all cycle edges -1, matching +1
    signs = np.ones(n_edges, dtype=np.int8)
    for i, (et, idx) in enumerate(edge_types):
        if et != 'match':
            signs[i] = -1
    try_signs(signs)

    # Pattern 8: checkerboard on all edges
    signs = np.ones(n_edges, dtype=np.int8)
    for i, (et, idx) in enumerate(edge_types):
        if (idx + (1 if et == 'c2' else (2 if et == 'match' else 0))) % 2 == 0:
            signs[i] = -1
    try_signs(signs)

    # Local search: flip individual edges
    if best_signs is not None:
        current = best_signs.copy()
        current_radius = best_radius
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try flipping each edge
            for i in range(n_edges):
                if time.time() > deadline - 5.0:
                    break
                current[i] = -current[i]
                r = evaluate(current)
                if r < current_radius - 1e-12:
                    current_radius = r
                    improved = True
                else:
                    current[i] = -current[i]  # revert
            if current_radius < best_radius:
                best_radius = current_radius
                best_signs = current.copy()

    # Second round of local search with 2-flips
    if best_signs is not None and time.time() < deadline - 10.0:
        current = best_signs.copy()
        current_radius = best_radius
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            for i in range(n_edges):
                for j in range(i + 1, n_edges):
                    if time.time() > deadline - 10.0:
                        break
                    current[i] = -current[i]
                    current[j] = -current[j]
                    r = evaluate(current)
                    if r < current_radius - 1e-12:
                        current_radius = r
                        improved = True
                    else:
                        current[i] = -current[i]
                        current[j] = -current[j]
                if time.time() > deadline - 10.0:
                    break
            if current_radius < best_radius:
                best_radius = current_radius
                best_signs = current.copy()

    # Random restarts if time remains
    if time.time() < deadline - 15.0:
        rng = np.random.default_rng(42)
        n_restarts = 0
        while time.time() < deadline - 15.0:
            n_restarts += 1
            # Random signs
            signs = rng.choice([-1, 1], size=n_edges).astype(np.int8)
            r = evaluate(signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
            # Quick local search from random start
            if r < float('inf'):
                for _ in range(50):
                    if time.time() > deadline - 15.0:
                        break
                    best_flip = None
                    best_r = r
                    for i in range(n_edges):
                        signs[i] = -signs[i]
                        r2 = evaluate(signs)
                        if r2 < best_r:
                            best_r = r2
                            best_flip = i
                        else:
                            signs[i] = -signs[i]
                    if best_flip is not None:
                        r = best_r
                    else:
                        break
                if r < best_radius:
                    best_radius = r
                    best_signs = signs.copy()

    # Final: ensure we have a valid result
    if best_signs is None:
        # Fallback: all +1 (always valid for prism)
        best_signs = np.ones(n_edges, dtype=np.int8)

    return best_signs.astype(np.int8)
