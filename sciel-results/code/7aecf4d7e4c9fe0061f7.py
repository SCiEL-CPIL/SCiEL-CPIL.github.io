"""Combined program: systematic patterns + local-search refinement."""
import time
import numpy as np

def _build_signed_adj(graph, signs):
    n = graph['n']
    edges = graph['edges']
    A = np.zeros((n, n), dtype=np.float64)
    for (u, v), s in zip(edges, signs):
        A[u, v] = float(s)
        A[v, u] = float(s)
    return A

def _spectral_radius(A):
    try:
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))
    except Exception:
        return float('inf')

def _check_valid(graph, signs):
    """Return (valid, radius) using the provided helper."""
    try:
        from autoresearch_suite.science_tcs_expansion import check_lift
        radius, details = check_lift(graph, signs)
        return True, float(radius)
    except Exception:
        return False, float('inf')

def _check_valid_fast(graph, signs, edge_idx):
    """Faster validity check: build signed adj, check connectivity, compute radius."""
    n = graph['n']
    A = _build_signed_adj(graph, signs)
    # Check connectivity of the 2-lift via BFS
    # Build lift adjacency: for each edge (u,v) with sign s,
    # in the lift: (u,0)-(v,s*0 mod 2) and (u,1)-(v,(1+s)/2 mod 2)
    # Actually: +1 means equal sides, -1 means opposite.
    # Edge (u,v) sign +1: (u,0)-(v,0), (u,1)-(v,1)
    # Edge (u,v) sign -1: (u,0)-(v,1), (u,1)-(v,0)
    from collections import deque
    lift_n = 2 * n
    adj = [[] for _ in range(lift_n)]
    for (u, v), s in zip(graph['edges'], signs):
        if s == 1:
            adj[u].append(v)
            adj[v].append(u)
            adj[u + n].append(v + n)
            adj[v + n].append(u + n)
        else:
            adj[u].append(v + n)
            adj[v + n].append(u)
            adj[u + n].append(v)
            adj[v].append(u + n)
    # BFS from node 0
    visited = [False] * lift_n
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
    if count != lift_n:
        return False, float('inf')
    # 3-regularity is guaranteed by prism structure (each vertex has degree 3 in base)
    r = _spectral_radius(A)
    return True, r

def _generate_candidate_signs(m, pattern_id):
    """Deterministic candidate sign vectors for a prism graph with parameter m."""
    n = 2 * m
    signs = []
    if pattern_id == 0:
        signs = [1] * (3 * m)
    elif pattern_id == 1:
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
    elif pattern_id == 2:
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1)
    elif pattern_id == 3:
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
        for i in range(m): signs.append(1)
    elif pattern_id == 4:
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(-1)
    elif pattern_id == 5:
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
        for i in range(m): signs.append(1)
    elif pattern_id == 6:
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
        for i in range(m): signs.append(1 if i % 2 != 0 else -1)
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
    elif pattern_id == 7:
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(-1)
    elif pattern_id == 8:
        for i in range(3 * m): signs.append(1 if i % 2 == 0 else -1)
    elif pattern_id == 9:
        for i in range(m): signs.append(1 if i < m // 2 else -1)
        for i in range(m): signs.append(1 if i < m // 2 else -1)
        for i in range(m): signs.append(1 if i < m // 2 else -1)
    elif pattern_id == 10:
        # phase-shifted matching, alternating
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1 if i % 2 == 1 else -1)
    elif pattern_id == 11:
        # one -1 on matching at even position, rest +1
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(-1 if i == 0 else 1)
    elif pattern_id == 12:
        # -1 on matching at odd positions
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(-1 if i % 2 == 1 else 1)
    elif pattern_id == 13:
        # top cycle single -1
        for i in range(m): signs.append(-1 if i == 0 else 1)
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1)
    elif pattern_id == 14:
        # bottom cycle single -1
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(-1 if i == 0 else 1)
        for i in range(m): signs.append(1)
    elif pattern_id == 15:
        # Coboundary: top cycle all -1 (flips one side), matching +1
        for i in range(m): signs.append(-1)
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1)
    elif pattern_id == 16:
        # Coboundary: bottom cycle all -1
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(-1)
        for i in range(m): signs.append(1)
    elif pattern_id == 17:
        # Coboundary: matching all -1
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(-1)
    elif pattern_id == 18:
        # Alternating on top + alternating on bottom (same phase)
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
        for i in range(m): signs.append(1)
    elif pattern_id == 19:
        # Alternating on top + alternating on bottom (opposite phase)
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
        for i in range(m): signs.append(1 if i % 2 == 1 else -1)
        for i in range(m): signs.append(1)
    elif pattern_id == 20:
        # Top all +1, bottom alternating, matching alternating
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
    elif pattern_id == 21:
        # Top alternating, bottom all +1, matching alternating
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
    elif pattern_id == 22:
        # All three cycles alternating
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
    elif pattern_id == 23:
        # Top +1, bottom +1, matching: one -1 at middle
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(-1 if i == m // 2 else 1)
    elif pattern_id == 24:
        # Top alternating, bottom alternating, matching all -1
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
        for i in range(m): signs.append(-1)
    else:
        signs = [1] * (3 * m)
    return np.array(signs, dtype=np.int8)

def _local_search(graph, signs, deadline):
    """Greedy 1-flip + 2-flip local search minimizing spectral radius subject to validity."""
    n_edges = len(graph['edges'])
    edges = graph['edges']
    A = _build_signed_adj(graph, signs)
    valid, radius = _check_valid(graph, signs)
    if not valid:
        return signs, float('inf')
    best = radius
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Single flips
        for i in range(n_edges):
            if time.time() > deadline:
                break
            s_old = signs[i]
            A[edges[i][0], edges[i][1]] *= -1.0
            A[edges[i][1], edges[i][0]] *= -1.0
            r = _spectral_radius(A)
            A[edges[i][0], edges[i][1]] *= -1.0
            A[edges[i][1], edges[i][0]] *= -1.0
            if r < best - 1e-12:
                signs[i] = -s_old
                valid, rv = _check_valid(graph, signs)
                if valid:
                    best = rv
                    improved = True
                else:
                    signs[i] = s_old
        # Pair flips (only if time remains and m is small enough)
        if time.time() < deadline and n_edges <= 27:
            for i in range(n_edges):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n_edges):
                    if time.time() > deadline:
                        break
                    si = signs[i]
                    sj = signs[j]
                    # Flip edge i
                    A[edges[i][0], edges[i][1]] *= -1.0
                    A[edges[i][1], edges[i][0]] *= -1.0
                    # Flip edge j
                    A[edges[j][0], edges[j][1]] *= -1.0
                    A[edges[j][1], edges[j][0]] *= -1.0
                    r = _spectral_radius(A)
                    # Revert both
                    A[edges[i][0], edges[i][1]] *= -1.0
                    A[edges[i][1], edges[i][0]] *= -1.0
                    A[edges[j][0], edges[j][1]] *= -1.0
                    A[edges[j][1], edges[j][0]] *= -1.0
                    if r < best - 1e-12:
                        signs[i] = -si
                        signs[j] = -sj
                        valid, rv = _check_valid(graph, signs)
                        if valid:
                            best = rv
                            improved = True
                        else:
                            signs[i] = si
                            signs[j] = sj
    return signs, best

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin for 60s limit

    m = graph['n'] // 2
    n_edges = len(graph['edges'])

    best_signs = np.ones(n_edges, dtype=np.int8)
    best_radius = float('inf')

    # Phase 1: evaluate systematic patterns
    n_patterns = 25
    for pattern_id in range(n_patterns):
        if time.time() > deadline:
            break
        signs = _generate_candidate_signs(m, pattern_id)
        valid, radius = _check_valid(graph, signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()

    # Fallback if no valid pattern (should not happen; all +1 is valid)
    if not np.isfinite(best_radius):
        best_signs = np.ones(n_edges, dtype=np.int8)
        valid, best_radius = _check_valid(graph, best_signs)

    # Phase 2: local-search refine the best pattern
    if time.time() < deadline:
        best_signs, best_radius = _local_search(graph, best_signs.copy(), deadline)

    # Phase 3: if time remains, refine a couple of other good patterns and keep the best
    if time.time() < deadline:
        candidates = []
        for pattern_id in range(n_patterns):
            if time.time() > deadline:
                break
            signs = _generate_candidate_signs(m, pattern_id)
            valid, radius = _check_valid(graph, signs)
            if valid:
                candidates.append((radius, pattern_id))
        candidates.sort()
        # Refine top 4 candidates (skip the one already refined)
        for radius, pattern_id in candidates[:4]:
            if time.time() > deadline:
                break
            if radius >= best_radius - 1e-9:
                continue
            signs = _generate_candidate_signs(m, pattern_id)
            refined, r = _local_search(graph, signs.copy(), deadline)
            if r < best_radius:
                best_radius = r
                best_signs = refined

    # Phase 4: final polish - if time remains, do another round of local search
    if time.time() < deadline - 2.0:
        best_signs, best_radius = _local_search(graph, best_signs.copy(), deadline)

    return best_signs
