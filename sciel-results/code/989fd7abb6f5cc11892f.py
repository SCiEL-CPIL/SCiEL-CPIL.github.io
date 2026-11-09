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
    else:
        signs = [1] * (3 * m)
    return np.array(signs, dtype=np.int8)

def _local_search(graph, signs, deadline):
    """Greedy 1-flip local search minimizing spectral radius subject to validity."""
    n_edges = len(graph['edges'])
    A = _build_signed_adj(graph, signs)
    valid, radius = _check_valid(graph, signs)
    if not valid:
        return signs, float('inf')
    best = radius
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_edges):
            if time.time() > deadline:
                break
            s_old = signs[i]
            A[graph['edges'][i][0], graph['edges'][i][1]] *= -1.0
            A[graph['edges'][i][1], graph['edges'][i][0]] *= -1.0
            r = _spectral_radius(A)
            # revert
            A[graph['edges'][i][0], graph['edges'][i][1]] *= -1.0
            A[graph['edges'][i][1], graph['edges'][i][0]] *= -1.0
            if r < best - 1e-12:
                # validate the flip
                signs[i] = -s_old
                valid, rv = _check_valid(graph, signs)
                if valid:
                    best = rv
                    improved = True
                else:
                    signs[i] = s_old
    return signs, best

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin for 60s limit

    m = graph['n'] // 2
    n_edges = len(graph['edges'])

    best_signs = np.ones(n_edges, dtype=np.int8)
    best_radius = float('inf')

    # Phase 1: evaluate systematic patterns
    for pattern_id in range(15):
        if time.time() > deadline:
            break
        signs = _generate_candidate_signs(m, pattern_id)
        valid, radius = _check_valid(graph, signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()

    # Fallback if no valid pattern (should not happen; all +1 is valid)
    if best_signs is None or not np.isfinite(best_radius):
        best_signs = np.ones(n_edges, dtype=np.int8)
        valid, best_radius = _check_valid(graph, best_signs)

    # Phase 2: local-search refine the best pattern
    if time.time() < deadline:
        best_signs, best_radius = _local_search(graph, best_signs.copy(), deadline)

    # Phase 3: if time remains, refine a couple of other good patterns and keep the best
    if time.time() < deadline:
        # refine the 2nd and 3rd best patterns quickly
        candidates = []
        for pattern_id in range(15):
            if time.time() > deadline:
                break
            signs = _generate_candidate_signs(m, pattern_id)
            valid, radius = _check_valid(graph, signs)
            if valid:
                candidates.append((radius, pattern_id))
        candidates.sort()
        for radius, pattern_id in candidates[:3]:
            if time.time() > deadline:
                break
            signs = _generate_candidate_signs(m, pattern_id)
            refined, r = _local_search(graph, signs.copy(), deadline)
            if r < best_radius:
                best_radius = r
                best_signs = refined

    return best_signs
