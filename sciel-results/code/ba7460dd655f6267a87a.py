"""Combined program: systematic patterns + local-search + simulated annealing."""
import time
import numpy as np
from collections import deque

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

def _is_lift_connected(graph, signs):
    """Fast connectivity check for the 2-lift using BFS."""
    n = graph['n']
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
    return count == lift_n

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
    if not _is_lift_connected(graph, signs):
        return signs, float('inf')
    best = _spectral_radius(A)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Single flips
        for i in range(n_edges):
            if time.time() > deadline:
                break
            s_old = signs[i]
            u, v = edges[i]
            A[u, v] *= -1.0
            A[v, u] *= -1.0
            r = _spectral_radius(A)
            A[u, v] *= -1.0
            A[v, u] *= -1.0
            if r < best - 1e-12:
                signs[i] = -s_old
                if _is_lift_connected(graph, signs):
                    best = r
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
                    ui, vi = edges[i]
                    uj, vj = edges[j]
                    A[ui, vi] *= -1.0
                    A[vi, ui] *= -1.0
                    A[uj, vj] *= -1.0
                    A[vj, uj] *= -1.0
                    r = _spectral_radius(A)
                    A[ui, vi] *= -1.0
                    A[vi, ui] *= -1.0
                    A[uj, vj] *= -1.0
                    A[vj, uj] *= -1.0
                    if r < best - 1e-12:
                        signs[i] = -si
                        signs[j] = -sj
                        if _is_lift_connected(graph, signs):
                            best = r
                            improved = True
                        else:
                            signs[i] = si
                            signs[j] = sj
    return signs, best


def _simulated_annealing(graph, signs, deadline, rng):
    """Simulated annealing to escape local minima."""
    n_edges = len(graph['edges'])
    edges = graph['edges']
    A = _build_signed_adj(graph, signs)
    if not _is_lift_connected(graph, signs):
        return signs, float('inf')
    current_radius = _spectral_radius(A)
    best_radius = current_radius
    best_signs = signs.copy()
    
    t_start = time.time()
    t_end = deadline
    total_time = t_end - t_start
    if total_time <= 0:
        return best_signs, best_radius
    
    # Annealing schedule: exponential cooling
    T0 = 0.5
    T_min = 1e-6
    n_steps = 5000  # will be limited by time
    alpha = (T_min / T0) ** (1.0 / max(n_steps, 1))
    T = T0
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        # Pick a random edge to flip
        i = rng.integers(n_edges)
        u, v = edges[i]
        A[u, v] *= -1.0
        A[v, u] *= -1.0
        new_radius = _spectral_radius(A)
        delta = new_radius - current_radius
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            signs[i] = -signs[i]
            current_radius = new_radius
            if current_radius < best_radius - 1e-12:
                best_radius = current_radius
                best_signs = signs.copy()
                # Verify connectivity of best
                if not _is_lift_connected(graph, best_signs):
                    best_radius = current_radius  # keep tracking but mark issue
        else:
            # Revert
            signs[i] = -signs[i]
            A[u, v] *= -1.0
            A[v, u] *= -1.0
        
        T *= alpha
    
    # Final check: ensure best_signs is connected
    if _is_lift_connected(graph, best_signs):
        return best_signs, best_radius
    # Fallback: return current if connected, else original
    if _is_lift_connected(graph, signs):
        return signs, current_radius
    return best_signs, best_radius

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin for 90s limit

    m = graph['n'] // 2
    n_edges = len(graph['edges'])
    rng = np.random.default_rng(3)

    best_signs = np.ones(n_edges, dtype=np.int8)
    best_radius = float('inf')

    # Phase 1: evaluate systematic patterns
    n_patterns = 25
    for pattern_id in range(n_patterns):
        if time.time() > deadline:
            break
        signs = _generate_candidate_signs(m, pattern_id)
        if _is_lift_connected(graph, signs):
            A = _build_signed_adj(graph, signs)
            radius = _spectral_radius(A)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()

    # Fallback if no valid pattern (should not happen; all +1 is valid)
    if not np.isfinite(best_radius):
        best_signs = np.ones(n_edges, dtype=np.int8)
        A = _build_signed_adj(graph, best_signs)
        best_radius = _spectral_radius(A)

    # Phase 2: local-search refine the best pattern
    if time.time() < deadline:
        best_signs, best_radius = _local_search(graph, best_signs.copy(), deadline)

    # Phase 3: simulated annealing from the best local optimum
    if time.time() < deadline - 3.0:
        sa_signs, sa_radius = _simulated_annealing(graph, best_signs.copy(), deadline - 2.0, rng)
        if sa_radius < best_radius and _is_lift_connected(graph, sa_signs):
            best_radius = sa_radius
            best_signs = sa_signs
            # Polish with local search after SA
            if time.time() < deadline - 1.0:
                best_signs, best_radius = _local_search(graph, best_signs.copy(), deadline - 1.0)

    # Phase 4: if time remains, try SA from other good patterns
    if time.time() < deadline - 5.0:
        candidates = []
        for pattern_id in range(n_patterns):
            if time.time() > deadline - 5.0:
                break
            signs = _generate_candidate_signs(m, pattern_id)
            if _is_lift_connected(graph, signs):
                A = _build_signed_adj(graph, signs)
                radius = _spectral_radius(A)
                candidates.append((radius, pattern_id))
        candidates.sort()
        for radius, pattern_id in candidates[:3]:
            if time.time() > deadline - 3.0:
                break
            if radius >= best_radius - 1e-9:
                continue
            signs = _generate_candidate_signs(m, pattern_id)
            # Local search first, then SA
            if time.time() < deadline - 3.0:
                refined, r = _local_search(graph, signs.copy(), deadline - 3.0)
            else:
                refined, r = signs, radius
            if r < best_radius and time.time() < deadline - 2.0:
                sa2_signs, sa2_radius = _simulated_annealing(graph, refined.copy(), deadline - 1.0, rng)
                if sa2_radius < r and _is_lift_connected(graph, sa2_signs):
                    r = sa2_radius
                    refined = sa2_signs
                if r < best_radius:
                    best_radius = r
                    best_signs = refined

    # Phase 5: final local search polish
    if time.time() < deadline - 1.0:
        best_signs, best_radius = _local_search(graph, best_signs.copy(), deadline - 0.5)

    # Final validation
    if not _is_lift_connected(graph, best_signs):
        # Should not happen, but fallback to all +1
        best_signs = np.ones(n_edges, dtype=np.int8)
        best_radius = _spectral_radius(_build_signed_adj(graph, best_signs))

    return best_signs
