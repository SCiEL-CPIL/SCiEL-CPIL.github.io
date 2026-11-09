"""Combined program: systematic patterns + fast local-search + simulated annealing polish."""
import time
import numpy as np
from collections import deque

def _build_signed_adj(graph, signs):
    n = graph['n']
    edges = graph['edges']
    A = np.zeros((n, n), dtype=np.float64)
    for (u, v), s in zip(edges, signs):
        sv = float(s)
        A[u, v] = sv
        A[v, u] = sv
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
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1 if i % 2 == 1 else -1)
    elif pattern_id == 11:
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(-1 if i == 0 else 1)
    elif pattern_id == 12:
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(-1 if i % 2 == 1 else 1)
    elif pattern_id == 13:
        for i in range(m): signs.append(-1 if i == 0 else 1)
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1)
    elif pattern_id == 14:
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(-1 if i == 0 else 1)
        for i in range(m): signs.append(1)
    elif pattern_id == 15:
        for i in range(m): signs.append(-1)
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1)
    elif pattern_id == 16:
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(-1)
        for i in range(m): signs.append(1)
    elif pattern_id == 17:
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(-1)
    elif pattern_id == 18:
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
        for i in range(m): signs.append(1)
    elif pattern_id == 19:
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
        for i in range(m): signs.append(1 if i % 2 == 1 else -1)
        for i in range(m): signs.append(1)
    elif pattern_id == 20:
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
    elif pattern_id == 21:
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
    elif pattern_id == 22:
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
    elif pattern_id == 23:
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(1)
        for i in range(m): signs.append(-1 if i == m // 2 else 1)
    elif pattern_id == 24:
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
        for i in range(m): signs.append(1 if i % 2 == 0 else -1)
        for i in range(m): signs.append(-1)
    else:
        signs = [1] * (3 * m)
    return np.array(signs, dtype=np.int8)

def _local_search_fast(graph, signs, deadline):
    """Fast local search: use eigvalsh for radius, BFS for validity. 1-flip + 2-flip."""
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
                if _is_lift_connected(graph, signs):
                    best = r
                    improved = True
                else:
                    signs[i] = s_old
        # Pair flips
        if time.time() < deadline and n_edges <= 27:
            for i in range(n_edges):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n_edges):
                    if time.time() > deadline:
                        break
                    A[edges[i][0], edges[i][1]] *= -1.0
                    A[edges[i][1], edges[i][0]] *= -1.0
                    A[edges[j][0], edges[j][1]] *= -1.0
                    A[edges[j][1], edges[j][0]] *= -1.0
                    r = _spectral_radius(A)
                    A[edges[i][0], edges[i][1]] *= -1.0
                    A[edges[i][1], edges[i][0]] *= -1.0
                    A[edges[j][0], edges[j][1]] *= -1.0
                    A[edges[j][1], edges[j][0]] *= -1.0
                    if r < best - 1e-12:
                        signs[i] = -signs[i]
                        signs[j] = -signs[j]
                        if _is_lift_connected(graph, signs):
                            best = r
                            improved = True
                        else:
                            signs[i] = -signs[i]
                            signs[j] = -signs[j]
    return signs, best

def _simulated_annealing(graph, signs, radius, deadline):
    """Simulated annealing to escape local minima. Returns (signs, radius)."""
    n_edges = len(graph['edges'])
    edges = graph['edges']
    A = _build_signed_adj(graph, signs)
    best_signs = signs.copy()
    best_radius = radius
    current_radius = radius
    current_signs = signs.copy()
    
    T_start = 0.05
    T_end = 0.001
    n_iterations = 5000
    # Time budget: use remaining time
    remaining = max(0, deadline - time.time())
    if remaining < 1.0:
        return best_signs, best_radius
    
    # Scale iterations to time
    n_iterations = min(n_iterations, int(remaining * 2000))
    
    for it in range(n_iterations):
        if time.time() > deadline:
            break
        T = T_start * (T_end / T_start) ** (it / max(1, n_iterations - 1))
        
        # Pick random edge to flip
        i = np.random.randint(n_edges)
        A[edges[i][0], edges[i][1]] *= -1.0
        A[edges[i][1], edges[i][0]] *= -1.0
        new_radius = _spectral_radius(A)
        A[edges[i][0], edges[i][1]] *= -1.0
        A[edges[i][1], edges[i][0]] *= -1.0
        
        if new_radius < current_radius:
            current_signs[i] = -current_signs[i]
            current_radius = new_radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_signs = current_signs.copy()
        else:
            delta = new_radius - current_radius
            if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                # Check connectivity before accepting
                current_signs[i] = -current_signs[i]
                if _is_lift_connected(graph, current_signs):
                    current_radius = new_radius
                else:
                    current_signs[i] = -current_signs[i]
    
    # Final local search polish
    if time.time() < deadline - 0.5:
        best_signs, best_radius = _local_search_fast(graph, best_signs, deadline)
    
    return best_signs, best_radius

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin for 60s limit (evaluator kills at 90s)

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

    # Fallback if no valid pattern
    if not np.isfinite(best_radius):
        best_signs = np.ones(n_edges, dtype=np.int8)
        valid, best_radius = _check_valid(graph, best_signs)

    # Phase 2: fast local-search refine the best pattern
    if time.time() < deadline:
        best_signs, best_radius = _local_search_fast(graph, best_signs.copy(), deadline)

    # Phase 3: refine a couple of other good patterns and keep the best
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
        for radius, pattern_id in candidates[:4]:
            if time.time() > deadline:
                break
            if radius >= best_radius - 1e-9:
                continue
            signs = _generate_candidate_signs(m, pattern_id)
            refined, r = _local_search_fast(graph, signs.copy(), deadline)
            if r < best_radius:
                best_radius = r
                best_signs = refined

    # Phase 4: simulated annealing to escape local minima
    if time.time() < deadline - 2.0:
        best_signs, best_radius = _simulated_annealing(graph, best_signs.copy(), best_radius, deadline)

    # Phase 5: final polish - one more round of local search
    if time.time() < deadline - 1.0:
        best_signs, best_radius = _local_search_fast(graph, best_signs.copy(), deadline)

    # Final validation
    valid, final_radius = _check_valid(graph, best_signs)
    if not valid:
        # Fallback to all +1
        best_signs = np.ones(n_edges, dtype=np.int8)

    return best_signs
