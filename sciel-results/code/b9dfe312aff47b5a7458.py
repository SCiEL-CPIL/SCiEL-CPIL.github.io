"""Sign-edges solver for prism-graph spectral lifts: structured init + local search."""
import time
import numpy as np

_DEADLINE = 85.0


def _build_signed_adj(graph, signs):
    n = graph['n']
    edges = graph['edges']
    A = np.zeros((n, n), dtype=np.float64)
    for (u, v), s in zip(edges, signs):
        sv = float(s)
        A[u, v] = sv
        A[v, u] = sv
    return A


def _precompute_adj(graph):
    """Precompute adjacency list with edge indices for fast connectivity checks."""
    n = graph['n']
    edges = graph['edges']
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    return adj


def _is_coboundary_fast(adj, signs):
    """Check if sign assignment is a coboundary (lift is disconnected)."""
    n = len(adj)
    f = [-1] * n
    f[0] = 0
    stack = [0]
    while stack:
        u = stack.pop()
        for v, idx in adj[u]:
            s = signs[idx]
            expected_fv = f[u] + (0 if s == 1 else 1)
            if f[v] == -1:
                f[v] = expected_fv % 2
                stack.append(v)
            else:
                if f[v] != expected_fv % 2:
                    return False
    return True


def _lift_radius_fast(adj, graph, signs):
    """Return spectral radius if lift is connected and valid, else inf."""
    if _is_coboundary_fast(adj, signs):
        return float('inf')
    A = _build_signed_adj(graph, signs)
    try:
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))
    except Exception:
        return float('inf')


def _eigenvector_importance(graph, signs):
    """Return per-edge importance based on the top eigenvector."""
    A = _build_signed_adj(graph, signs)
    try:
        eigvals, eigvecs = np.linalg.eigh(A)
        idx = np.argmax(np.abs(eigvals))
        vec = np.abs(eigvecs[:, idx])
        edges = graph['edges']
        importance = np.zeros(len(edges))
        for i, (u, v) in enumerate(edges):
            importance[i] = vec[u] * vec[v]
        return importance
    except Exception:
        return np.ones(len(edges))


def _phase_based_init(graph, m):
    """Generate sign patterns using phase synchronization on the prism structure."""
    n_edges = len(graph['edges'])
    inits = []
    
    for base_phase in np.linspace(0, np.pi, 8):
        signs = np.ones(n_edges, dtype=np.int64)
        for i in range(m):
            theta_u = i * base_phase
            theta_v = (i + 1) % m * base_phase
            diff = theta_u - theta_v
            signs[i] = 1 if np.cos(diff) >= 0 else -1
            signs[m + i] = 1 if np.cos(diff) >= 0 else -1
        inits.append(signs.copy())
    
    signs = np.ones(n_edges, dtype=np.int64)
    for i in range(m):
        diff = 2 * np.pi * i / m
        signs[i] = 1 if np.cos(diff) >= 0 else -1
        signs[m + i] = 1 if np.cos(diff) >= 0 else -1
    inits.append(signs.copy())
    
    signs = np.ones(n_edges, dtype=np.int64)
    for i in range(m):
        diff_bottom = 2 * np.pi * i / m
        diff_top = 2 * np.pi * (i + 0.5) / m
        signs[i] = 1 if np.cos(diff_bottom) >= 0 else -1
        signs[m + i] = 1 if np.cos(diff_top) >= 0 else -1
    inits.append(signs.copy())
    
    return inits


def _generate_structured_inits(graph, m):
    """Generate smart initial sign patterns for prism graphs."""
    n_edges = len(graph['edges'])
    inits = []
    
    inits.append(np.ones(n_edges, dtype=np.int64))
    inits.append(-np.ones(n_edges, dtype=np.int64))
    
    signs = np.ones(n_edges, dtype=np.int64)
    for i in range(m):
        signs[i] = 1 if i % 2 == 0 else -1
    inits.append(signs.copy())
    
    signs = np.ones(n_edges, dtype=np.int64)
    for i in range(m):
        signs[m + i] = 1 if i % 2 == 0 else -1
    inits.append(signs.copy())
    
    signs = np.ones(n_edges, dtype=np.int64)
    for i in range(m):
        signs[2*m + i] = 1 if i % 2 == 0 else -1
    inits.append(signs.copy())
    
    rng = np.random.default_rng(42)
    for _ in range(15):
        signs = rng.choice([-1, 1], size=n_edges)
        inits.append(signs)
    
    signs = np.ones(n_edges, dtype=np.int64)
    signs[0] = -1
    inits.append(signs.copy())
    
    signs = np.ones(n_edges, dtype=np.int64)
    signs[m] = -1
    inits.append(signs.copy())
    
    signs = np.ones(n_edges, dtype=np.int64)
    signs[2*m] = -1
    inits.append(signs.copy())
    
    signs = np.ones(n_edges, dtype=np.int64)
    signs[0] = -1
    signs[m] = -1
    inits.append(signs.copy())
    
    inits.extend(_phase_based_init(graph, m))
    
    if m % 2 == 1:
        for pos in range(m):
            signs = np.ones(n_edges, dtype=np.int64)
            signs[pos] = -1
            inits.append(signs.copy())
        for pos in range(m):
            signs = np.ones(n_edges, dtype=np.int64)
            signs[m + pos] = -1
            inits.append(signs.copy())
    
    return inits


def _local_search(adj, graph, signs, radius, deadline):
    """Local search with single and double flips, ordered by eigenvector importance."""
    n_edges = len(graph['edges'])
    signs = signs.copy()
    best_radius = radius
    best_signs = signs.copy()
    
    improved = True
    stall_count = 0
    max_stall = 4
    
    while improved and time.time() < deadline:
        improved = False
        
        if time.time() < deadline:
            importance = _eigenvector_importance(graph, best_signs)
            order = np.argsort(-importance)
        else:
            order = np.arange(n_edges)
        
        for idx in order:
            i = int(idx)
            if time.time() >= deadline:
                break
            signs = best_signs.copy()
            signs[i] = -signs[i]
            r = _lift_radius_fast(adj, graph, signs)
            if r < best_radius - 1e-12:
                best_radius = r
                best_signs = signs.copy()
                improved = True
        
        if improved:
            stall_count = 0
            continue
        stall_count += 1
        
        if stall_count > max_stall:
            break
        
        if time.time() >= deadline:
            break
        for idx_i in order:
            if time.time() >= deadline:
                break
            i = int(idx_i)
            for idx_j in order:
                j = int(idx_j)
                if j <= i:
                    continue
                if time.time() >= deadline:
                    break
                signs = best_signs.copy()
                signs[i] = -signs[i]
                signs[j] = -signs[j]
                r = _lift_radius_fast(adj, graph, signs)
                if r < best_radius - 1e-12:
                    best_radius = r
                    best_signs = signs.copy()
                    improved = True
                    break
            if improved:
                break
        
        if not improved and stall_count >= 2 and time.time() < deadline:
            for idx_i in order[:min(6, n_edges)]:
                if time.time() >= deadline:
                    break
                i = int(idx_i)
                for idx_j in order[:min(10, n_edges)]:
                    if time.time() >= deadline:
                        break
                    j = int(idx_j)
                    if j <= i:
                        continue
                    for idx_k in order[:min(12, n_edges)]:
                        if time.time() >= deadline:
                            break
                        k = int(idx_k)
                        if k <= j:
                            continue
                        signs = best_signs.copy()
                        signs[i] = -signs[i]
                        signs[j] = -signs[j]
                        signs[k] = -signs[k]
                        r = _lift_radius_fast(adj, graph, signs)
                        if r < best_radius - 1e-12:
                            best_radius = r
                            best_signs = signs.copy()
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            stall_count = 0
    
    return best_signs, best_radius


def _simulated_annealing(adj, graph, signs, radius, deadline):
    """Simulated annealing to escape local minima."""
    n_edges = len(graph['edges'])
    current_signs = signs.copy()
    current_radius = radius
    best_signs = signs.copy()
    best_radius = radius
    
    T_start = 0.5
    T_end = 0.01
    n_iter = 5000
    
    elapsed_budget = max(0, deadline - time.time())
    if elapsed_budget < 2:
        return best_signs, best_radius
    
    rng = np.random.default_rng(123)
    T = T_start
    cooling = (T_end / T_start) ** (1.0 / n_iter)
    
    for it in range(n_iter):
        if time.time() >= deadline:
            break
        T *= cooling
        
        # Pick a random edge to flip
        i = rng.integers(0, n_edges)
        candidate = current_signs.copy()
        candidate[i] = -candidate[i]
        
        r_candidate = _lift_radius_fast(adj, graph, candidate)
        
        if r_candidate < current_radius - 1e-12:
            current_signs = candidate
            current_radius = r_candidate
            if r_candidate < best_radius - 1e-12:
                best_radius = r_candidate
                best_signs = candidate.copy()
        else:
            delta = r_candidate - current_radius
            if T > 0 and rng.random() < np.exp(-delta / T):
                current_signs = candidate
                current_radius = r_candidate
    
    return best_signs, best_radius


def _polish(adj, graph, signs, radius, deadline):
    """Final high-precision polish: try all single flips with tight tolerance."""
    n_edges = len(graph['edges'])
    best_signs = signs.copy()
    best_radius = radius
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_edges):
            if time.time() >= deadline:
                break
            candidate = best_signs.copy()
            candidate[i] = -candidate[i]
            r = _lift_radius_fast(adj, graph, candidate)
            if r < best_radius - 1e-14:
                best_radius = r
                best_signs = candidate
                improved = True
    
    return best_signs, best_radius


def sign_edges(graph, budget):
    start = time.time()
    deadline = start + _DEADLINE
    
    n = graph['n']
    m = n // 2
    n_edges = len(graph['edges'])
    
    if n_edges == 0:
        return np.array([], dtype=np.int64)
    
    adj = _precompute_adj(graph)
    
    inits = _generate_structured_inits(graph, m)
    
    # Evaluate all inits, keep top few for multi-start
    init_scores = []
    for init in inits:
        if time.time() >= deadline:
            break
        r = _lift_radius_fast(adj, graph, init)
        if r < float('inf'):
            init_scores.append((r, init.copy()))
    
    if not init_scores:
        rng = np.random.default_rng(3)
        for _ in range(min(budget, 50)):
            if time.time() >= deadline:
                break
            signs = rng.choice([-1, 1], size=n_edges)
            r = _lift_radius_fast(adj, graph, signs)
            if r < float('inf'):
                init_scores.append((r, signs.copy()))
    
    if not init_scores:
        return np.ones(n_edges, dtype=np.int64)
    
    # Sort by radius, take top starts for multi-start local search
    init_scores.sort(key=lambda x: x[0])
    
    best_signs = None
    best_radius = float('inf')
    
    # Multi-start: run local search from top 3 initializations
    n_starts = min(3, len(init_scores))
    for s in range(n_starts):
        if time.time() >= deadline:
            break
        r_init, signs_init = init_scores[s]
        ls_signs, ls_radius = _local_search(adj, graph, signs_init, r_init, deadline)
        if ls_radius < best_radius:
            best_radius = ls_radius
            best_signs = ls_signs.copy()
    
    # Simulated annealing from best found so far
    if time.time() < deadline - 5:
        sa_signs, sa_radius = _simulated_annealing(adj, graph, best_signs, best_radius, deadline)
        if sa_radius < best_radius:
            best_radius = sa_radius
            best_signs = sa_signs.copy()
    
    # Local search again after SA (in case SA found a better basin)
    if time.time() < deadline - 3:
        ls_signs, ls_radius = _local_search(adj, graph, best_signs, best_radius, deadline)
        if ls_radius < best_radius:
            best_radius = ls_radius
            best_signs = ls_signs.copy()
    
    # Final polish
    if time.time() < deadline - 2:
        best_signs, best_radius = _polish(adj, graph, best_signs, best_radius, deadline)
    
    return np.asarray(best_signs, dtype=np.int64)
