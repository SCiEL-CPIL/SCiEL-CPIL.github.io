"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _evaluate_fast(nbrs, scores):
    """Fast evaluation using precomputed neighbor lists."""
    n = len(nbrs)
    if n == 0:
        return 0
    indices = np.argsort(-scores, kind='stable')
    blocked = np.zeros(n, dtype=bool)
    count = 0
    for v in indices:
        if not blocked[v]:
            count += 1
            blocked[v] = True
            if len(nbrs[v]) > 0:
                blocked[nbrs[v]] = True
    return count

def _k_core(n, nbrs):
    """Compute k-core decomposition."""
    if n == 0:
        return np.zeros(0, dtype=int)
    
    core = np.zeros(n, dtype=int)
    deg = np.array([len(nbrs[i]) for i in range(n)], dtype=int)
    remaining = np.ones(n, dtype=bool)
    
    for k in range(n - 1, -1, -1):
        if not np.any(remaining):
            break
        # Find vertices with degree == k among remaining
        mask = remaining & (deg == k)
        if np.any(mask):
            core[mask] = k
            remaining[mask] = False
            # Update degrees of neighbors
            for i in np.where(mask)[0]:
                for j in nbrs[i]:
                    if remaining[j]:
                        deg[j] -= 1
    return core

def _fiedler_fast(L, n, iters=100):
    """Approximate Fiedler vector via power iteration on shifted Laplacian."""
    if n == 0:
        return np.array([])
    max_deg = int(np.max(np.sum(L, axis=1))) if n > 0 else 0
    c = 2.0 * max_deg + 1.0
    M = c * np.eye(n) - L
    ones = np.ones(n) / np.sqrt(n)
    v = np.random.randn(n)
    v = v - np.dot(v, ones) * ones
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-10:
        v = np.random.randn(n)
        v = v - np.dot(v, ones) * ones
        norm_v = np.linalg.norm(v)
    v /= norm_v
    for _ in range(iters):
        w = M @ v
        w = w - np.dot(w, ones) * ones
        norm_w = np.linalg.norm(w)
        if norm_w < 1e-10:
            break
        v = w / norm_w
    return v

def _greedy_is_size(nbrs, order):
    """Compute size of greedy independent set given a fixed order (indices)."""
    n = len(nbrs)
    if n == 0:
        return 0
    blocked = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if not blocked[v]:
            count += 1
            blocked[v] = True
            if len(nbrs[v]) > 0:
                blocked[nbrs[v]] = True
    return count

def _local_search(nbrs, scores, max_iters=5000, timeout=None):
    """
    Local search on the permutation of vertices.
    We swap adjacent elements in the ordering to improve the greedy IS size.
    """
    n = len(nbrs)
    if n == 0:
        return scores
    
    # Initial order
    order = np.argsort(-scores, kind='stable')
    current_size = _greedy_is_size(nbrs, order)
    best_size = current_size
    best_order = order.copy()
    
    start_time = time.time()
    no_improve = 0
    max_no_improve = 20
    
    # Precompute adjacency sets for fast checking? 
    # Actually, we just need to check if a swap changes the IS.
    # A swap at positions i and i+1 only affects the IS if the relative order of these two
    # and their neighbors changes in a way that matters.
    # Simpler: just re-evaluate. But re-evaluation is O(n^2) worst case.
    # With n=200, O(n^2) is 40k ops. 5000 iters * 40k = 200M ops. Might be slow in Python.
    # Let's optimize _greedy_is_size or use a faster check.
    
    # Alternative: Hill climbing on scores directly is hard.
    # Let's stick to swapping in the order array.
    # To speed up, we can check if the swap is "relevant".
    # If u and v are not adjacent and don't share neighbors, swapping them might not change anything?
    # Actually, it changes the order relative to others.
    
    # Let's just run a limited number of swaps.
    # Random restarts or simulated annealing?
    # Let's try random swaps with acceptance.
    
    for _ in range(max_iters):
        if timeout and time.time() - start_time > timeout:
            break
        if no_improve > max_no_improve:
            break
            
        # Pick two random positions to swap
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        order[i], order[j] = order[j], order[i]
        new_size = _greedy_is_size(nbrs, order)
        
        if new_size > current_size:
            current_size = new_size
            no_improve = 0
            if current_size > best_size:
                best_size = current_size
                best_order = order.copy()
        else:
            # Revert
            order[i], order[j] = order[j], order[i]
            no_improve += 1
            
    # Reconstruct scores from best_order
    # We want scores such that argsort(-scores) == best_order
    # So score[best_order[k]] = n - k
    new_scores = np.zeros(n)
    for k, v in enumerate(best_order):
        new_scores[v] = n - k
        
    return new_scores

def priority(adjacency):
    """
    Compute vertex scores for the Maximum Independent Set greedy decoder.
    """
    deadline = time.time() + 145
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]

    if n == 0:
        return np.array([])

    nbrs = [np.where(adjacency[i])[0] for i in range(n)]
    degree = np.array([len(nbrs[i]) for i in range(n)], dtype=float)

    def norm(x):
        x = np.asarray(x, dtype=float)
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    D_n = norm(degree)

    mean_nd = np.zeros(n)
    for i in range(n):
        if degree[i] > 0:
            mean_nd[i] = np.mean(degree[nbrs[i]])
        else:
            mean_nd[i] = 0
    MeanND_n = norm(mean_nd)

    snd = np.zeros(n)
    for i in range(n):
        if degree[i] > 0:
            snd[i] = np.sum(degree[nbrs[i]])
    SND_n = norm(snd)

    A = adjacency.astype(float)
    A_sq = A @ A
    tri = np.diag(A_sq) / 2.0
    Tri_n = norm(tri)

    clus = np.zeros(n)
    mask = degree > 1
    clus[mask] = 2.0 * tri[mask] / (degree[mask] * (degree[mask] - 1.0))
    Clus_n = norm(clus)

    try:
        D_mat = np.diag(degree)
        L = D_mat - A
        fiedler = _fiedler_fast(L, n, iters=100)
        abs_fiedler = np.abs(fiedler)
        AbsFiedler_n = norm(abs_fiedler)
    except Exception:
        AbsFiedler_n = np.zeros(n)

    core = _k_core(n, nbrs)
    Core_n = norm(core.astype(float))

    # Features: D, MeanND, SND, Tri, Clus, Core, Fiedler
    # Indices:    0      1        2     3     4      5      6
    
    candidates = [
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        (-0.8, -0.3, 0.0, 0.0, 0.0, -0.3, 1.0),
        (-1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5),
        (-1.0, -0.3, -0.2, 0.0, 0.0, -0.3, 0.5),
        (-0.9, -0.4, -0.3, -0.2, -0.2, -0.4, 0.8),
        (-0.8, -0.4, -0.4, -0.3, -0.3, -0.4, 0.7),
        (-1.0, -0.6, 0.0, 0.0, 0.0, -0.6, 0.6),
        (-0.9, 0.0, 0.0, 0.0, 0.0, -0.5, 0.5),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
        (-1.0, -0.5, 0.0, 0.0, 0.0, -0.5, 0.5),
        (-1.0, -0.4, -0.2, 0.0, 0.0, -0.2, 0.3),
        (-0.7, -0.3, -0.2, 0.0, -0.2, -0.3, 0.4),
        (-1.0, 0.0, 0.0, 0.0, 0.0, -0.3, 0.3),
        (-1.0, -0.5, -0.3, -0.2, -0.1, -0.4, 0.5),
        (-0.8, -0.2, 0.0, -0.2, 0.0, -0.2, 0.4),
        (-1.0, -0.6, -0.4, -0.3, -0.3, -0.5, 0.6),
        # More diverse
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
        (-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]

    best_count = 0
    best_scores = -D_n.copy()
    best_weights = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for weights in candidates:
        if time.time() > deadline:
            break
        w = np.array(weights)
        scores = (w[0] * D_n + w[1] * MeanND_n + w[2] * SND_n +
                  w[3] * Tri_n + w[4] * Clus_n + w[5] * Core_n +
                  w[6] * AbsFiedler_n)
        count = _evaluate_fast(nbrs, scores)
        if count > best_count:
            best_count = count
            best_scores = scores.copy()
            best_weights = w.copy()

    # Hill Climbing with adaptive step size
    if time.time() < deadline - 20:
        base_weights = best_weights.copy()
        step_size = 0.1
        steps = 0
        no_improve_count = 0
        while time.time() < deadline - 20 and steps < 1000:
            steps += 1
            if steps % 20 == 0:
                step_size = np.random.choice([0.05, 0.1, 0.2, 0.3])
            perturbation = np.random.normal(0, step_size, 7)
            new_weights = base_weights + perturbation
            scores = (new_weights[0] * D_n + new_weights[1] * MeanND_n +
                      new_weights[2] * SND_n + new_weights[3] * Tri_n +
                      new_weights[4] * Clus_n + new_weights[5] * Core_n +
                      new_weights[6] * AbsFiedler_n)
            count = _evaluate_fast(nbrs, scores)
            if count > best_count:
                best_count = count
                best_scores = scores.copy()
                best_weights = new_weights.copy()
                base_weights = new_weights.copy()
                no_improve_count = 0
                step_size = max(0.03, step_size * 0.8)
            else:
                no_improve_count += 1
                if no_improve_count > 15 or np.random.random() < 0.03:
                    base_weights = best_weights.copy()
                    no_improve_count = 0

    # Final Polish: Local Search on Permutation
    # This is the key improvement. Instead of tweaking weights, we directly optimize the order.
    if time.time() < deadline - 2:
        # Use the best scores found so far as the initial order
        # Run local search with a tight time budget
        ls_timeout = max(1.0, deadline - time.time() - 1.0)
        best_scores = _local_search(nbrs, best_scores, max_iters=10000, timeout=ls_timeout)
        # Update best_count just in case, though we return scores
        final_count = _evaluate_fast(nbrs, best_scores)
        # best_count = max(best_count, final_count) # Not needed for return

    return best_scores
# EVOLVE-BLOCK-END
