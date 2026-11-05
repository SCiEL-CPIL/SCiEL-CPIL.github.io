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


def _fiedler_fast(L, n, iters=200):
    """Approximate Fiedler vector via power iteration on shifted Laplacian."""
    max_deg = int(np.max(np.sum(L, axis=1)))
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


def priority(adjacency):
    """
    Compute vertex scores for the Maximum Independent Set greedy decoder.
    Uses a combination of local degree, global centrality (Fiedler vector),
    and local structural features to better approximate which vertices to pick.
    """
    deadline = time.time() + 145
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]

    if n == 0:
        return np.array([])

    # Precompute neighbor lists for fast evaluation
    nbrs = [np.where(adjacency[i])[0] for i in range(n)]
    degree = np.array([len(nbrs[i]) for i in range(n)], dtype=float)

    # --- Feature Computation ---
    def norm(x):
        x = np.asarray(x, dtype=float)
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    # Degree
    D_n = norm(degree)

    # Mean neighbor degree
    mean_nd = np.zeros(n)
    for i in range(n):
        if degree[i] > 0:
            mean_nd[i] = np.mean(degree[nbrs[i]])
        else:
            mean_nd[i] = 0
    MeanND_n = norm(mean_nd)

    # Sum of neighbor degrees
    snd = np.zeros(n)
    for i in range(n):
        if degree[i] > 0:
            snd[i] = np.sum(degree[nbrs[i]])
    SND_n = norm(snd)

    # Triangles
    A = adjacency.astype(float)
    A_sq = A @ A
    tri = np.diag(A_sq) / 2.0
    Tri_n = norm(tri)

    # Clustering coefficient
    clus = np.zeros(n)
    mask = degree > 1
    clus[mask] = 2.0 * tri[mask] / (degree[mask] * (degree[mask] - 1.0))
    Clus_n = norm(clus)

    # Fiedler vector (second-smallest eigenvector of Laplacian)
    try:
        D_mat = np.diag(degree)
        L = D_mat - A
        fiedler = _fiedler_fast(L, n, iters=300)
        abs_fiedler = np.abs(fiedler)
        AbsFiedler_n = norm(abs_fiedler)
    except Exception:
        AbsFiedler_n = np.zeros(n)

    # Core numbers (simplified k-core)
    core = np.zeros(n, dtype=int)
    deg_copy = degree.astype(int).copy()
    remaining = np.ones(n, dtype=bool)
    max_deg = int(deg_copy.max()) if n > 0 else 0
    for k in range(max_deg, -1, -1):
        for i in range(n):
            if remaining[i] and deg_copy[i] >= k:
                core[i] = k
                remaining[i] = False
                for j in nbrs[i]:
                    if remaining[j]:
                        deg_copy[j] -= 1
    Core_n = norm(core.astype(float))

    # --- Initial Score Candidates ---
    candidates = [
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 1.0),
        (-1.0, -0.5, 0.0, 0.0, 0.0, -0.5, 1.0),
        (-0.8, -0.3, 0.0, 0.0, 0.0, -0.3, 1.0),
        (-0.7, -0.2, 0.0, 0.0, 0.0, -0.2, 0.8),
        (-1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5),
        (-1.0, -0.3, -0.2, 0.0, 0.0, -0.3, 0.5),
        (-0.9, -0.4, -0.3, -0.2, -0.2, -0.4, 0.8),
        (-0.8, -0.4, -0.4, -0.3, -0.3, -0.4, 0.7),
        (-1.0, -0.6, 0.0, 0.0, 0.0, -0.6, 0.6),
        (-0.9, 0.0, 0.0, 0.0, 0.0, -0.5, 0.5),
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

    # --- Local Polish: Hill Climbing on Score Weights ---
    if time.time() < deadline - 15:
        base_weights = best_weights.copy()
        for _ in range(100):
            if time.time() > deadline:
                break
            perturbation = np.random.normal(0, 0.1, 7)
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
            else:
                base_weights = best_weights.copy()

    # --- Final Polish: Local Swap Improvement ---
    if time.time() < deadline - 5:
        # Try swapping pairs of vertices in the selection order
        # This is a local search to refine the ordering
        selected_order = np.argsort(-best_scores, kind='stable')
        
        for iteration in range(50):
            if time.time() > deadline:
                break
            
            improved = False
            # Try swapping adjacent pairs in the selection order
            for i in range(len(selected_order) - 1):
                if time.time() > deadline:
                    break
                
                # Swap vertices at positions i and i+1
                v1, v2 = selected_order[i], selected_order[i+1]
                selected_order[i], selected_order[i+1] = v2, v1
                
                # Evaluate
                blocked = np.zeros(n, dtype=bool)
                count = 0
                for v in selected_order:
                    if not blocked[v]:
                        count += 1
                        blocked[v] = True
                        if len(nbrs[v]) > 0:
                            blocked[nbrs[v]] = True
                
                if count > best_count:
                    best_count = count
                    improved = True
                    # Update best_scores to reflect new order
                    # We need to recompute scores from the new order
                    # For now, just keep the count and order
                else:
                    # Swap back
                    selected_order[i], selected_order[i+1] = v1, v2
            
            if not improved:
                break

        # Reconstruct scores from final order
        # Assign scores based on position in the final order
        final_scores = np.zeros(n)
        for rank, v in enumerate(selected_order):
            final_scores[v] = -rank
        
        best_scores = final_scores

    return best_scores
# EVOLVE-BLOCK-END
