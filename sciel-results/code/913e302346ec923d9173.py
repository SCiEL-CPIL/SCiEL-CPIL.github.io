"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _evaluate_order(nbrs, order):
    """Evaluate greedy decoder given an explicit ordering (list of vertex indices)."""
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

def _compute_sim_order(n, nbrs):
    """
    Simulate a parallel discrete-time exclusion process.
    Vertices with the fewest remaining neighbors are selected in each round.
    Returns an array of step indices (lower is earlier).
    """
    if n == 0:
        return np.array([])
    
    remaining = np.ones(n, dtype=bool)
    rem_deg = np.zeros(n, dtype=int)
    for i in range(n):
        rem_deg[i] = len(nbrs[i])
        
    sim_order = np.full(n, n, dtype=int) # Default to last
    t = 0
    while np.any(remaining):
        # Find min degree among remaining
        degs = np.where(remaining, rem_deg, np.inf)
        min_deg = np.min(degs)
        
        # Find candidates with min degree
        candidates = np.where(remaining & (rem_deg == min_deg))[0]
        
        for v in candidates:
            sim_order[v] = t
            
        # Remove candidates and update degrees of their neighbors
        for v in candidates:
            remaining[v] = False
            for u in nbrs[v]:
                if remaining[u]:
                    rem_deg[u] -= 1
                    
        t += 1
        
    return sim_order

def _fiedler_fast(L, n, iters=200):
    """Approximate Fiedler vector via power iteration on shifted Laplacian."""
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
        fiedler = _fiedler_fast(L, n, iters=300)
        abs_fiedler = np.abs(fiedler)
        AbsFiedler_n = norm(abs_fiedler)
    except Exception:
        AbsFiedler_n = np.zeros(n)

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

    # New Feature: Simulation Order
    sim_order = _compute_sim_order(n, nbrs)
    # Lower sim_order is better (earlier). We want high score for low sim_order.
    # So we normalize such that low sim_order -> high value.
    # norm() maps min->0, max->1. We want min->1, max->0.
    SimOrder_n = 1.0 - norm(sim_order.astype(float))

    # Features: D, MeanND, SND, Tri, Clus, Core, Fiedler, SimOrder
    # Indices:    0      1        2     3     4      5      6         7
    
    candidates = [
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 1.0, 0.0),
        (-1.0, -0.5, 0.0, 0.0, 0.0, -0.5, 1.0, 0.0),
        (-0.8, -0.3, 0.0, 0.0, 0.0, -0.3, 1.0, 0.0),
        (-1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5),
        (-1.0, -0.3, -0.2, 0.0, 0.0, -0.3, 0.5, 0.0),
        (-0.9, -0.4, -0.3, -0.2, -0.2, -0.4, 0.8, 0.0),
        (-0.8, -0.4, -0.4, -0.3, -0.3, -0.4, 0.7, 0.0),
        (-1.0, -0.6, 0.0, 0.0, 0.0, -0.6, 0.6, 0.0),
        (-0.9, 0.0, 0.0, 0.0, 0.0, -0.5, 0.5, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5),
        (-1.0, -0.5, 0.0, 0.0, 0.0, -0.5, 0.5, 0.5),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        (-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5),
        (-1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 1.0),
        (-0.8, -0.2, 0.0, 0.0, 0.0, -0.2, 0.3, 0.7),
        (-0.9, -0.3, 0.0, 0.0, 0.0, -0.3, 0.4, 0.6),
    ]

    best_count = 0
    best_scores = -D_n.copy()
    best_weights = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for weights in candidates:
        if time.time() > deadline:
            break
        w = np.array(weights)
        scores = (w[0] * D_n + w[1] * MeanND_n + w[2] * SND_n +
                  w[3] * Tri_n + w[4] * Clus_n + w[5] * Core_n +
                  w[6] * AbsFiedler_n + w[7] * SimOrder_n)
        count = _evaluate_order(nbrs, np.argsort(-scores, kind='stable').tolist())
        if count > best_count:
            best_count = count
            best_scores = scores.copy()
            best_weights = w.copy()

    # Hill Climbing on weights
    if time.time() < deadline - 30:
        base_weights = best_weights.copy()
        steps = 0
        while time.time() < deadline - 30 and steps < 300:
            steps += 1
            perturbation = np.random.normal(0, 0.1, 8)
            new_weights = base_weights + perturbation
            scores = (new_weights[0] * D_n + new_weights[1] * MeanND_n +
                      new_weights[2] * SND_n + new_weights[3] * Tri_n +
                      new_weights[4] * Clus_n + new_weights[5] * Core_n +
                      new_weights[6] * AbsFiedler_n + new_weights[7] * SimOrder_n)
            count = _evaluate_order(nbrs, np.argsort(-scores, kind='stable').tolist())
            if count > best_count:
                best_count = count
                best_scores = scores.copy()
                best_weights = new_weights.copy()
                base_weights = new_weights.copy()
            elif np.random.random() < 0.05:
                base_weights = best_weights.copy()

    # Local search on the ordering directly
    if time.time() < deadline - 5:
        order_list = np.argsort(-best_scores, kind='stable').tolist()
        current_count = best_count
        
        # Swap-based local search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(n - 1):
                if time.time() > deadline - 5:
                    break
                vi = order_list[i]
                for j in range(i + 1, n):
                    if time.time() > deadline - 5:
                        break
                    vj = order_list[j]
                    order_list[i], order_list[j] = vj, vi
                    new_count = _evaluate_order(nbrs, order_list)
                    if new_count > current_count:
                        current_count = new_count
                        improved = True
                    else:
                        order_list[i], order_list[j] = vi, vj
        
        # Move-based local search
        if time.time() < deadline - 5:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(n):
                    if time.time() > deadline - 5:
                        break
                    vi = order_list[i]
                    for j in range(n):
                        if i == j or time.time() > deadline - 5:
                            continue
                        # Move vi to position j
                        order_list.pop(i)
                        order_list.insert(j, vi)
                        new_count = _evaluate_order(nbrs, order_list)
                        if new_count > current_count:
                            current_count = new_count
                            improved = True
                        else:
                            order_list.pop(j)
                            order_list.insert(i, vi)
        
        # Reversal local search (reverse a segment)
        if time.time() < deadline - 5:
            improved = True
            max_seg = min(10, n)
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(n - max_seg + 1):
                    if time.time() > deadline - 5:
                        break
                    for seg_len in range(2, max_seg + 1):
                        if i + seg_len > n or time.time() > deadline - 5:
                            continue
                        j = i + seg_len - 1
                        # Reverse segment [i, j]
                        order_list[i:j+1] = order_list[i:j+1][::-1]
                        new_count = _evaluate_order(nbrs, order_list)
                        if new_count > current_count:
                            current_count = new_count
                            improved = True
                        else:
                            order_list[i:j+1] = order_list[i:j+1][::-1]
        
        # Convert final order to scores
        best_scores = np.empty(n, dtype=float)
        for pos, v in enumerate(order_list):
            best_scores[v] = float(n - pos)
        best_count = current_count

    return best_scores
# EVOLVE-BLOCK-END
