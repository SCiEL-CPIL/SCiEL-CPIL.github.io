"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

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
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), # Pure SimOrder
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
        # Combinations with SimOrder
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5),
        (-1.0, -0.5, 0.0, 0.0, 0.0, -0.5, 0.5, 0.5),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        (-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5),
        (-1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 1.0),
        (-0.8, -0.2, 0.0, 0.0, 0.0, -0.2, 0.3, 0.7),
        (-0.9, -0.3, 0.0, 0.0, 0.0, -0.3, 0.4, 0.6),
        # Additional diverse candidates
        (-1.0, -0.4, -0.2, 0.0, 0.0, -0.2, 0.3, 0.7),
        (-0.7, -0.3, -0.2, 0.0, -0.2, -0.3, 0.4, 0.5),
        (-1.0, 0.0, 0.0, 0.0, 0.0, -0.3, 0.3, 0.7),
        (-1.0, -0.5, -0.3, -0.2, -0.1, -0.4, 0.5, 0.3),
        (-0.8, -0.2, 0.0, -0.2, 0.0, -0.2, 0.4, 0.6),
        (-1.0, -0.6, -0.4, -0.3, -0.3, -0.5, 0.6, 0.2),
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
        count = _evaluate_fast(nbrs, scores)
        if count > best_count:
            best_count = count
            best_scores = scores.copy()
            best_weights = w.copy()

    # Hill Climbing with adaptive step size
    if time.time() < deadline - 10:
        base_weights = best_weights.copy()
        step_size = 0.1
        steps = 0
        no_improve_count = 0
        while time.time() < deadline - 10 and steps < 800:
            steps += 1
            # Occasionally try larger jumps
            if steps % 20 == 0:
                step_size = np.random.choice([0.05, 0.1, 0.2, 0.3])
            perturbation = np.random.normal(0, step_size, 8)
            new_weights = base_weights + perturbation
            scores = (new_weights[0] * D_n + new_weights[1] * MeanND_n +
                      new_weights[2] * SND_n + new_weights[3] * Tri_n +
                      new_weights[4] * Clus_n + new_weights[5] * Core_n +
                      new_weights[6] * AbsFiedler_n + new_weights[7] * SimOrder_n)
            count = _evaluate_fast(nbrs, scores)
            if count > best_count:
                best_count = count
                best_scores = scores.copy()
                best_weights = new_weights.copy()
                base_weights = new_weights.copy()
                no_improve_count = 0
                # Shrink step slightly on success
                step_size = max(0.03, step_size * 0.8)
            else:
                no_improve_count += 1
                # Occasionally reset to best
                if no_improve_count > 15 or np.random.random() < 0.03:
                    base_weights = best_weights.copy()
                    no_improve_count = 0

    # Final Polish: Efficient local improvement via targeted pair swaps
    if time.time() < deadline - 3:
        scores = best_scores.copy()
        order = np.argsort(-scores, kind='stable')
        # Only try swapping vertices that are close in rank and not adjacent
        # to each other (swapping adjacent vertices rarely helps)
        adj_set = set()
        for i in range(n):
            for j in nbrs[i]:
                adj_set.add((min(i,j), max(i,j)))
        
        improved_any = True
        max_sweeps = 50
        sweep = 0
        while improved_any and sweep < max_sweeps and time.time() < deadline - 3:
            improved_any = False
            sweep += 1
            # Sample a subset of positions to try swapping
            n_try = min(n, 40)
            try_indices = np.random.choice(n, size=n_try, replace=False)
            for i_idx in try_indices:
                if time.time() > deadline - 3:
                    break
                i = int(i_idx)
                # Try swapping with a few other random positions
                for _ in range(3):
                    j = int(np.random.randint(n))
                    if i == j:
                        continue
                    # Skip if i and j are adjacent (swapping neighbors is rarely beneficial)
                    if (min(i,j), max(i,j)) in adj_set:
                        continue
                    scores[i], scores[j] = scores[j], scores[i]
                    count = _evaluate_fast(nbrs, scores)
                    if count > best_count:
                        best_count = count
                        best_scores = scores.copy()
                        improved_any = True
                        order = np.argsort(-scores, kind='stable')
                    else:
                        scores[i], scores[j] = scores[j], scores[i]
        
        scores = best_scores

    return best_scores
# EVOLVE-BLOCK-END
