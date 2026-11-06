"""Rewritten program for Maximum Independent Set greedy scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_size(adjacency, order):
    """Compute independent set size given a vertex order (indices sorted by priority)."""
    n = len(order)
    selected = np.zeros(n, dtype=bool)
    available = np.ones(n, dtype=bool)
    count = 0
    for v in order:
        if available[v]:
            selected[v] = True
            count += 1
            # Remove v and its neighbors from available
            neighbors = adjacency[v].astype(bool)
            available[v] = False
            available[neighbors] = False
    return count


def _score_order(adjacency, priority_vec):
    """Evaluate a priority vector by running the greedy decoder."""
    n = len(priority_vec)
    order = np.argsort(-priority_vec, kind='stable')
    return _greedy_size(adjacency, order)


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    n = adjacency.shape[0]
    A = adjacency.astype(bool)
    
    # Precompute degrees
    deg = A.sum(axis=1).astype(float)
    
    # Base score: negative degree (lower degree = higher priority)
    base_score = -deg.copy()
    
    # Compute neighborhood edge counts for each vertex
    # E_N(v) = number of edges among neighbors of v
    # This is related to the "clustering" of the neighborhood
    # Vertices whose neighbors are well-connected (high E_N) are "more redundant"
    # so we deprioritize them slightly; vertices with sparse neighborhoods are more valuable
    
    C = np.zeros(n)  # neighborhood edge count
    for v in range(n):
        if deg[v] < 2:
            C[v] = 0.0
        else:
            # Get neighbors
            nbrs = np.where(A[v])[0]
            if len(nbrs) > 0:
                # Count edges among neighbors
                sub = A[np.ix_(nbrs, nbrs)]
                C[v] = sub.sum() / 2.0
    
    # Normalize C to [0, 1] range if possible
    max_C = C.max()
    if max_C > 0:
        C_norm = C / max_C
    else:
        C_norm = C
    
    # Strategy 1: Base + neighborhood sparsity bonus
    # Prefer low degree AND sparse neighborhood
    score1 = base_score - 0.5 * C_norm
    
    # Strategy 2: Pure degree
    score2 = base_score.copy()
    
    # Strategy 3: Degree adjusted by neighborhood density
    # density_v = 2*C[v] / (deg[v]*(deg[v]-1)) if deg[v] >= 2
    density = np.zeros(n)
    mask = deg >= 2
    density[mask] = 2.0 * C[mask] / (deg[mask] * (deg[mask] - 1.0))
    score3 = base_score - density
    
    # Strategy 4: Inverse of (degree + neighborhood_edges)
    score4 = -1.0 / (deg + C + 1e-9)
    
    # Evaluate each strategy
    candidates = [score1, score2, score3, score4]
    
    best_score = -1
    best_vec = base_score.copy()
    
    for s in candidates:
        if time.time() > deadline:
            break
        size = _score_order(adjacency, s)
        if size > best_score:
            best_score = size
            best_vec = s.copy()
    
    # Try some random perturbations of the best strategy
    rng = np.random.default_rng(42)
    
    # Try blending with random noise
    for _ in range(50):
        if time.time() > deadline:
            break
        noise = rng.normal(0, 0.1, size=n)
        blended = best_vec + noise
        size = _score_order(adjacency, blended)
        if size > best_score:
            best_score = size
            best_vec = blended.copy()
    
    # Try local search: swap pairs of vertices in the ordering
    # This is expensive, so do a limited number of iterations
    if time.time() < deadline - 10.0:
        # Get current order
        order = np.argsort(-best_vec, kind='stable')
        current_size = best_score
        
        # Try random swaps
        for _ in range(200):
            if time.time() > deadline:
                break
            
            # Pick two random positions in the order
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            
            # Swap in the order
            order[i], order[j] = order[j], order[i]
            
            # Evaluate
            new_size = _greedy_size(adjacency, order)
            
            if new_size > current_size:
                current_size = new_size
                # Update best_vec to reflect this order
                # Reconstruct priority from order: earlier = higher priority
                new_priority = np.zeros(n)
                for rank, v in enumerate(order):
                    new_priority[v] = n - rank
                best_vec = new_priority
            else:
                # Revert
                order[i], order[j] = order[j], order[i]
    
    return best_vec.astype(float)
# EVOLVE-BLOCK-END
