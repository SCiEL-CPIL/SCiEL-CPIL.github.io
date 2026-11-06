"""Combined program for Maximum Independent Set greedy scoring.
Integrates spectral relaxation (MWIS-inspired) for initial scoring with
faster incremental evaluation, dynamic min-degree heuristics, and local search.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _greedy_size(adjacency_bool, order):
    """Compute independent set size given a vertex order (indices sorted by priority, ascending)."""
    n = len(order)
    available = np.ones(n, dtype=bool)
    count = 0
    for v in order:
        if available[v]:
            count += 1
            available[v] = False
            available[adjacency_bool[v]] = False
    return count


def _greedy_size_incremental(A, order, pos1, pos2):
    """Fast evaluation of a single swap at positions pos1 and pos2 in the order.
    Only re-simulates from min(pos1, pos2) onward.
    """
    n = len(order)
    lo = min(pos1, pos2)
    if lo >= n:
        return -1

    available = np.ones(n, dtype=bool)
    count = 0
    for i in range(lo):
        v = order[i]
        if available[v]:
            count += 1
            available[v] = False
            available[A[v]] = False

    v1, v2 = order[pos1], order[pos2]
    for i in range(lo, n):
        if i == pos1:
            v = v2
        elif i == pos2:
            v = v1
        else:
            v = order[i]
        if available[v]:
            count += 1
            available[v] = False
            available[A[v]] = False
    return count


def _spectral_scores_mwis(A, n, rng, num_iters=80):
    """
    Generate scores using a spectral relaxation approach inspired by MWIS.
    We want to find a vector x where vertices that are 'far apart' get high scores.
    Use power iteration on M = I - alpha * A to find the dominant eigenvector,
    which approximates a solution to the Lagrangian relaxation of MWIS.
    """
    alpha = 0.9  # Strong penalty for adjacency
    M = np.eye(n) - alpha * A.astype(float)
    
    # Start with a random vector
    x = rng.random(n)
    x = x / (np.linalg.norm(x) + 1e-10)
    
    for _ in range(num_iters):
        x_new = M @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-10:
            break
        x = x_new / norm
    
    # Ensure positive scores
    x = np.abs(x)
    
    # Combine with degree information: low degree + high spectral score
    deg = A.sum(axis=1).astype(float)
    deg_norm = deg / (deg.max() + 1e-10)
    
    x_mean = x.mean()
    x_std = x.std() + 1e-10
    x_norm = (x - x_mean) / x_std
    
    # Vertices with low degree and high spectral score are preferred
    combined = deg_norm - 0.5 * x_norm
    return combined


def _dynamic_min_degree_order(A, n):
    """Generate order using dynamic min-degree heuristic."""
    available = np.ones(n, dtype=bool)
    order = np.zeros(n, dtype=int)
    deg = A.sum(axis=1).astype(int)
    
    for pos in range(n):
        if not available.any():
            break
        cur_deg = np.where(available, deg, np.inf)
        v = np.argmin(cur_deg)
        order[pos] = v
        available[v] = False
        neighbors = np.where(A[v])[0]
        deg[neighbors] -= 1
        
    return order


def _core_order(A, deg, n):
    """Generate order based on core decomposition (peel lowest degree first)."""
    order = np.zeros(n, dtype=int)
    remaining = set(range(n))
    deg_copy = deg.copy().astype(int)
    
    for pos in range(n):
        if not remaining:
            break
        min_deg = min(deg_copy[v] for v in remaining)
        candidates = [v for v in remaining if deg_copy[v] == min_deg]
        v = random.choice(candidates)
        order[pos] = v
        remaining.remove(v)
        for u in np.where(A[v])[0]:
            if u in remaining:
                deg_copy[u] -= 1
    return order


def _generate_heuristic_orders(A, deg, n, rng):
    """Generate a diverse set of candidate orders using different heuristics."""
    orders = []
    
    # 1. Ascending degree (classic greedy for MIS)
    orders.append(np.argsort(deg, kind='stable'))
    
    # 2. Ascending degree with random tie-breaking
    noise = rng.random(n) * 0.5
    orders.append(np.argsort(deg + noise, kind='stable'))
    
    # 3. Dynamic min-degree
    orders.append(_dynamic_min_degree_order(A, n))
    
    # 4. Core number order
    orders.append(_core_order(A, deg, n))
    
    # 5. Spectral MWIS-inspired scores
    spectral_scores = _spectral_scores_mwis(A, n, rng, num_iters=50)
    orders.append(np.argsort(spectral_scores, kind='stable'))
    
    # 6. Pure random
    orders.append(rng.permutation(n))
    
    # 7. Weighted random with degree bias
    weights = np.exp(-deg / max(deg.max(), 1))
    weights /= weights.sum()
    perm = rng.choice(n, size=n, replace=False, p=weights)
    orders.append(perm)
    
    # 8. Multiple random perturbations of degree order
    base_order = np.argsort(deg, kind='stable')
    for _ in range(2):
        perturbed = base_order.copy()
        for _ in range(n // 10):
            i, j = rng.integers(0, n, size=2)
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        orders.append(perturbed)
    
    # 9. 2-hop neighborhood size
    two_hop = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) == 0:
            two_hop[v] = 0
        else:
            two_hop_nbrs = set()
            for u in nbrs:
                two_hop_nbrs.update(np.where(A[u])[0])
            two_hop_nbrs.discard(v)
            two_hop_nbrs.difference_update(nbrs)
            two_hop[v] = len(two_hop_nbrs)
    orders.append(np.argsort(two_hop, kind='stable'))
    
    return orders


def _local_search(A, start_order, n, rng, deadline):
    """Perform local search starting from start_order using incremental evaluation.
    Returns best order and size.
    """
    current_order = start_order.copy()
    current_size = _greedy_size(A, current_order)
    best_order = current_order.copy()
    best_size = current_size
    no_improve = 0
    max_no_improve = 200

    while time.time() < deadline:
        if no_improve > max_no_improve:
            break

        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue

        new_size = _greedy_size_incremental(A, current_order, i, j)

        if new_size > current_size:
            current_order[i], current_order[j] = current_order[j], current_order[i]
            current_size = new_size
            no_improve = 0
            if current_size > best_size:
                best_size = current_size
                best_order = current_order.copy()
        else:
            no_improve += 1

    return best_order, best_size


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n = adjacency.shape[0]
    A = adjacency.astype(bool)
    
    # Precompute degrees
    deg = A.sum(axis=1).astype(float)
    
    rng = np.random.default_rng(12345)
    
    # Generate diverse candidate orders
    candidates = _generate_heuristic_orders(A, deg, n, rng)
    
    best_size = -1
    best_order = None
    
    # Evaluate all initial candidates
    for order in candidates:
        if time.time() > deadline - 20:
            break
        size = _greedy_size(A, order)
        if size > best_size:
            best_size = size
            best_order = order.copy()
    
    if best_order is None:
        best_order = np.argsort(deg, kind='stable')
        best_size = _greedy_size(A, best_order)
    
    # Multi-start local search
    ls_deadline = deadline - 5.0
    
    # Do local search from best initial candidate
    best_order, best_size = _local_search(A, best_order, n, rng, ls_deadline)

    # If time remains, do additional restarts from other candidates
    if time.time() < ls_deadline - 10:
        other_starts = [candidates[0], candidates[2], candidates[4]] if len(candidates) >= 5 else [candidates[0]]
        for start in other_starts:
            if time.time() > ls_deadline - 5:
                break
            o, s = _local_search(A, start, n, rng, ls_deadline)
            if s > best_size:
                best_size = s
                best_order = o

    # Final polish: try adjacent swaps around the best solution
    if time.time() < deadline - 2:
        for _ in range(300):
            if time.time() > deadline - 2:
                break
            i = rng.integers(0, n - 1)
            new_size = _greedy_size_incremental(A, best_order, i, i + 1)
            if new_size > best_size:
                best_order[i], best_order[i + 1] = best_order[i + 1], best_order[i]
                best_size = new_size
    
    # Convert best_order to priority vector
    # Higher priority means selected earlier.
    # Vertex at index 0 in order gets highest priority.
    priority_vec = np.zeros(n)
    for idx, v in enumerate(best_order):
        priority_vec[v] = n - idx
    
    return priority_vec.astype(float)
# EVOLVE-BLOCK-END
