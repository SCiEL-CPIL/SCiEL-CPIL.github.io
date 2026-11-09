"""Guided local search penalty for TSP using model-and-optimize approach."""
import numpy as np
import time

_DEADLINE = 330.0
_start_time = None


def _two_opt(D, tour, max_passes=10, tol=-1e-10):
    """2-opt on distance matrix D starting from tour. Returns improved tour."""
    n = len(tour)
    if n < 4:
        return tour
    tour = list(tour)
    D_local = D
    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 1):
            a, b = tour[i - 1], tour[i]
            for j in range(i + 1, n):
                c, d = tour[j], tour[(j + 1) % n]
                delta = D_local[a, c] + D_local[b, d] - D_local[a, b] - D_local[c, d]
                if delta < tol:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return tour


def _two_opt_precise(D, tour, max_passes=20, tol=-1e-12):
    """High-precision 2-opt polish with tighter tolerance and more passes."""
    n = len(tour)
    if n < 4:
        return tour
    tour = list(tour)
    D_local = D
    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 1):
            a, b = tour[i - 1], tour[i]
            for j in range(i + 1, n):
                c, d = tour[j], tour[(j + 1) % n]
                delta = D_local[a, c] + D_local[b, d] - D_local[a, b] - D_local[c, d]
                if delta < tol:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return tour


def _closed_len(D, tour):
    """Compute the length of the closed tour using distance matrix D."""
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += D[tour[i], tour[(i + 1) % n]]
    return total


def _find_best_2opt_replacement(D, tour, edge_idx, n):
    """Find the best 2-opt swap that removes edge (tour[edge_idx], tour[(edge_idx+1)%n]).
    
    Returns (delta, i, j) where delta is the improvement and (i,j) defines the swap.
    Returns (inf, -1, -1) if no improving swap exists.
    """
    a, b = tour[edge_idx], tour[(edge_idx + 1) % n]
    best_delta = float('inf')
    best_i, best_j = -1, -1
    
    # 2-opt swap: remove edges (a,b) and (c,d), add (a,c) and (b,d)
    # where (c,d) is another edge in the tour
    for i in range(n):
        if i == edge_idx:
            continue
        c, d = tour[i], tour[(i + 1) % n]
        delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
        if delta < best_delta:
            best_delta = delta
            best_i, best_j = i, edge_idx
    
    return best_delta, best_i, best_j


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = edge_distance.shape[0]
    tour = list(local_opt_tour)
    
    # Check if we're close to the deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5:
        # Return a simple penalty to be safe
        max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
        if max_used < 1e-10:
            max_used = 1.0
        return edge_distance + 0.1 * edge_n_used / max_used
    
    # Get the edges in the current local optimum tour
    tour_edges = set()
    for i in range(n):
        u, v = tour[i], tour[(i + 1) % n]
        tour_edges.add((min(u, v), max(u, v)))
    
    # Get usage counts for tour edges
    tour_usage = []
    for (u, v) in tour_edges:
        tour_usage.append(edge_n_used[u, v])
    
    if tour_usage:
        max_tour_usage = max(tour_usage)
        min_tour_usage = min(tour_usage)
    else:
        max_tour_usage = 1.0
        min_tour_usage = 0.0
    
    if max_tour_usage < 1e-10:
        max_tour_usage = 1.0
    
    best_D_pen = None
    best_true_len = float('inf')
    best_tour = tour
    candidate_results = []
    
    # Strategy 1: Edge-reversal heuristic
    # For each tour edge, find the best 2-opt replacement and penalize the original edge
    # more aggressively based on how good the replacement is
    if time.time() - _start_time < _DEADLINE - 30:
        D_pen = edge_distance.copy()
        
        # For each edge in the tour, compute the best 2-opt swap that removes it
        # and use that to set a targeted penalty
        for i in range(n):
            u, v = tour[i], tour[(i + 1) % n]
            usage = edge_n_used[u, v]
            if usage <= 0:
                continue
            
            # Find the best 2-opt swap that removes this edge
            delta, best_i, best_j = _find_best_2opt_replacement(edge_distance, tour, i, n)
            
            # The penalty should be larger if:
            # 1. The edge has been used many times (stuck)
            # 2. There's a good alternative (delta is very negative)
            if delta < 0:
                # Good alternative exists; penalize more to force exploration
                # Scale penalty by how attractive the alternative is
                penalty_scale = min(-delta, 1.0)  # cap at 1.0
                penalty = 0.5 * usage * penalty_scale
            else:
                # No good alternative; use standard penalty
                penalty = 0.1 * usage
            
            D_pen[u, v] += penalty
            D_pen[v, u] += penalty
        
        # Ensure symmetry
        D_pen = (D_pen + D_pen.T) / 2.0
        np.fill_diagonal(D_pen, 0.0)
        
        # Run 2-opt on the penalized matrix
        tour_pen = _two_opt(D_pen, tour, max_passes=10)
        
        # Final polish: run 2-opt on true distances
        if time.time() - _start_time < _DEADLINE - 10:
            tour_polished = _two_opt_precise(edge_distance, tour_pen, max_passes=15, tol=-1e-12)
            true_len = _closed_len(edge_distance, tour_polished)
            candidate_results.append((true_len, D_pen.copy(), tour_polished))
            if true_len < best_true_len:
                best_true_len = true_len
                best_tour = tour_polished
                best_D_pen = D_pen.copy()
        else:
            true_len = _closed_len(edge_distance, tour_pen)
            candidate_results.append((true_len, D_pen.copy(), tour_pen))
            if true_len < best_true_len:
                best_true_len = true_len
                best_tour = tour_pen
                best_D_pen = D_pen.copy()
    
    # Strategy 2: Linear penalty with multiple alpha values
    alpha_candidates = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    
    for alpha in alpha_candidates:
        # Check deadline
        if time.time() - _start_time > _DEADLINE - 20:
            break
        
        D_pen = edge_distance.copy()
        
        # Apply penalty to edges in the current tour
        for (u, v) in tour_edges:
            usage = edge_n_used[u, v]
            if usage > 0:
                penalty = alpha * usage
                D_pen[u, v] += penalty
                D_pen[v, u] += penalty
        
        # Ensure symmetry
        D_pen = (D_pen + D_pen.T) / 2.0
        np.fill_diagonal(D_pen, 0.0)
        
        # Run 2-opt on the penalized matrix
        tour_pen = _two_opt(D_pen, tour, max_passes=10)
        
        # Final polish: run 2-opt on true distances
        if time.time() - _start_time < _DEADLINE - 10:
            tour_polished = _two_opt_precise(edge_distance, tour_pen, max_passes=15, tol=-1e-12)
            true_len = _closed_len(edge_distance, tour_polished)
            candidate_results.append((true_len, D_pen.copy(), tour_polished))
            if true_len < best_true_len:
                best_true_len = true_len
                best_tour = tour_polished
                best_D_pen = D_pen.copy()
        else:
            true_len = _closed_len(edge_distance, tour_pen)
            candidate_results.append((true_len, D_pen.copy(), tour_pen))
            if true_len < best_true_len:
                best_true_len = true_len
                best_tour = tour_pen
                best_D_pen = D_pen.copy()
    
    # Strategy 3: Exponential penalty
    for alpha, beta in [(0.1, 1.0), (0.15, 1.5), (0.2, 2.0)]:
        # Check deadline
        if time.time() - _start_time > _DEADLINE - 20:
            break
        
        D_pen = edge_distance.copy()
        
        for (u, v) in tour_edges:
            usage = edge_n_used[u, v]
            if usage > 0:
                penalty = alpha * np.exp(beta * usage)
                D_pen[u, v] += penalty
                D_pen[v, u] += penalty
        
        D_pen = (D_pen + D_pen.T) / 2.0
        np.fill_diagonal(D_pen, 0.0)
        
        tour_pen = _two_opt(D_pen, tour, max_passes=10)
        
        if time.time() - _start_time < _DEADLINE - 10:
            tour_polished = _two_opt_precise(edge_distance, tour_pen, max_passes=15, tol=-1e-12)
            true_len = _closed_len(edge_distance, tour_polished)
            candidate_results.append((true_len, D_pen.copy(), tour_polished))
            if true_len < best_true_len:
                best_true_len = true_len
                best_tour = tour_polished
                best_D_pen = D_pen.copy()
        else:
            true_len = _closed_len(edge_distance, tour_pen)
            candidate_results.append((true_len, D_pen.copy(), tour_pen))
            if true_len < best_true_len:
                best_true_len = true_len
                best_tour = tour_pen
                best_D_pen = D_pen.copy()
    
    # Soft-min selection using log-sum-exp for smoothness
    if candidate_results:
        # Sort by true length
        candidate_results.sort(key=lambda x: x[0])
        
        # Use log-sum-exp based weighting to select a smooth combination
        # This is a soft-min: D_soft = sum_i exp(-len_i / tau) * D_i / sum_i exp(-len_i / tau)
        # with small tau to emphasize the best candidates
        if len(candidate_results) >= 2:
            # Use temperature-based soft-min
            lens = np.array([x[0] for x in candidate_results[:5]])  # top 5
            min_len = lens.min()
            tau = 0.01  # small temperature to emphasize best
            
            # Compute weights: w_i = exp(-(len_i - min_len) / tau)
            weights = np.exp(-(lens - min_len) / tau)
            weights = weights / weights.sum()
            
            # Weighted average of the distance matrices
            best_D_pen = np.zeros_like(edge_distance)
            for w, (_, D_i, _) in zip(weights, candidate_results[:5]):
                best_D_pen += w * D_i
        else:
            best_D_pen = candidate_results[0][1]
    
    if best_D_pen is not None:
        if not np.all(np.isfinite(best_D_pen)):
            best_D_pen = edge_distance.copy()
        if best_D_pen.shape != edge_distance.shape:
            best_D_pen = edge_distance.copy()
        return best_D_pen
    else:
        max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
        if max_used < 1e-10:
            max_used = 1.0
        return edge_distance + 0.1 * edge_n_used / max_used
