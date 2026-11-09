"""Guided local search penalty for TSP; direct GLS with rank-based soft constraint."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 300.0
_start_time = None


def _two_opt(D, tour, max_passes=10, tol=-1e-10):
    """2-opt on distance matrix D starting from tour. Returns improved tour."""
    n = len(tour)
    if n < 4:
        return tour
    tour = list(tour)
    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 1):
            a, b = tour[i - 1], tour[i]
            for j in range(i + 1, n):
                c, d = tour[j], tour[(j + 1) % n]
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
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
    else:
        max_tour_usage = 1.0
    
    if max_tour_usage < 1e-10:
        max_tour_usage = 1.0
    
    # Try multiple penalty strategies
    best_D_pen = None
    best_len = float('inf')
    best_tour = tour
    
    # Strategy parameters: (base_penalty, exponent, scope)
    # scope: 0 = penalize only current tour edges, 1 = penalize all edges by usage, 2 = rank-based
    strategies = [
        (0.1, 1.0, 0),
        (0.2, 1.0, 0),
        (0.3, 1.0, 0),
        (0.15, 2.0, 0),
        (0.1, 1.5, 0),
        (0.05, 1.0, 1),
        (0.1, 1.0, 1),
        (0.2, 1.0, 1),
        (0.15, 1.0, 2),
        (0.25, 1.0, 2),
        (0.1, 2.0, 2),
    ]
    
    for base_pen, exponent, scope in strategies:
        # Check deadline
        if time.time() - _start_time > _DEADLINE - 10:
            break
        
        D_pen = edge_distance.copy()
        
        if scope == 0:
            # Penalize only edges in the current tour with power law
            for (u, v) in tour_edges:
                usage = edge_n_used[u, v]
                if usage > 0:
                    normalized_usage = usage / max_tour_usage
                    penalty = base_pen * (normalized_usage ** exponent)
                    D_pen[u, v] += penalty
                    D_pen[v, u] += penalty
        elif scope == 1:
            # Penalize all edges based on their usage
            max_used = edge_n_used.max()
            if max_used < 1e-10:
                max_used = 1.0
            usage_matrix = edge_n_used / max_used
            penalty_matrix = base_pen * (usage_matrix ** exponent)
            D_pen = D_pen + penalty_matrix
        else:
            # Rank-based soft constraint: sort edges by usage, apply convex penalty on rank
            # This creates a steep barrier against most frequently used edges
            all_edges_usage = []
            for i in range(n):
                for j in range(i + 1, n):
                    all_edges_usage.append((edge_n_used[i, j], i, j))
            
            # Sort by usage descending
            all_edges_usage.sort(key=lambda x: -x[0])
            
            num_edges = len(all_edges_usage)
            for rank, (usage, i, j) in enumerate(all_edges_usage):
                if usage <= 0:
                    continue
                # Normalized rank: 0 for most used, 1 for least used
                norm_rank = rank / max(num_edges - 1, 1)
                # Convex penalty: higher for high-usage (low-rank) edges
                # Use (1 - norm_rank)^exponent so most-used edges get max penalty
                penalty = base_pen * ((1.0 - norm_rank) ** exponent)
                D_pen[i, j] += penalty
                D_pen[j, i] += penalty
        
        # Ensure symmetry
        D_pen = (D_pen + D_pen.T) / 2.0
        np.fill_diagonal(D_pen, 0.0)
        
        # Run 2-opt on the penalized matrix
        tour_pen = _two_opt(D_pen, tour, max_passes=10)
        
        # Final polish: run 2-opt on true distances with tighter tolerance
        if time.time() - _start_time < _DEADLINE - 5:
            tour_polished = _two_opt(edge_distance, tour_pen, max_passes=15, tol=-1e-12)
            true_len = _closed_len(edge_distance, tour_polished)
            if true_len < best_len:
                best_len = true_len
                best_tour = tour_polished
                best_D_pen = D_pen.copy()
        else:
            true_len = _closed_len(edge_distance, tour_pen)
            if true_len < best_len:
                best_len = true_len
                best_tour = tour_pen
                best_D_pen = D_pen.copy()
    
    # Additional refinement: fine-tune the best penalty with small multiplicative adjustments
    if best_D_pen is not None and time.time() - _start_time < _DEADLINE - 15:
        for mult in [0.7, 1.3, 1.5]:
            D_alt = best_D_pen.copy()
            # Scale only the penalty portion: recompute as edge_distance + mult * (D_alt - edge_distance)
            penalty_part = D_alt - edge_distance
            D_alt = edge_distance + mult * penalty_part
            D_alt = (D_alt + D_alt.T) / 2.0
            np.fill_diagonal(D_alt, 0.0)
            
            tour_alt = _two_opt(D_alt, tour, max_passes=10)
            tour_alt = _two_opt(edge_distance, tour_alt, max_passes=15, tol=-1e-12)
            alt_len = _closed_len(edge_distance, tour_alt)
            
            if alt_len < best_len:
                best_len = alt_len
                best_tour = tour_alt
                best_D_pen = D_alt.copy()
            
            if time.time() - _start_time > _DEADLINE - 10:
                break
    
    # Final high-precision polish on the best tour found
    if best_tour is not tour and time.time() - _start_time < _DEADLINE - 3:
        # One more 2-opt pass with very tight tolerance
        final_tour = _two_opt(edge_distance, best_tour, max_passes=20, tol=-1e-14)
        final_len = _closed_len(edge_distance, final_tour)
        if final_len < best_len:
            best_tour = final_tour
            best_len = final_len
    
    if best_D_pen is not None:
        return best_D_pen
    else:
        # Fallback: return edge_distance with mild penalty
        max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
        if max_used < 1e-10:
            max_used = 1.0
        return edge_distance + 0.1 * edge_n_used / max_used
# EVOLVE-BLOCK-END
