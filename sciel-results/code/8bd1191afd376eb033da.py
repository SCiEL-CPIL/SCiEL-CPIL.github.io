"""Guided local search penalty for TSP; direct GLS approach."""
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
        min_tour_usage = min(tour_usage)
    else:
        max_tour_usage = 1.0
        min_tour_usage = 0.0
    
    # The penalty should scale with how many times the current tour has been found
    # Use max_tour_usage as an indicator of how "stuck" we are
    if max_tour_usage < 1e-10:
        max_tour_usage = 1.0
    
    # Try multiple penalty strategies
    best_D_pen = None
    best_len = float('inf')
    best_tour = tour
    
    # Strategy parameters: (base_penalty, exponent, scope)
    # scope: 0 = penalize only current tour edges, 1 = penalize all edges by usage
    # Use fewer, more targeted strategies
    strategies = [
        (0.1, 1.0, 0),
        (0.2, 1.0, 0),
        (0.3, 1.0, 0),
        (0.1, 1.5, 0),
        (0.2, 1.5, 0),
        (0.1, 2.0, 0),
        (0.2, 2.0, 0),
        # Penalize all edges based on usage
        (0.05, 1.0, 1),
        (0.1, 1.0, 1),
        (0.1, 1.5, 1),
    ]
    
    for base_pen, exponent, scope in strategies:
        # Check deadline
        if time.time() - _start_time > _DEADLINE - 10:
            break
        
        D_pen = edge_distance.copy()
        
        if scope == 0:
            # Penalize only edges in the current tour
            for (u, v) in tour_edges:
                usage = edge_n_used[u, v]
                if usage > 0:
                    # Penalty scales with usage
                    penalty = base_pen * (usage / max_tour_usage) ** exponent
                    D_pen[u, v] += penalty
                    D_pen[v, u] += penalty
        else:
            # Penalize all edges based on their usage
            max_used = edge_n_used.max()
            if max_used < 1e-10:
                max_used = 1.0
            usage_matrix = edge_n_used / max_used
            penalty_matrix = base_pen * (usage_matrix ** exponent)
            D_pen = D_pen + penalty_matrix
        
        # Ensure symmetry
        D_pen = (D_pen + D_pen.T) / 2.0
        np.fill_diagonal(D_pen, 0.0)
        
        # Run 2-opt on the penalized matrix
        tour_pen = _two_opt(D_pen, tour, max_passes=10)
        
        # Polish with tighter tolerance on true distances
        tour_polished = _two_opt(edge_distance, tour_pen, max_passes=5, tol=-1e-12)
        
        # Evaluate on true distances
        true_len = _closed_len(edge_distance, tour_polished)
        
        if true_len < best_len:
            best_len = true_len
            best_tour = tour_polished
            best_D_pen = D_pen
    
    if best_D_pen is not None:
        # Final polish: run 2-opt on the best penalized matrix with tighter tolerance
        final_tour = _two_opt(best_D_pen, best_tour, max_passes=3, tol=-1e-12)
        return best_D_pen
    else:
        # Fallback: return edge_distance with mild penalty
        max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
        if max_used < 1e-10:
            max_used = 1.0
        return edge_distance + 0.1 * edge_n_used / max_used
# EVOLVE-BLOCK-END
