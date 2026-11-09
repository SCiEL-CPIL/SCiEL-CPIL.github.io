# EVOLVE-BLOCK-START
"""Guided local search penalty for TSP with adaptive multi-strategy penalty selection."""
import numpy as np
import time

_DEADLINE = 300.0
_start_time = None


def _two_opt(D, tour, max_passes=10):
    """2-opt on distance matrix D starting from tour. Returns improved tour."""
    n = len(tour)
    if n < 4:
        return list(tour)
    tour = list(tour)
    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 1):
            a, b = tour[i - 1], tour[i]
            for j in range(i + 1, n):
                c, d = tour[j], tour[(j + 1) % n]
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                if delta < -1e-10:
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


def _apply_penalty_global(edge_distance, edge_n_used, base_mult, decay, med_dist):
    """Apply global penalty to all edges based on usage count."""
    D_pen = edge_distance.copy()
    max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
    if max_used < 1e-10:
        max_used = 1.0
    usage = edge_n_used / max_used
    penalty = base_mult * med_dist * (usage ** decay)
    D_pen += penalty
    D_pen = (D_pen + D_pen.T) / 2.0
    np.fill_diagonal(D_pen, 0.0)
    return D_pen


def _apply_penalty_tour(edge_distance, edge_n_used, tour, base_mult, decay, med_dist):
    """Apply penalty only to edges in the current tour based on usage count."""
    D_pen = edge_distance.copy()
    n = len(tour)
    max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
    if max_used < 1e-10:
        max_used = 1.0
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        if u == v:
            continue
        count = edge_n_used[u, v]
        if count > 0:
            usage_ratio = count / max_used
            penalty = base_mult * med_dist * (usage_ratio ** decay)
            D_pen[u, v] += penalty
            D_pen[v, u] += penalty
    D_pen = (D_pen + D_pen.T) / 2.0
    np.fill_diagonal(D_pen, 0.0)
    return D_pen


def _apply_penalty_hybrid(edge_distance, edge_n_used, tour, base_mult, decay, med_dist, tour_weight):
    """Hybrid penalty: combine global and tour-specific penalties."""
    D_global = _apply_penalty_global(edge_distance, edge_n_used, base_mult * (1.0 - tour_weight), decay, med_dist)
    D_tour = _apply_penalty_tour(edge_distance, edge_n_used, tour, base_mult * tour_weight, decay, med_dist)
    D_pen = (D_global + D_tour) / 2.0
    D_pen = (D_pen + D_pen.T) / 2.0
    np.fill_diagonal(D_pen, 0.0)
    return D_pen


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = edge_distance.shape[0]
    if n <= 1:
        return edge_distance.copy()
    
    tour = list(local_opt_tour)
    
    # Check if we're close to the deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5:
        # Return a simple penalty to be safe
        max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
        if max_used < 1e-10:
            max_used = 1.0
        return edge_distance * (1.0 + 0.1 * edge_n_used / max_used)
    
    # Compute median distance for scaling
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    med_dist = np.median(edge_distance[mask])
    if med_dist <= 0:
        med_dist = 1.0
    
    # Build set of edges in current tour for quick lookup
    tour_edges = set()
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        if u != v:
            tour_edges.add((min(u, v), max(u, v)))
    
    max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
    if max_used < 1e-10:
        max_used = 1.0
    
    # Current tour length on true distances
    current_len = _closed_len(edge_distance, tour)
    
    # Define penalty strategies
    # Each: (type, base_mult, decay, tour_weight, description)
    # type: 'global', 'tour', 'hybrid'
    strategies = [
        # Global penalties
        ('global', 0.03, 1.0, 0.0, "global mild linear"),
        ('global', 0.06, 1.0, 0.0, "global moderate linear"),
        ('global', 0.10, 1.0, 0.0, "global strong linear"),
        ('global', 0.05, 1.5, 0.0, "global mild 1.5-power"),
        ('global', 0.08, 1.5, 0.0, "global moderate 1.5-power"),
        ('global', 0.02, 2.0, 0.0, "global mild squared"),
        ('global', 0.04, 2.0, 0.0, "global moderate squared"),
        # Tour-specific penalties
        ('tour', 0.05, 1.0, 0.0, "tour mild linear"),
        ('tour', 0.10, 1.0, 0.0, "tour moderate linear"),
        ('tour', 0.15, 1.0, 0.0, "tour strong linear"),
        ('tour', 0.20, 1.0, 0.0, "tour very strong linear"),
        ('tour', 0.05, 1.5, 0.0, "tour mild 1.5-power"),
        ('tour', 0.10, 1.5, 0.0, "tour moderate 1.5-power"),
        ('tour', 0.03, 2.0, 0.0, "tour mild squared"),
        ('tour', 0.06, 2.0, 0.0, "tour moderate squared"),
        ('tour', 0.10, 2.0, 0.0, "tour strong squared"),
        # Hybrid penalties
        ('hybrid', 0.10, 1.0, 0.5, "hybrid 50/50 linear"),
        ('hybrid', 0.15, 1.0, 0.5, "hybrid strong 50/50 linear"),
        ('hybrid', 0.20, 1.0, 0.5, "hybrid very strong 50/50 linear"),
        ('hybrid', 0.10, 1.0, 0.3, "hybrid 30% global linear"),
        ('hybrid', 0.10, 1.0, 0.7, "hybrid 70% global linear"),
        ('hybrid', 0.10, 1.5, 0.5, "hybrid 50/50 1.5-power"),
        ('hybrid', 0.08, 2.0, 0.5, "hybrid 50/50 squared"),
    ]
    
    best_D_pen = None
    best_len = current_len
    best_strategy_idx = -1
    
    for idx, (stype, base_mult, decay, tour_weight, desc) in enumerate(strategies):
        # Check deadline
        if time.time() - _start_time > _DEADLINE - 15:
            break
        
        # Apply the penalty
        if stype == 'global':
            D_pen = _apply_penalty_global(edge_distance, edge_n_used, base_mult, decay, med_dist)
        elif stype == 'tour':
            D_pen = _apply_penalty_tour(edge_distance, edge_n_used, tour, base_mult, decay, med_dist)
        else:  # hybrid
            D_pen = _apply_penalty_hybrid(edge_distance, edge_n_used, tour, base_mult, decay, med_dist, tour_weight)
        
        # Run 2-opt on the penalized matrix
        tour_pen = _two_opt(D_pen, tour, max_passes=10)
        
        # Evaluate on true distances
        true_len = _closed_len(edge_distance, tour_pen)
        
        if true_len < best_len - 1e-10:
            best_len = true_len
            best_D_pen = D_pen
            best_strategy_idx = idx
    
    # If we found a better tour, return the corresponding penalized matrix
    if best_D_pen is not None:
        return best_D_pen
    else:
        # Fallback: return original distance
        return edge_distance.copy()
# EVOLVE-BLOCK-END
