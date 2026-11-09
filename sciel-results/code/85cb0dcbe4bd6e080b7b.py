"""Guided local search penalty rule for TSP with adaptive GLS penalty."""
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 280.0  # Safety margin against 330s kill


def _t2opt(dist, tour, max_passes=10):
    """Run 2-opt on the given tour under the given distance matrix."""
    n = len(tour)
    if n < 4:
        return list(tour)
    best_tour = list(tour)
    best_len = 0.0
    for i in range(n):
        a = best_tour[i]
        b = best_tour[(i + 1) % n]
        best_len += dist[a, b]
    
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            a = best_tour[i]
            b = best_tour[i + 1]
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue
                c = best_tour[j]
                d = best_tour[(j + 1) % n]
                delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
                if delta < -1e-10:
                    best_tour[i + 1:j + 1] = best_tour[i + 1:j + 1][::-1]
                    best_len += delta
                    improved = True
        if not improved:
            break
    return best_tour, best_len


def _build_penalized(edge_distance, edge_n_used, alpha):
    """Build penalized distance matrix using linear penalty."""
    penalty = alpha * edge_n_used
    result = edge_distance + penalty
    result = (result + result.T) / 2.0
    np.fill_diagonal(result, 0.0)
    return result


def _bootstrap(edge_distance, local_opt_tour, edge_n_used):
    """
    Try several penalty scales, run 2-opt on each, and pick the alpha
    that gives the best true-distance tour length after 2-opt.
    """
    n = len(local_opt_tour)
    
    # Compute tour length and edge usage sum
    tour_edges = [(local_opt_tour[i], local_opt_tour[(i + 1) % n]) for i in range(n)]
    tour_len = sum(edge_distance[a, b] for a, b in tour_edges)
    edge_sum = sum(edge_n_used[a, b] for a, b in tour_edges)
    
    if edge_sum <= 0:
        return edge_distance.copy()
    
    # Base scale: normalize so that total penalty is a fraction of tour length
    base = 0.15 * tour_len / edge_sum
    
    # Focused candidate alphas: geometric spread centered on base
    multipliers = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
    all_alphas = list(multipliers * base)
    
    # Add a few fixed values for robustness
    fixed_alphas = [0.01, 0.05, 0.1, 0.2, 0.3]
    all_alphas.extend(fixed_alphas)
    
    # Deduplicate, sort, filter
    all_alphas = sorted(set(round(a, 12) for a in all_alphas if 0 < a < 1.0))
    if not all_alphas:
        all_alphas = [base]
    
    best_alpha = base
    best_true_len = float('inf')
    
    true_dist = edge_distance
    
    for alpha in all_alphas:
        if time.perf_counter() > _DEADLINE - 25:
            break
        
        # Build penalized matrix
        penalized = _build_penalized(edge_distance, edge_n_used, alpha)
        
        # Run 2-opt on penalized matrix
        tour_p, _ = _t2opt(penalized, local_opt_tour, max_passes=10)
        
        # Evaluate true length efficiently
        tour_p_arr = np.array(tour_p)
        next_arr = np.roll(tour_p_arr, -1)
        true_len = true_dist[tour_p_arr, next_arr].sum()
        
        if true_len < best_true_len:
            best_true_len = true_len
            best_alpha = alpha
    
    # Build final penalized matrix with best alpha
    result = _build_penalized(edge_distance, edge_n_used, best_alpha)
    
    # Tiny noise for symmetry breaking
    rng = np.random.default_rng(42)
    noise = rng.uniform(0, 1e-6, size=(n, n))
    result += noise
    result = (result + result.T) / 2.0
    np.fill_diagonal(result, 0.0)
    result = np.maximum(result, 0.0)
    result = np.where(np.isfinite(result), result, edge_distance)
    
    return result


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Adaptive GLS penalty with bootstrap-tuned scale.
    """
    n = len(local_opt_tour)
    
    if time.perf_counter() > _DEADLINE - 20:
        # Emergency: return unpenalized
        return edge_distance.copy()
    
    # Fast path: if no edges have been used yet, no penalty needed
    tour_edges = [(local_opt_tour[i], local_opt_tour[(i + 1) % n]) for i in range(n)]
    edge_sum = sum(edge_n_used[a, b] for a, b in tour_edges)
    
    if edge_sum <= 0:
        return edge_distance.copy()
    
    return _bootstrap(edge_distance, local_opt_tour, edge_n_used)
