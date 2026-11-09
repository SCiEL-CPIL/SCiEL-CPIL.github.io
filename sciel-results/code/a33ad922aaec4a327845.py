"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 330.0
_MARGIN = 5.0  # seconds of safety margin


def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _start_time)


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


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    GLS penalty: penalize edges in the current local optimum tour
    based on their historical usage frequency.
    """
    n = len(local_opt_tour)
    if n <= 2:
        return edge_distance.copy()

    # Check deadline
    if _time_left() < 0.5:
        return edge_distance.copy()

    tour = list(local_opt_tour)

    # Compute median edge distance for scale normalization
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    med_dist = np.median(edge_distance[mask])
    if med_dist <= 0:
        med_dist = 1.0

    # Identify edges in the current tour
    tour_edges = set()
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        if u < v:
            tour_edges.add((u, v))
        else:
            tour_edges.add((v, u))

    # Try multiple penalty strategies and keep the best result
    best_len = float('inf')
    best_D_pen = None

    # Penalty strategies: (alpha, decay)
    # alpha controls penalty strength, decay controls how usage count affects penalty
    strategies = [
        (0.15, 1.0),
        (0.20, 1.0),
        (0.25, 1.0),
        (0.30, 1.0),
        (0.35, 1.0),
        (0.40, 1.0),
        (0.20, 1.5),
        (0.25, 1.5),
        (0.30, 1.5),
        (0.35, 1.5),
        (0.25, 2.0),
        (0.30, 2.0),
        (0.35, 2.0),
        (0.40, 2.0),
    ]

    max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
    if max_used < 1e-10:
        max_used = 1.0

    for alpha, decay in strategies:
        if _time_left() < 2.0:
            break

        # Create penalty matrix
        result = edge_distance.copy()

        # Apply penalty to tour edges based on usage
        for (u, v) in tour_edges:
            usage = edge_n_used[u, v]
            if usage > 0:
                # Normalized usage with decay
                norm_usage = usage / max_used
                penalty = alpha * med_dist * (norm_usage ** decay)
                result[u, v] += penalty
                result[v, u] += penalty

        # Add small deterministic perturbation to non-tour edges to break symmetry
        # Only add if we have time
        if _time_left() > 3.0:
            noise_scale = 0.01 * med_dist
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) not in tour_edges and (j, i) not in tour_edges:
                        pert = noise_scale * 0.1 * np.sin(i * 0.7 + j * 1.3)
                        result[i, j] += pert
                        result[j, i] += pert

        # Run 2-opt on the penalized matrix
        tour_pen = _two_opt(result, tour, max_passes=10)

        # Evaluate on true distances
        true_len = _closed_len(edge_distance, tour_pen)

        if true_len < best_len:
            best_len = true_len
            best_D_pen = result

    if best_D_pen is not None:
        return best_D_pen
    else:
        # Fallback: return original with mild penalty
        result = edge_distance.copy()
        for (u, v) in tour_edges:
            usage = edge_n_used[u, v]
            if usage > 0:
                penalty = 0.2 * med_dist * usage / (1.0 + usage)
                result[u, v] += penalty
                result[v, u] += penalty
        return result
# EVOLVE-BLOCK-END
