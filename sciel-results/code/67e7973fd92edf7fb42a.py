"""Guided local search penalty for TSP with model-and-optimize approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _two_opt(d, tour, max_passes=10):
    """Run 2-opt improvement on a tour using distance matrix d."""
    n = len(tour)
    tour = list(tour)
    
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d2 = tour[j], tour[(j + 1) % n]
                # Check if 2-opt move improves
                if d[a][c] + d[b][d2] < d[a][b] + d[c][d2]:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return tour


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty with smooth force field using log-sum-exp soft-min
    for distance to nearest tour edge, plus usage-modulated penalties.
    """
    start_time = time.time()
    DEADLINE = 280

    n = edge_distance.shape[0]
    if n == 0:
        return edge_distance.copy()

    D = edge_distance

    # Extract tour edges
    tour_edges = []
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_edges.append((a, b))

    m = len(tour_edges)

    # Build arrays of tour edge endpoints
    a_arr = np.array([e[0] for e in tour_edges], dtype=int)
    b_arr = np.array([e[1] for e in tour_edges], dtype=int)

    # Compute soft-min closeness using log-sum-exp for smoothness
    # For each (i,j), closeness = softmin over tour edges of min(D[i,a]+D[j,b], D[i,b]+D[j,a])
    # We use a temperature parameter for the soft-min
    
    # Precompute D columns for all tour edge endpoints
    # Stack all endpoint distances
    # Shape: (m, n) for da, (m, n) for db
    da_stack = D[:, a_arr]  # (n, m)
    db_stack = D[:, b_arr]  # (n, m)

    # For each (i, j), we need softmin over t of min(da_stack[i,t] + db_stack[j,t], db_stack[i,t] + da_stack[j,t])
    # c1[i,j,t] = da_stack[i,t] + db_stack[j,t]
    # c2[i,j,t] = db_stack[i,t] + da_stack[j,t]
    # c[i,j,t] = min(c1[i,j,t], c2[i,j,t])
    # closeness[i,j] = softmin_t(c[i,j,t])
    
    # To avoid creating huge (n,n,m) arrays, we process in chunks or use broadcasting carefully
    # For n=100, n*n*m = 100*100*100 = 1,000,000 which is manageable
    
    # Build c1 and c2 as (n, n, m) arrays
    # da_stack has shape (n, m), db_stack has shape (n, m)
    # c1[i,j,t] = da_stack[i,t] + db_stack[j,t]
    # Use broadcasting: da_stack[:, None, :] + db_stack[None, :, :]
    c1 = da_stack[:, None, :] + db_stack[None, :, :]  # (n, n, m)
    c2 = db_stack[:, None, :] + da_stack[None, :, :]  # (n, n, m)
    c = np.minimum(c1, c2)  # (n, n, m)

    # Soft-min along axis=2 (over tour edges)
    # softmin(x) = -log(sum(exp(-x/tau))) * tau
    # Use stable computation: subtract max before exp
    tau = 0.05  # temperature for soft-min
    
    # For numerical stability
    c_neg = -c / tau  # (n, n, m)
    # Subtract max along axis 2 for stability
    c_max = np.max(c_neg, axis=2, keepdims=True)  # (n, n, 1)
    c_neg_stable = c_neg - c_max
    
    # Sum of exps
    exp_sum = np.sum(np.exp(c_neg_stable), axis=2)  # (n, n)
    
    # softmin = -(max + log(sum(exp(stable)))) * tau
    # Note: c_max is already -max(c)/tau, so -c_max = max(c)/tau
    # softmin = -(c_max + log(exp_sum)) * tau = -c_max*tau - log(exp_sum)*tau
    # But c_max = -max(c)/tau, so -c_max*tau = max(c)
    # And -log(exp_sum)*tau is the correction term
    min_closeness = -(c_max.squeeze(-1) + np.log(exp_sum)) * tau

    # Scale factor: use median of off-diagonal distances as typical edge length
    off_diag = D[~np.eye(n, dtype=bool)]
    scale = np.median(off_diag) if len(off_diag) > 0 else 0.1
    if scale <= 0:
        scale = 0.1

    # Force field: decays with distance to nearest tour edge
    # beta controls the sharpness of the repulsion
    beta = 8.0
    force_field = np.exp(-beta * min_closeness)

    # Initialize penalty/reward matrix
    penalty = np.zeros((n, n))

    # Mask for current tour edges
    tour_mask = np.zeros((n, n), dtype=bool)
    for t in range(m):
        a = a_arr[t]
        b = b_arr[t]
        tour_mask[a, b] = True
        tour_mask[b, a] = True

    # 1. Apply penalty to current tour edges
    # Scale penalty with usage: more used edges get higher penalty
    max_used = edge_n_used.max() if edge_n_used.size > 0 else 0
    if max_used > 0:
        usage_norm = edge_n_used / max_used
    else:
        usage_norm = np.zeros_like(edge_n_used)
    
    # Penalty for tour edges: base + usage-dependent component
    base_penalty = 0.3 * scale
    usage_penalty = 0.3 * scale * usage_norm
    penalty[tour_mask] += base_penalty + usage_penalty[tour_mask]

    # 2. Apply small negative reward to non-tour edges that are close to the tour
    # and have low usage
    reward_mask = ~tour_mask
    reward_base = 0.05 * scale
    reward_amount = reward_base * force_field * (1.0 - usage_norm)
    
    penalty[reward_mask] -= reward_amount[reward_mask]

    # Combine: additive penalty/reward on top of true distances
    result = D + penalty

    # Ensure symmetry
    result = (result + result.T) / 2.0
    np.fill_diagonal(result, 0.0)

    # Clamp to non-negative
    result = np.maximum(result, 0.0)

    # Replace any non-finite values
    result = np.nan_to_num(result, nan=0.0, posinf=D.max() if D.max() > 0 else 1.0, neginf=0.0)

    return result
# EVOLVE-BLOCK-END
