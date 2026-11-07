"""Guided local search penalty rule for the TSP (direct approach)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.monotonic()
_DEADLINE = 325.0  # safety margin before 330s kill


def _elapsed():
    return time.monotonic() - _start_time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Direct approach with repulsive force field around tour edges.
    
    Key ideas:
    1. Base penalty scales with usage count (soft-saturating)
    2. Tour edges get a usage-aware boost (heavily-used tour edges penalized more)
    3. A smooth 'field' penalty decays with metric distance to nearest tour edge
    4. All penalties scaled by median edge length for adaptivity
    """
    # Deadline guard
    if _elapsed() > _DEADLINE:
        return np.asarray(edge_distance, dtype=float).copy()
    
    D = np.asarray(edge_distance, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1] or D.shape[0] < 3:
        return D.copy()
    
    n = D.shape[0]
    
    # Guard against non-finite input
    if not np.all(np.isfinite(D)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        return base
    
    used = np.asarray(edge_n_used, dtype=float)
    if used.shape != D.shape:
        used = np.zeros_like(D)
    if not np.all(np.isfinite(used)):
        used = np.nan_to_num(used, nan=0.0, posinf=0.0, neginf=0.0)
    used = np.maximum(used, 0.0)
    
    # Reference scale
    ref = float(np.median(D))
    if not np.isfinite(ref) or ref <= 0.0:
        ref = 1.0
    
    # Parse tour
    tour = np.asarray(local_opt_tour, dtype=int)
    tour_edges = []
    if tour.ndim == 1 and len(tour) >= 2:
        for i in range(len(tour) - 1):
            a, b = int(tour[i]), int(tour[i + 1])
            if 0 <= a < n and 0 <= b < n:
                tour_edges.append((a, b))
        a, b = int(tour[-1]), int(tour[0])
        if 0 <= a < n and 0 <= b < n:
            tour_edges.append((a, b))
    
    # Focused penalty: primarily on tour edges with usage-aware scaling
    # Minimal base penalty for non-tour used edges (inspiration: focused > broad)
    base_rate = 0.02
    penalty = base_rate * ref * used / (1.0 + used)
    
    # Tour edge penalties: stronger, usage-scaled
    tour_boost_base = 0.45 * ref
    if tour_edges:
        for (a, b) in tour_edges:
            u = 0.5 * (used[a, b] + used[b, a])
            # Saturating usage scaling: more penalty for heavily used edges
            boost = tour_boost_base * (1.0 + 0.6 * u / (1.0 + u))
            penalty[a, b] += boost
            penalty[b, a] += boost
        
        # Repulsive force field: smooth soft-min distance to nearest tour edge length
        # This creates a continuous exclusion zone around the current tour's edge lengths
        tour_dists = np.array([D[a, b] for (a, b) in tour_edges])
        
        # Efficient soft-min: compute per-pair min distance to tour edge lengths
        # Use a chunked approach to avoid n^3 memory
        field_penalty = np.zeros((n, n), dtype=float)
        temp = 0.05 * ref
        decay_scale = 0.08 * ref
        field_rate = 0.10
        
        # Process in chunks to limit memory
        chunk_size = max(1, n // 4)
        for i_start in range(0, n, chunk_size):
            i_end = min(i_start + chunk_size, n)
            # diff shape: (chunk, n, n_tour_edges)
            diff = np.abs(D[i_start:i_end, :, np.newaxis] - tour_dists[np.newaxis, np.newaxis, :])
            diff_clipped = np.clip(diff, 0.0, 10.0 * ref)
            # log-sum-exp for numerical stability
            max_val = np.max(-diff_clipped / temp, axis=2, keepdims=True)
            log_sum = max_val[..., 0] + np.log(np.sum(np.exp(-diff_clipped / temp - max_val), axis=2))
            soft_min_diff = -temp * log_sum  # shape (chunk, n)
            
            fp_chunk = field_rate * ref * np.exp(-soft_min_diff / decay_scale)
            field_penalty[i_start:i_end, :] = fp_chunk
        
        # Zero out diagonal and tour edges (already penalized above)
        np.fill_diagonal(field_penalty, 0.0)
        for (a, b) in tour_edges:
            field_penalty[a, b] = 0.0
            field_penalty[b, a] = 0.0
        
        penalty += field_penalty
    
    # Hard cap
    max_penalty = 0.85 * ref
    penalty = np.minimum(penalty, max_penalty)
    penalty = np.maximum(penalty, 0.0)
    
    # Add penalty to the distance matrix
    out = D + penalty
    
    # Enforce exact symmetry
    out = 0.5 * (out + out.T)
    
    # Ensure diagonal is zero
    np.fill_diagonal(out, 0.0)
    
    if not np.all(np.isfinite(out)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        out = 0.5 * (base + base.T)
        np.fill_diagonal(out, 0.0)
    
    return out
# EVOLVE-BLOCK-END
