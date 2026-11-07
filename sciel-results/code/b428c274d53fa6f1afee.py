"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right
    after item i in the same bin. Larger is more attractive.
    
    Uses a "dead gap" heuristic: for each pair (i, j), compute the residual
    space after placing both items in a bin. Count how many other items could
    fit in that residual. Pairs that leave small residuals (few or zero items
    can fit) are highly attractive because they create voids that can't be
    exploited, effectively "locking in" tight packings.
    
    Combines this with a spectral embedding to capture global structural
    affinities and FFD bias to prefer larger items.
    """
    start_time = time.time()
    
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Deadline guard
    if time.time() - start_time > 290:
        # Fallback: simple gap-based heuristic
        pair_sum = d[:, None] + d[None, :]
        gap = capacity - pair_sum
        valid = gap >= 0
        np.fill_diagonal(valid, False)
        H = np.where(valid, d[None, :] / (gap + 1.0), 1e-9)
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        return H
    
    # --- Step 1: Compute dead gap score ---
    # For pair (i, j), residual = capacity - (d_i + d_j)
    # Count how many items k (k != i, k != j) can fit in residual: d_k <= residual
    # Fewer fitting items = better (dead gap)
    
    pair_sum = d[:, None] + d[None, :]
    residual = capacity - pair_sum  # shape (n, n)
    fits = residual >= 0
    np.fill_diagonal(fits, False)
    
    # For each pair (i, j), count items that fit in residual[i,j]
    # residual has shape (n, n). For each (i,j), we need count of d[k] <= residual[i,j]
    # where k != i and k != j.
    
    # Sort demands for efficient counting
    d_sorted = np.sort(d)
    
    # For each residual value, use searchsorted to count items that fit
    # residual[i,j] is the space left. Items with d[k] <= residual[i,j] can fit.
    # We need to exclude i and j themselves.
    
    # First, count all items that fit (including i and j)
    # searchsorted gives index of first element > value, so count = index
    residual_flat = residual.ravel()
    fit_counts_all = np.searchsorted(d_sorted, residual_flat, side='right')
    fit_counts_all = fit_counts_all.reshape(n, n)
    
    # Now subtract 1 if d[i] <= residual[i,j] (item i itself fits)
    # and subtract 1 if d[j] <= residual[i,j] (item j itself fits)
    # d[i] <= residual[i,j] means d[i] <= capacity - d[i] - d[j], i.e., 2*d[i] + d[j] <= capacity
    # d[j] <= residual[i,j] means d[j] <= capacity - d[i] - d[j], i.e., d[i] + 2*d[j] <= capacity
    
    di = d[:, None]  # shape (n, 1)
    dj = d[None, :]  # shape (1, n)
    
    i_fits = (2.0 * di + dj) <= capacity  # shape (n, n)
    j_fits = (di + 2.0 * dj) <= capacity  # shape (n, n)
    
    # Only count if the pair actually fits
    fit_counts = fit_counts_all - i_fits.astype(np.int64) - j_fits.astype(np.int64)
    fit_counts = np.maximum(fit_counts, 0)
    
    # Deadline guard
    if time.time() - start_time > 290:
        # Fallback: use dead gap score alone
        # Inverse of (fit_counts + 1): fewer fitting items = higher score
        dead_score = np.where(fits, 1.0 / (1.0 + fit_counts.astype(np.float64)), 0.0)
        size_bias = d[None, :] / capacity
        H = dead_score * size_bias
        H = np.where(fits, H, 1e-9)
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        return H
    
    # --- Step 2: Spectral embedding for global structure ---
    
    # Build adjacency from dead gap score: pairs with few fitting items are more connected
    edge_weight = np.where(fits, 1.0 / (1.0 + fit_counts.astype(np.float64)), 0.0)
    
    max_w = np.max(edge_weight) if np.any(edge_weight > 0) else 0.0
    if max_w > 0:
        adj = edge_weight / max_w
    else:
        adj = np.zeros_like(edge_weight)
    
    # Make symmetric
    adj_sym = (adj + adj.T) / 2.0
    np.fill_diagonal(adj_sym, 0.0)
    
    # Compute spectral embedding
    try:
        eigvals, eigvecs = np.linalg.eigh(adj_sym)
    except np.linalg.LinAlgError:
        # Fallback to simple heuristic
        dead_score = np.where(fits, 1.0 / (1.0 + fit_counts.astype(np.float64)), 0.0)
        size_bias = d[None, :] / capacity
        H = dead_score * size_bias
        H = np.where(fits, H, 1e-9)
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        return H
    
    # Use top k eigenvectors
    k = min(4, n)
    top_eigvecs = eigvecs[:, -k:]
    embedding = top_eigvecs
    
    # Compute pairwise squared distances
    diff = embedding[:, None, :] - embedding[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Convert to similarity via Gaussian kernel
    positive_dists = dist_sq[dist_sq > 0]
    if len(positive_dists) > 0:
        sigma = np.median(positive_dists)
    else:
        sigma = 1.0
    if sigma < 1e-12:
        sigma = 1.0
    
    similarity = np.exp(-dist_sq / (2.0 * sigma))
    
    # Deadline guard
    if time.time() - start_time > 290:
        # Combine what we have
        dead_score = np.where(fits, 1.0 / (1.0 + fit_counts.astype(np.float64)), 0.0)
        size_bias = d[None, :] / capacity
        sim_max = np.max(similarity)
        if sim_max > 0:
            similarity_norm = similarity / sim_max
        else:
            similarity_norm = similarity
        H = dead_score * size_bias * (0.3 + 0.7 * similarity_norm)
        H = np.where(fits, H, 1e-9)
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        return H
    
    # --- Step 3: Combine dead gap score with spectral similarity and FFD bias ---
    
    # Dead gap score: inverse of number of items that can fit in residual
    dead_score = np.where(fits, 1.0 / (1.0 + fit_counts.astype(np.float64)), 0.0)
    
    # FFD bias: favor placing larger items
    size_bias = d[None, :] / capacity
    
    # Normalize similarity
    sim_max = np.max(similarity)
    if sim_max > 0:
        similarity_norm = similarity / sim_max
    else:
        similarity_norm = similarity
    
    # Base heuristic from dead gap and size
    base_heuristic = dead_score * size_bias
    
    # Combine with spectral similarity as a modulator
    heuristic = base_heuristic * (0.3 + 0.7 * similarity_norm)
    
    # Ensure non-fitting pairs are 1e-9
    heuristic = np.where(fits, heuristic, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Deadline guard
    if time.time() - start_time > 290:
        return heuristic
    
    return heuristic
# EVOLVE-BLOCK-END
