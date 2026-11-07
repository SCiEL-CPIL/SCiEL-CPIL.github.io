"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right
    after item i in the same bin. Larger is more attractive.
    
    Combines a spectral embedding of the item compatibility graph (to capture
    global structural affinities and community structures) with a local
    gap-based metric that favors tight fits and larger items (FFD principle).
    
    The spectral component identifies items that naturally co-occur in good
    packings, while the local component ensures that within a bin, the most
    space-efficient placements are preferred.
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
    
    # --- Step 1: Build the compatibility graph and spectral embedding ---
    
    # Pair sums and gaps
    pair_sum = d[:, None] + d[None, :]
    gap = capacity - pair_sum
    fits = gap >= 0
    np.fill_diagonal(fits, False)
    
    # Edge weights for the compatibility graph: tighter fits have larger weights
    # Use negative residual (i.e., smaller gap -> larger weight)
    # We want weights in a stable range, so use -gap for fitting pairs
    edge_weight = np.where(fits, -gap, 0.0)
    
    # Normalize edge weights to [0, 1]
    max_w = np.max(edge_weight) if np.any(edge_weight > 0) else 0.0
    if max_w > 0:
        adj = edge_weight / max_w
    else:
        adj = np.zeros_like(edge_weight)
    
    # Make symmetric for real eigenvalues
    adj_sym = (adj + adj.T) / 2.0
    np.fill_diagonal(adj_sym, 0.0)
    
    # Compute spectral embedding
    try:
        eigvals, eigvecs = np.linalg.eigh(adj_sym)
    except np.linalg.LinAlgError:
        # Fallback to simple heuristic
        H = np.where(fits, d[None, :] / (gap + 1.0), 1e-9)
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        return H
    
    # Use top k eigenvectors (k=4 for 120 items)
    k = min(4, n)
    # eigvecs columns sorted by ascending eigenvalue; take largest
    top_eigvecs = eigvecs[:, -k:]
    embedding = top_eigvecs  # shape (n, k)
    
    # Compute pairwise squared distances in embedding space
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
        # Fallback: combine what we have so far
        gap_score = np.where(fits, 1.0 / (1.0 + gap), 0.0)
        size_bias = d[None, :] / capacity
        heuristic = similarity * gap_score * size_bias
        heuristic = np.where(fits, heuristic, 1e-9)
        heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
        return heuristic
    
    # --- Step 2: Local gap-based component with FFD bias ---
    
    # Prefer tight fits: 1/(1+gap) gives higher value for smaller gaps
    gap_score = np.where(fits, 1.0 / (1.0 + gap), 0.0)
    
    # FFD bias: favor placing larger items
    size_bias = d[None, :] / capacity  # shape (1, n), broadcasts to (n, n)
    
    # --- Step 3: Combine spectral similarity with local metrics ---
    
    # The spectral similarity captures global structure (which items tend to
    # be packed together in good solutions), while gap_score and size_bias
    # capture local efficiency.
    
    # Normalize similarity to [0, 1] range (it already is, but ensure)
    sim_max = np.max(similarity)
    if sim_max > 0:
        similarity_norm = similarity / sim_max
    else:
        similarity_norm = similarity
    
    # Weighted combination: spectral component provides a multiplicative boost
    # to pairs that are structurally compatible, while the local component
    # determines the base attractiveness.
    
    # Base heuristic from local metrics
    base_heuristic = gap_score * size_bias
    
    # Combined heuristic: spectral similarity modulates the local heuristic
    # We use a convex combination to balance the two components
    alpha = 0.5  # weight for spectral component
    
    heuristic = (1.0 - alpha) * base_heuristic + alpha * similarity_norm * base_heuristic
    
    # Alternative: multiplicative combination might be stronger
    # heuristic = base_heuristic * (0.5 + 0.5 * similarity_norm)
    
    # Actually, let's try a different combination that gives more weight
    # to the spectral component for structurally compatible pairs
    # and falls back to local metrics otherwise
    
    # Final combination: use spectral similarity as a multiplier that
    # amplifies structurally good pairs
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
