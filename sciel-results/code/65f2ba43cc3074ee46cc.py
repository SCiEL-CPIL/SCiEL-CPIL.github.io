# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] for i.i.d. draws from the discrete measure.
    
    Uses the formula: P = sum_{k,l} w_k * w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i * w_j)
    
    For efficiency, we precompute the convolution of the distribution with itself
    to get the distribution of X1+X2, then for each (k,l) we need the CDF of
    X1+X2 evaluated at 2*s_l - s_k (strictly less than).
    """
    n = len(support)
    # Compute all pairwise sums s_i + s_j and their probabilities
    # The distribution of S = X1 + X2 has support {s_i + s_j} with prob w_i * w_j
    
    # Use numpy for the pairwise sums
    # sum_matrix[i,j] = s_i + s_j
    # prob_matrix[i,j] = w_i * w_j
    
    sum_matrix = support[:, None] + support[None, :]  # shape (n, n)
    prob_matrix = weights[:, None] * weights[None, :]  # shape (n, n)
    
    # Flatten
    all_sums = sum_matrix.ravel()  # shape (n*n,)
    all_probs = prob_matrix.ravel()  # shape (n*n,)
    
    # Sort by sum value
    sort_idx = np.argsort(all_sums, kind='mergesort')
    sorted_sums = all_sums[sort_idx]
    sorted_probs = all_probs[sort_idx]
    
    # Cumulative sum for CDF
    cum_probs = np.cumsum(sorted_probs)
    
    # For each (k, l), we need sum of probs where s_i + s_j < 2*s_l - s_k
    # threshold = 2*s_l - s_k
    # We need the CDF at threshold (strictly less than)
    
    # Vectorize over (k, l) pairs
    # thresholds[k, l] = 2 * s_l - s_k
    thresholds = 2.0 * support[None, :] - support[:, None]  # shape (n, n), thresholds[k,l]
    
    # For each threshold, find the number of sorted_sums that are strictly less
    # Using searchsorted: np.searchsorted(sorted_sums, threshold, side='left') gives
    # the index of the first element >= threshold, so the count of elements < threshold
    # is that index.
    
    # But we need the sum of probabilities for those elements, not just the count.
    # We can use np.searchsorted to get the index, then use cum_probs[index-1] (with care for index=0)
    
    # Vectorized approach:
    # For each threshold, find idx = searchsorted(sorted_sums, threshold, side='left')
    # Then the CDF value is cum_probs[idx-1] if idx > 0, else 0
    
    # Let's do this vectorized
    # thresholds_flat = thresholds.ravel()  # shape (n*n,)
    # idxs = np.searchsorted(sorted_sums, thresholds_flat, side='left')
    # cdf_vals = np.where(idxs > 0, cum_probs[np.maximum(idxs - 1, 0)], 0.0)
    # cdf_vals[idxs == 0] = 0.0
    
    # Actually, let me be more careful:
    thresholds_flat = thresholds.ravel()
    idxs = np.searchsorted(sorted_sums, thresholds_flat, side='left')
    
    # Create an array for cdf values
    cdf_vals = np.zeros(len(thresholds_flat), dtype=np.float64)
    mask = idxs > 0
    cdf_vals[mask] = cum_probs[idxs[mask] - 1]
    
    # Now P = sum_{k,l} w_k * w_l * cdf_vals[k*n + l]
    # w_k * w_l is just prob_matrix[k, l]
    # cdf_vals reshaped to (n, n)
    cdf_matrix = cdf_vals.reshape(n, n)
    
    P = np.sum(prob_matrix * cdf_matrix)
    
    return P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # 10 second safety margin
    
    # Strategy: Use a geometric-like distribution on non-negative integers.
    # The key insight is that we want X4 to be large with some probability, 
    # which makes 2*X4 large, increasing the chance that X1+X2+X3 < 2*X4.
    # But we also need X1, X2, X3 to be small with high probability.
    # A distribution that puts mass on 0 and then has a tail might work well.
    
    # Let's try a mixture: high probability on 0, and a geometric tail.
    # Or a two-point distribution: mass p on 0 and mass (1-p) on some large value M.
    
    # For a two-point distribution: support = [0, M], weights = [p, 1-p]
    # P[X1+X2+X3 < 2*X4]
    # X4 is 0 with prob p, M with prob (1-p)
    # If X4 = 0: need X1+X2+X3 < 0, impossible since support is non-negative. So P = 0.
    # If X4 = M: need X1+X2+X3 < 2M.
    # X1+X2+X3 can be 0, M, 2M, or 3M.
    # P(X1+X2+X3 < 2M | X4=M) = P(X1+X2+X3 = 0) + P(X1+X2+X3 = M)
    # = p^3 + 3*p^2*(1-p)
    # So P = (1-p) * (p^3 + 3*p^2*(1-p))
    
    # Let's optimize this: f(p) = (1-p)*(p^3 + 3*p^2*(1-p))
    # = (1-p)*(p^3 + 3p^2 - 3p^3)
    # = (1-p)*(3p^2 - 2p^3)
    # = 3p^2 - 2p^3 - 3p^3 + 2p^4
    # = 3p^2 - 5p^3 + 2p^4
    
    # f'(p) = 6p - 15p^2 + 8p^3
    # Set to 0: 8p^3 - 15p^2 + 6p = 0
    # p(8p^2 - 15p + 6) = 0
    # 8p^2 - 15p + 6 = 0
    # p = (15 ± sqrt(225 - 192)) / 16 = (15 ± sqrt(33)) / 16
    # sqrt(33) ≈ 5.7446
    # p = (15 + 5.7446)/16 ≈ 1.2965 (invalid, > 1)
    # p = (15 - 5.7446)/16 ≈ 0.5784
    
    # Let's compute f(0.5784):
    p_opt = (15 - np.sqrt(33)) / 16
    f_val = 3*p_opt**2 - 5*p_opt**3 + 2*p_opt**4
    
    # For comparison, the uniform on {0,...,31} gives some value.
    # Let's compute a few candidate distributions and pick the best.
    
    candidates = []
    
    # Candidate 1: Two-point distribution with optimal p
    support1 = np.array([0.0, 1.0])
    weights1 = np.array([p_opt, 1 - p_opt])
    candidates.append((support1, weights1))
    
    # Candidate 2: Three-point distribution
    # support = [0, 1, 2], weights = [a, b, c]
    # This is more complex to optimize analytically, so let's try a few values.
    
    # Let's try a geometric distribution on {0, 1, 2, ..., n-1}
    # w_i = (1-q) * q^i
    # For different q values.
    
    best_P = -1
    best_support = support1
    best_weights = weights1
    
    # Evaluate the two-point candidate
    P1 = _compute_P(support1, weights1)
    if P1 > best_P:
        best_P = P1
        best_support = support1.copy()
        best_weights = weights1.copy()
    
    # Try geometric distributions with different parameters
    for n_pts in [16, 32, 64]:
        for q in np.linspace(0.5, 0.95, 20):
            if time.time() > deadline:
                break
            support_g = np.arange(n_pts, dtype=np.float64)
            weights_g = (1 - q) * q ** np.arange(n_pts)
            # Normalize to make sure
            weights_g = weights_g / weights_g.sum()
            P_g = _compute_P(support_g, weights_g)
            if P_g > best_P:
                best_P = P_g
                best_support = support_g.copy()
                best_weights = weights_g.copy()
    
    if time.time() > deadline:
        break
    
    # Try a distribution with mass on 0 and a geometric tail starting from 1
    # support = [0, 1, 2, ..., n-1]
    # weights = [p0, (1-p0)*q, (1-p0)*q^2, ...]
    for n_pts in [16, 32, 64]:
        for p0 in np.linspace(0.3, 0.8, 10):
            for q in np.linspace(0.5, 0.9, 10):
                if time.time() > deadline:
                    break
                support_mix = np.arange(n_pts, dtype=np.float64)
                weights_mix = np.zeros(n_pts)
                weights_mix[0] = p0
                remaining = 1 - p0
                for i in range(1, n_pts):
                    weights_mix[i] = remaining * q ** (i - 1) * (1 - q)
                # Normalize
                weights_mix = weights_mix / weights_mix.sum()
                P_mix = _compute_P(support_mix, weights_mix)
                if P_mix > best_P:
                    best_P = P_mix
                    best_support = support_mix.copy()
                    best_weights = weights_mix.copy()
    
    # Ensure validity
    if len(best_support) < 1:
        best_support = np.array([0.0])
        best_weights = np.array([1.0])
    
    # Clean up: ensure no negative values
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    # Ensure sum of weights > 0
    if best_weights.sum() <= 1e-9:
        best_support = np.array([0.0])
        best_weights = np.array([1.0])
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
