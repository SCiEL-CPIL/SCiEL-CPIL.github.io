# EVOLVE-BLOCK-START
import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = 580.0  # safety margin before 630s kill

    def compute_P(support, weights):
        """Compute P[X1+X2+X3 < 2*X4] exactly."""
        n = len(support)
        # Pairwise sums: S[i,j] = support[i] + support[j]
        S = support[:, None] + support[None, :]  # n x n
        # For each (k, l) pair, compute sum of w_i*w_j where S[i,j] < 2*s_l - s_k
        # P = sum_{k,l} w_k * w_l * (sum_{i,j: S[i,j] < 2*s_l - s_k} w_i * w_j)
        # Vectorized: for each l, threshold = 2*s_l - s_k for all k
        # Let's build it as: for each l, for each k, count weighted pairs
        # n can be up to ~50-100 for speed
        w = weights / weights.sum()
        # S is n x n, w_i*w_j is outer product
        ww = w[:, None] * w[None, :]  # n x n
        # For each l: threshold_k = 2*s_l - s_k for k=0..n-1
        # We need for each k: sum of ww[i,j] where S[i,j] < 2*s_l - s_k
        # This is: for fixed l, create matrix M_k where M_k[i,j] = ww[i,j] if S[i,j] < 2*s_l - s_k else 0
        # Sum over i,j for each k, then dot with w_k
        
        # Efficient: for each l, compute cumulative sum over S
        # Sort S and use searchsorted
        S_flat = S.flatten()  # n^2
        ww_flat = ww.flatten()  # n^2
        # For each l and each k, threshold = 2*s_l - s_k
        # We need sum of ww_flat where S_flat < threshold
        # Use argsort approach or just loop over l (n iterations)
        
        P = 0.0
        for l in range(n):
            threshold = 2.0 * support[l] - support  # shape (n,)
            # For each k, find sum of ww where S < threshold[k]
            # Sort S once
            # Actually, let's precompute sorted S and cumulative ww
            # Do it per l for clarity
            sorted_idx = np.argsort(S_flat)
            S_sorted = S_flat[sorted_idx]
            ww_sorted = ww_flat[sorted_idx]
            cumsum_ww = np.cumsum(ww_sorted)
            # For each k: find first index where S_sorted >= threshold[k]
            # Then sum = cumsum_ww[idx-1] if idx > 0 else 0
            idxs = np.searchsorted(S_sorted, threshold, side='left')
            # sum_ww[k] = cumsum_ww[idxs[k]-1] if idxs[k] > 0 else 0
            sums = np.zeros(n)
            mask = idxs > 0
            sums[mask] = cumsum_ww[idxs[mask] - 1]
            P += np.dot(w, sums)
        
        return P

    def try_geometric(r, n_pts):
        """Try geometric distribution: w_i ∝ r^i, support = i * scale."""
        if r <= 0 or r >= 1 or n_pts < 2:
            return None, 0.0
        support = np.arange(n_pts, dtype=np.float64)
        weights = r ** support
        # Normalize
        wsum = weights.sum()
        if wsum <= 1e-9:
            return None, 0.0
        weights = weights / wsum
        P = compute_P(support, weights)
        return (support, weights), P

    def try_two_point(p, M):
        """Two point: mass p at 0, mass (1-p) at M."""
        if p <= 0 or p >= 1 or M <= 0:
            return None, 0.0
        support = np.array([0.0, M], dtype=np.float64)
        weights = np.array([p, 1.0 - p], dtype=np.float64)
        P = compute_P(support, weights)
        return (support, weights), P

    def try_three_point(p0, p1, p2, a, b):
        """Three point: mass p0 at 0, p1 at a, p2 at b."""
        if p0 <= 0 or p1 <= 0 or p2 <= 0 or a <= 0 or b <= 0:
            return None, 0.0
        support = np.array([0.0, a, b], dtype=np.float64)
        weights = np.array([p0, p1, p2], dtype=np.float64)
        P = compute_P(support, weights)
        return (support, weights), P

    best_result = None
    best_P = 0.0

    # Try geometric distributions with various r and n
    for r in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
        for n_pts in [10, 15, 20, 25, 30, 40, 50]:
            if time.time() > deadline:
                break
            res, P = try_geometric(r, n_pts)
            if res is not None and P > best_P:
                best_P = P
                best_result = res
        if time.time() > deadline:
            break

    # Try two-point distributions
    for p in [0.3, 0.4, 0.5, 0.55, 0.58, 0.6, 0.65, 0.7]:
        for M in [1.0, 2.0, 3.0, 5.0, 10.0]:
            if time.time() > deadline:
                break
            res, P = try_two_point(p, M)
            if res is not None and P > best_P:
                best_P = P
                best_result = res
        if time.time() > deadline:
            break

    # Try three-point distributions
    for p0 in [0.4, 0.5, 0.6, 0.7]:
        for p1 in [0.1, 0.2, 0.3]:
            p2 = 1.0 - p0 - p1
            if p2 <= 0.01:
                continue
            for a in [0.5, 1.0, 1.5, 2.0]:
                for b in [2.0, 3.0, 4.0, 5.0, 8.0, 10.0]:
                    if b <= a:
                        continue
                    if time.time() > deadline:
                        break
                    res, P = try_three_point(p0, p1, p2, a, b)
                    if res is not None and P > best_P:
                        best_P = P
                        best_result = res
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # If nothing found, return baseline
    if best_result is None:
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights

    support, weights = best_result
    return support, weights

# EVOLVE-BLOCK-END
