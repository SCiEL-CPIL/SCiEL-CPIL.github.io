# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly."""
    n = len(support)
    if n == 0:
        return 0.0
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    s = support
    
    # 1. Compute distribution of S3 = X1 + X2 + X3
    # P(S3 = s_i + s_j + s_k) = sum_{i,j,k: s_i+s_j+s_k = v} w_i w_j w_k
    # We can do this by convolving the discrete measure with itself 3 times.
    # However, support points are arbitrary, so we can't just use np.convolve directly on indices unless support is uniform grid.
    # For arbitrary support, we must sum over all triples (i, j, k).
    # Complexity: O(n^3) to generate all sums, then sort/bucket.
    # Given n ~ 100, n^3 = 1,000,000 which is fast.
    
    # Generate all sums of 3 points and their probabilities
    # s_i + s_j + s_k
    # w_i * w_j * w_k
    
    # To save memory and time, we can iterate or use broadcasting if n is small enough.
    # n=100 -> 100^3 = 1e6 floats. 1e6 * 8 bytes = 8MB. This is fine.
    
    s_i = s[:, None, None]
    s_j = s[None, :, None]
    s_k = s[None, None, :]
    
    sums_3 = s_i + s_j + s_k
    prods_3 = w[:, None, None] * w[None, :, None] * w[None, None, :]
    
    # Flatten
    sums_3_flat = sums_3.flatten()
    prods_3_flat = prods_3.flatten()
    
    # We need to sum probabilities for identical sums.
    # Since floating point sums might not be identical due to precision, 
    # but for discrete support with rational-like values, exact equality might hold if inputs are simple.
    # However, to be robust, we should sort and group, or use np.unique with return_inverse.
    # np.unique is stable and handles floating point exact matches.
    # If support values are floats like 0.1, 0.2, 0.3, 0.1+0.2+0.3 == 0.6 might be True or False.
    # The task says "computed exactly in float64". This implies we treat the float64 values as the identity.
    # So we group by exact float64 value.
    
    unique_sums, inverse_indices = np.unique(sums_3_flat, return_inverse=True)
    
    # Sum probabilities for each unique sum
    # np.bincount requires integer indices. inverse_indices are integers.
    probs_3 = np.bincount(inverse_indices, weights=prods_3_flat)
    
    # Now we have unique_sums (array of size M <= n^3) and probs_3 (array of size M)
    # P = sum_{k, l} w_k w_l P(S3 < 2 s_l - s_k)
    # Note: The sum is over k (for X4? No, indices in formula: X1+X2+X3 < 2*X4)
    # Formula: P = sum_{k, l} w_k * w_l * (sum_{i, j} w_i * w_j [s_i + s_j < 2*s_l - s_k])
    # Wait, the formula in the prompt is:
    # P = sum over k, l of w_k * w_l * (sum of w_i * w_j over pairs (i, j) with s_i + s_j < 2*s_l - s_k)
    # This formula seems to define X1,X2 from i,j and X3,X4 from k,l?
    # Let's re-read carefully: "P = Prob[X1 + X2 + X3 < 2*X4]"
    # The formula provided: "P = sum over k, l of w_k * w_l * (sum of w_i * w_j over pairs (i, j) with s_i + s_j < 2*s_l - s_k)"
    # This formula calculates P(X1+X2 < 2X4 - X3) where X1,X2 ~ mu and X3,X4 ~ mu.
    # This is equivalent to P(X1+X2+X3 < 2X4).
    # My previous derivation used S3 = X1+X2+X3.
    # The formula in prompt uses S2 = X1+X2.
    # Let's stick to the prompt's formula structure to be safe, or verify equivalence.
    # P(X1+X2+X3 < 2X4) = E[ P(X1+X2 < 2X4 - X3 | X3, X4) ]
    # = sum_{k,l} w_k w_l P(S2 < 2 s_l - s_k)
    # This matches the prompt's formula if k corresponds to X3 and l to X4.
    # So I should compute distribution of S2 = X1 + X2.
    
    # Let's re-calculate using S2 for efficiency and correctness relative to prompt formula.
    # S2 = X1 + X2.
    # n^2 terms.
    
    s_i2 = s[:, None]
    s_j2 = s[None, :]
    sums_2 = s_i2 + s_j2
    prods_2 = w[:, None] * w[None, :]
    
    sums_2_flat = sums_2.flatten()
    prods_2_flat = prods_2.flatten()
    
    unique_sums_2, inverse_indices_2 = np.unique(sums_2_flat, return_inverse=True)
    probs_2 = np.bincount(inverse_indices_2, weights=prods_2_flat)
    
    # Now P = sum_{k, l} w_k w_l P(S2 < 2 s_l - s_k)
    # P(S2 < T) = sum_{v in unique_sums_2, v < T} probs_2[v]
    
    # To speed up, sort unique_sums_2 and compute cumulative sum of probs_2.
    sort_idx = np.argsort(unique_sums_2)
    sorted_sums_2 = unique_sums_2[sort_idx]
    sorted_probs_2 = probs_2[sort_idx]
    cdf_2 = np.cumsum(sorted_probs_2)
    
    P_total = 0.0
    # Iterate over k (X3) and l (X4)
    for k in range(n):
        for l in range(n):
            T = 2.0 * s[l] - s[k]
            # Find count of sorted_sums_2 < T
            # np.searchsorted gives index of first element >= T
            idx = np.searchsorted(sorted_sums_2, T, side='left')
            if idx > 0:
                p_cond = cdf_2[idx - 1]
            else:
                p_cond = 0.0
            
            P_total += w[k] * w[l] * p_cond
            
    return P_total


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Returns a discrete measure maximizing P[X1+X2+X3 < 2*X4].
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    P_best = -1.0
    support_best = np.array([0.0], dtype=np.float64)
    weights_best = np.array([1.0], dtype=np.float64)
    
    def update_best(sup, wts, P_val):
        nonlocal P_best, support_best, weights_best
        if P_val > P_best:
            P_best = P_val
            support_best = sup.copy()
            weights_best = wts.copy()
    
    # 1. Try uniform distributions on {0, 1, ..., n-1}
    # Larger n approximates continuous uniform on [0,1] (scaled)
    # Note: scaling doesn't change P. So uniform on [0,1] is the limit.
    for n in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        if time.time() > deadline:
            break
        support = np.linspace(0.0, 1.0, n)
        weights = np.ones(n, dtype=np.float64)
        P = _compute_P(support, weights)
        update_best(support, weights, P)
        
    # 2. Try Beta-like distributions (discretized)
    # Beta(alpha, beta) on [0,1].
    # We want mass near 0 for X1,X2,X3 to make sum small, and mass near 1 for X4 to make 2X4 large?
    # Actually, we want X1+X2+X3 small AND 2X4 large.
    # So X1,X2,X3 should be small (Beta(1, 2) or similar), X4 should be large (Beta(2, 1) or similar).
    # But they are i.i.d. So we need a single distribution that balances this.
    # If distribution is skewed towards 0, sum is small, but 2X4 is also small.
    # If distribution is skewed towards 1, sum is large, but 2X4 is large.
    # Uniform might be a local optimum or not.
    # Let's try various Beta shapes.
    
    for n in [32, 64, 128, 256]:
        if time.time() > deadline:
            break
        for alpha in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
            for beta in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
                # Generate support points
                support = np.linspace(0.0, 1.0, n)
                # Calculate weights using Beta PDF
                # Beta PDF: x^(alpha-1) (1-x)^(beta-1) / B(alpha, beta)
                # We can just use the unnormalized weights, as they will be normalized in _compute_P
                # Handle x=0 or x=1 carefully if alpha<1 or beta<1
                x = support
                # Add small epsilon to avoid 0^negative
                x_safe = np.clip(x, 1e-12, 1.0 - 1e-12)
                weights = (x_safe ** (alpha - 1)) * ((1.0 - x_safe) ** (beta - 1))
                # Check for NaNs or Infs
                if not np.all(np.isfinite(weights)):
                    continue
                if np.sum(weights) <= 1e-9:
                    continue
                P = _compute_P(support, weights)
                update_best(support, weights, P)
                
    # 3. Try specific structured distributions like "Two peaks"
    # One peak near 0, one near 1.
    # This might help: small values for sum, large value for X4.
    # If we have high prob at 0 and high prob at 1.
    # P(X1+X2+X3 < 2X4)
    # If X4=1 (prob p), 2X4=2. Sum < 2 is always true for support in [0,1].
    # So if X4 is 1, condition is always satisfied.
    # If X4=0, condition is Sum < 0, which is false (unless sum=0 and strict inequality fails? 0<0 is False).
    # So we want X4 to be large.
    # But X1,X2,X3 also come from same dist.
    # If dist is {0, 1} with prob p at 1.
    # P = P(X4=1) * 1 + P(X4=0) * 0 = p.
    # So for 2-point {0,1}, P = p. Max p is 1 (degenerate at 1).
    # If degenerate at 1: X=1. 1+1+1 < 2*1 => 3 < 2 False. P=0.
    # If degenerate at 0: 0<0 False. P=0.
    # So we need spread.
    
    # Let's try {0, 1} with optimized p.
    # P = p (as derived above? Let's check).
    # X1,X2,X3,X4 in {0,1}.
    # Sum S = X1+X2+X3. S in {0,1,2,3}.
    # 2X4 in {0,2}.
    # If X4=1 (prob p), 2X4=2. S < 2 means S=0 or S=1.
    # P(S=0) = (1-p)^3.
    # P(S=1) = 3(1-p)^2 p.
    # P(S<2 | X4=1) = (1-p)^3 + 3(1-p)^2 p = (1-p)^2 (1-p + 3p) = (1-p)^2 (1+2p).
    # If X4=0 (prob 1-p), 2X4=0. S < 0 is impossible.
    # So P = p * (1-p)^2 (1+2p).
    # Let f(p) = p(1-2p+p^2)(1+2p) = p(1+2p-2p-4p^2+p^2+2p^3) = p(1-3p^2+2p^3) = p - 3p^3 + 2p^4.
    # f'(p) = 1 - 9p^2 + 8p^3 = 0.
    # p=1 is a root: 1-9+8=0.
    # (p-1)(8p^2 - p - 8) = 0? No.
    # Let's solve 8p^3 - 9p^2 + 1 = 0.
    # p=1 is a root. Divide by (p-1): 8p^2 - p - 1 = 0.
    # Roots: (1 +/- sqrt(1+32))/16 = (1 +/- sqrt(33))/16.
    # sqrt(33) ~ 5.74.
    # p1 = (1+5.74)/16 = 6.74/16 ~ 0.42.
    # p2 = (1-5.74)/16 < 0.
    # So optimal p for 2-point {0,1} is approx 0.42.
    # Value: f(0.42) ~ 0.42 - 3(0.074) + 2(0.031) ~ 0.42 - 0.22 + 0.06 = 0.26.
    # Uniform {0,1} (p=0.5) gives: 0.5 * 0.25 * 2 = 0.25.
    # So p=0.42 is slightly better.
    
    for p in [0.3, 0.4, 0.42, 0.45, 0.5, 0.6]:
        support = np.array([0.0, 1.0])
        weights = np.array([1-p, p])
        P = _compute_P(support, weights)
        update_best(support, weights, P)

    # 4. Try 3-point {0, a, 1}
    for a in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for p0 in [0.2, 0.3, 0.4]:
            for p1 in [0.2, 0.3, 0.4]:
                p2 = 1.0 - p0 - p1
                if p2 < 0: continue
                support = np.array([0.0, a, 1.0])
                weights = np.array([p0, p1, p2])
                P = _compute_P(support, weights)
                update_best(support, weights, P)

    return support_best, weights_best


# EVOLVE-BLOCK-END
