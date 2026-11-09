# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations_with_replacement

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 600s limit, keep margin

    def compute_P(support, weights):
        """
        Compute P[X1+X2+X3 < 2*X4] efficiently using convolution-like approach.
        For small n (<=10), use direct enumeration. For larger n, use smarter method.
        """
        n = len(support)
        if n <= 1:
            return 0.0
        
        # Build all pairwise sums s_i + s_j with weights w_i * w_j
        # Then for each X4 = support[k], we need P[X1+X2+X3 < 2*support[k]]
        # P[X1+X2+X3 < t] = sum_{i,j,l: s_i+s_j+s_l < t} w_i*w_j*w_l
        
        # For efficiency with small n, do direct computation
        # Precompute all triple sums
        total_P = 0.0
        for k in range(n):
            w4 = weights[k]
            if w4 <= 0:
                continue
            threshold = 2.0 * support[k]
            # Compute P[X1+X2+X3 < threshold]
            p_sum = 0.0
            for i in range(n):
                wi = weights[i]
                if wi <= 0:
                    continue
                for j in range(n):
                    wj = weights[j]
                    if wj <= 0:
                        continue
                    for l in range(n):
                        wl = weights[l]
                        if wl <= 0:
                            continue
                        s = support[i] + support[j] + support[l]
                        if s < threshold:
                            p_sum += wi * wj * wl
            total_P += w4 * p_sum
        return total_P

    def compute_P_fast(support, weights):
        """
        Faster computation using numpy vectorization for the inner loops.
        """
        n = len(support)
        if n <= 1:
            return 0.0
        
        # Create all pairwise sums: support[i] + support[j] for all i,j
        # shape (n, n)
        pair_sums = support[:, None] + support[None, :]  # (n, n)
        pair_weights = weights[:, None] * weights[None, :]  # (n, n)
        
        total_P = 0.0
        for k in range(n):
            w4 = weights[k]
            if w4 <= 0:
                continue
            threshold = 2.0 * support[k]
            # Need P[X1+X2+X3 < threshold]
            # X1+X2 has distribution given by pair_sums, pair_weights
            # X3 has distribution given by support, weights
            # P[X1+X2+X3 < t] = sum over (i,j,l) where pair_sums[i,j] + support[l] < t
            # of pair_weights[i,j] * weights[l]
            
            # For each (i,j) pair, the condition is support[l] < threshold - pair_sums[i,j]
            # So for each (i,j), count sum of weights[l] where support[l] < threshold - pair_sums[i,j]
            
            # Vectorized: for each (i,j), compute the cumulative weight of support values < threshold - pair_sums[i,j]
            # Since support is sorted, we can use searchsorted
            
            # Compute the effective thresholds for each (i,j)
            eff_thresholds = threshold - pair_sums  # (n, n)
            
            # For each effective threshold, find how many support values are < it
            # support is sorted, so use searchsorted
            # We want sum of weights[l] for support[l] < eff_thresholds[i,j]
            
            # Use searchsorted to find the index
            indices = np.searchsorted(support, eff_thresholds, side='left')  # (n, n)
            
            # Now we need cumulative sum of weights up to each index
            cum_weights = np.concatenate([[0.0], np.cumsum(weights)])  # length n+1
            
            # For each (i,j), the probability mass of X3 < eff_thresholds[i,j] is cum_weights[indices[i,j]]
            p_x3 = cum_weights[indices]  # (n, n)
            
            # P[X1+X2+X3 < threshold] = sum over (i,j) of pair_weights[i,j] * p_x3[i,j]
            p_sum = np.sum(pair_weights * p_x3)
            total_P += w4 * p_sum
        
        return total_P

    # Start with 2-point distribution
    def two_point_P(p):
        """P[X1+X2+X3 < 2*X4] for 2-point {0, 1} with mass p on 0."""
        # X4 = 0 with prob p: need X1+X2+X3 < 0 -> impossible
        # X4 = 1 with prob (1-p): need X1+X2+X3 < 2
        #   Sum < 2 means at most one of X1,X2,X3 is 1
        #   P[at most one 1] = p^3 + 3*p^2*(1-p)
        return (1.0 - p) * (p**3 + 3.0 * p**2 * (1.0 - p))
    
    # Golden section search for optimal p in 2-point case
    a, b = 0.01, 0.99
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = two_point_P(c)
    fd = two_point_P(d)
    
    max_iters = 200
    for i in range(max_iters):
        if time.time() > deadline - 60.0:
            break
        if fc < fd:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = two_point_P(d)
        else:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = two_point_P(c)
    
    p_opt = (a + b) / 2.0
    P_opt = two_point_P(p_opt)
    
    best_P = P_opt
    best_support = np.array([0.0, 1.0])
    best_weights = np.array([p_opt, 1.0 - p_opt])
    
    # Try 3-point distributions: {0, a, 1} with masses (m0, ma, m1)
    # Use a smarter search
    
    def three_point_P(a, m0, ma):
        m1 = 1.0 - m0 - ma
        if m1 < -1e-12 or m0 < -1e-12 or ma < -1e-12:
            return 0.0
        m1 = max(m1, 0.0)
        m0 = max(m0, 0.0)
        ma = max(ma, 0.0)
        
        points = np.array([0.0, a, 1.0])
        weights = np.array([m0, ma, m1])
        return compute_P_fast(points, weights)
    
    # Coarse grid search over a, then refine
    a_vals = np.linspace(0.05, 0.95, 15)
    
    for a in a_vals:
        if time.time() > deadline - 120.0:
            break
        # For this a, do a 2D search over (m0, ma)
        n_grid = 12
        for u in np.linspace(0.02, 0.98, n_grid):
            for v in np.linspace(0.02, 0.98, n_grid):
                m0 = u
                ma = v * (1.0 - u)
                m1 = (1.0 - v) * (1.0 - u)
                if m1 < 0.01 or m0 < 0.01 or ma < 0.01:
                    continue
                P = three_point_P(a, m0, ma)
                if P > best_P:
                    best_P = P
                    best_support = np.array([0.0, a, 1.0])
                    best_weights = np.array([m0, ma, m1])
    
    # Try 4-point distributions: {0, a, b, 1}
    def four_point_P(a, b, m0, ma, mb):
        m1 = 1.0 - m0 - ma - mb
        if m1 < -1e-12 or m0 < -1e-12 or ma < -1e-12 or mb < -1e-12:
            return 0.0
        m1 = max(m1, 0.0)
        m0 = max(m0, 0.0)
        ma = max(ma, 0.0)
        mb = max(mb, 0.0)
        
        points = np.array([0.0, a, b, 1.0])
        weights = np.array([m0, ma, mb, m1])
        return compute_P_fast(points, weights)
    
    # Very coarse 4-point search
    if time.time() < deadline - 180.0:
        a_vals4 = np.linspace(0.1, 0.4, 5)
        b_vals4 = np.linspace(0.6, 0.9, 5)
        for a in a_vals4:
            for b in b_vals4:
                if b <= a:
                    continue
                for m0 in np.linspace(0.05, 0.4, 4):
                    for ma in np.linspace(0.05, 0.3, 4):
                        for mb in np.linspace(0.05, 0.3, 4):
                            m1 = 1.0 - m0 - ma - mb
                            if m1 < 0.05:
                                continue
                            P = four_point_P(a, b, m0, ma, mb)
                            if P > best_P:
                                best_P = P
                                best_support = np.array([0.0, a, b, 1.0])
                                best_weights = np.array([m0, ma, mb, m1])
    
    # Final refinement: local search on weights
    if time.time() < deadline - 30.0:
        support = best_support.copy()
        weights = best_weights.copy()
        n = len(support)
        
        best_local_P = best_P
        best_local_weights = weights.copy()
        
        # Coordinate descent on weights
        step = 0.01
        for iteration in range(30):
            if time.time() > deadline - 15.0:
                break
            improved = False
            for idx in range(n):
                # Try increasing and decreasing weights[idx]
                for delta in [-step, step]:
                    perturbed = best_local_weights.copy()
                    perturbed[idx] += delta
                    if perturbed[idx] < 0:
                        continue
                    # Renormalize
                    perturbed = np.maximum(perturbed, 0.0)
                    s = perturbed.sum()
                    if s <= 1e-9:
                        continue
                    perturbed /= s
                    P = compute_P_fast(support, perturbed)
                    if P > best_local_P:
                        best_local_P = P
                        best_local_weights = perturbed.copy()
                        improved = True
            if not improved:
                break
        
        if best_local_P > best_P:
            best_P = best_local_P
            best_weights = best_local_weights
    
    support = best_support.astype(np.float64)
    weights = best_weights.astype(np.float64)
    
    # Ensure validity
    assert len(support) == len(weights)
    assert np.all(support >= -1e-9)
    assert np.all(weights >= -1e-9)
    assert np.sum(weights) > 1e-9
    
    return support, weights

# EVOLVE-BLOCK-END
