# EVOLVE-BLOCK-START
import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 600s limit, keep margin

    def compute_P(support, weights):
        """Compute P[X1+X2+X3 < 2*X4] exactly for a discrete measure."""
        n = len(support)
        # All pairwise sums s_i + s_j
        # For each (k, l): weight w_k * w_l * sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i * w_j
        # Vectorized: build all n^2 pair sums and their weights
        S = support[:, None] + support[None, :]  # n x n
        W = weights[:, None] * weights[None, :]  # n x n
        # For each k, l: threshold = 2*s_l - s_k
        # P = sum_{k,l} w_k * w_l * sum_{i,j: S[i,j] < 2*s_l - s_k} W[i,j]
        # Reshape to use broadcasting
        # S is n x n, we need for each (k,l) pair, sum of W where S < 2*s_l - s_k
        # 2*s_l - s_k: shape (n, n) where first index is k, second is l
        thresholds = 2.0 * support[None, :] - support[:, None]  # n x n, [k, l]
        # For each (k,l), we need sum of W[i,j] where S[i,j] < thresholds[k,l]
        # This is a 4D problem. For small n, do it with loops over k,l
        total = 0.0
        for k in range(n):
            for l in range(n):
                thr = 2.0 * support[l] - support[k]
                mask = S < thr
                inner_sum = np.sum(W * mask)
                total += weights[k] * weights[l] * inner_sum
        return total

    # 2-point distribution: {0, L} with mass p on 0 and (1-p) on L
    def two_point_P(p):
        """P[X1+X2+X3 < 2*X4] for 2-point {0, L} with mass p on 0."""
        return (1.0 - p) * (p**3 + 3.0 * p**2 * (1.0 - p))
    
    # Golden section search for optimal p
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
    
    # 3-point grid search: {0, a, 1} with masses (m0, ma, m1)
    # Use vectorized compute_P for speed
    
    a_vals = np.linspace(0.05, 0.95, 20)
    m0_vals = np.linspace(0.05, 0.85, 15)
    
    for a in a_vals:
        if time.time() > deadline - 60.0:
            break
        for m0 in m0_vals:
            ma_max = min(0.85, 1.0 - m0 - 0.05)
            if ma_max <= 0.01:
                continue
            for ma in np.linspace(0.05, ma_max, 12):
                m1 = 1.0 - m0 - ma
                if m1 < 0.01:
                    continue
                sup = np.array([0.0, a, 1.0])
                wts = np.array([m0, ma, m1])
                P = compute_P(sup, wts)
                if P > best_P:
                    best_P = P
                    best_support = sup.copy()
                    best_weights = wts.copy()
    
    # 4-point grid search: {0, a, b, 1} with masses (m0, ma, mb, m1)
    # Coarse search over a, b, m0, ma, mb with m1 = 1-m0-ma-mb
    if time.time() < deadline - 120.0:
        a_vals4 = np.linspace(0.1, 0.4, 5)
        b_vals4 = np.linspace(0.5, 0.9, 5)
        m0_vals4 = np.linspace(0.1, 0.5, 4)
        
        for a in a_vals4:
            if time.time() > deadline - 60.0:
                break
            for b in b_vals4:
                if time.time() > deadline - 60.0:
                    break
                for m0 in m0_vals4:
                    remaining = 1.0 - m0
                    if remaining < 0.1:
                        continue
                    # ma + mb + m1 = remaining, all >= 0.05
                    for ma in np.linspace(0.05, remaining - 0.1, 4):
                        if ma > remaining - 0.1:
                            continue
                        for mb in np.linspace(0.05, remaining - ma - 0.05, 4):
                            if mb > remaining - ma - 0.05:
                                continue
                            m1 = remaining - ma - mb
                            if m1 < 0.05:
                                continue
                            sup = np.array([0.0, a, b, 1.0])
                            wts = np.array([m0, ma, mb, m1])
                            P = compute_P(sup, wts)
                            if P > best_P:
                                best_P = P
                                best_support = sup.copy()
                                best_weights = wts.copy()
    
    # Local refinement using coordinate descent on the best candidate
    if time.time() < deadline - 30.0:
        n = len(best_support)
        support = best_support.copy()
        weights = best_weights.copy()
        
        # Coordinate descent: optimize each support point and weight
        # Weights: use simplex constraint, so optimize ratios
        # Support: keep non-negative
        
        for iteration in range(50):
            if time.time() > deadline - 30.0:
                break
            improved = False
            
            # Optimize support points (keep first at 0, last at 1 if n > 1)
            for i in range(1, n):
                if i == n - 1:
                    continue  # keep last point at 1
                old_s = support[i]
                # Try small perturbations
                for delta in [0.05, -0.05, 0.02, -0.02, 0.01, -0.01]:
                    new_s = old_s + delta
                    if new_s < 0.0 or new_s > support[-1] if i < n-1 else new_s > 2.0:
                        continue
                    # Ensure ordering
                    if i > 0 and new_s <= support[i-1]:
                        continue
                    if i < n-1 and new_s >= support[i+1]:
                        continue
                    support[i] = new_s
                    P_new = compute_P(support, weights)
                    if P_new > best_P:
                        best_P = P_new
                        improved = True
                    else:
                        support[i] = old_s
                        break
            
            # Optimize weights via simplex: adjust two weights at a time
            for i in range(n):
                for j in range(i+1, n):
                    old_wi, old_wj = weights[i], weights[j]
                    for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
                        # Move frac of min(wi,wj) from smaller to larger
                        if old_wi < old_wj:
                            move = frac * old_wi
                            new_wi = old_wi - move
                            new_wj = old_wj + move
                        else:
                            move = frac * old_wj
                            new_wi = old_wi + move
                            new_wj = old_wj - move
                        if new_wi < 0.01 or new_wj < 0.01:
                            continue
                        weights[i] = new_wi
                        weights[j] = new_wj
                        P_new = compute_P(support, weights)
                        if P_new > best_P:
                            best_P = P_new
                            improved = True
                        else:
                            weights[i] = old_wi
                            weights[j] = old_wj
                            break
            
            if not improved:
                break
        
        best_support = support.copy()
        best_weights = weights.copy()
    
    support = best_support.astype(np.float64)
    weights = best_weights.astype(np.float64)
    
    # Ensure validity
    assert len(support) == len(weights)
    assert np.all(support >= -1e-9)
    assert np.all(weights >= -1e-9)
    assert np.sum(weights) > 1e-9
    
    return support, weights

# EVOLVE-BLOCK-END
