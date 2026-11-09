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
    p = weights / w_sum
    
    pair_sums = support[:, None] + support[None, :]
    pair_probs = p[:, None] * p[None, :]
    thresholds = 2.0 * support[None, :] - support[:, None]
    
    total = 0.0
    for k in range(n):
        for l in range(n):
            thr = thresholds[k, l]
            mask = pair_sums < thr
            inner = np.sum(pair_probs * mask)
            total += p[k] * p[l] * inner
    
    return total


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 620.0
    
    # Baseline
    n_base = 32
    support_base = np.arange(n_base, dtype=np.float64)
    weights_base = np.ones(n_base, dtype=np.float64) / n_base
    best_support = support_base.copy()
    best_weights = weights_base.copy()
    best_P = _compute_P(best_support, best_weights)
    
    # Try smaller structured distributions
    candidates = []
    
    # Uniform on {0,...,m-1} for various m
    for m in [4, 8, 16, 32, 64]:
        s = np.arange(m, dtype=np.float64)
        w = np.ones(m, dtype=np.float64) / m
        candidates.append((s, w))
    
    # Two-point
    for b in [1.0, 2.0, 5.0, 10.0]:
        for p_val in np.linspace(0.1, 0.9, 9):
            s = np.array([0.0, b], dtype=np.float64)
            w = np.array([p_val, 1-p_val], dtype=np.float64)
            candidates.append((s, w))
    
    # Three-point
    for a in [1.0, 2.0]:
        for b in [5.0, 10.0]:
            for p1 in [0.2, 0.3, 0.5]:
                for p2 in [0.2, 0.3]:
                    p3 = 1.0 - p1 - p2
                    if p3 <= 0.05:
                        continue
                    s = np.array([0.0, a, b], dtype=np.float64)
                    w = np.array([p1, p2, p3], dtype=np.float64)
                    candidates.append((s, w))
    
    for s, w in candidates:
        if time.time() > deadline:
            break
        P_val = _compute_P(s, w)
        if P_val > best_P:
            best_P = P_val
            best_support = s.copy()
            best_weights = w.copy()
    
    # Coordinate ascent on weights
    if time.time() < deadline:
        s = best_support.copy()
        w = best_weights.copy()
        n = len(s)
        
        for iteration in range(30):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                old_wi = w[i]
                for factor in [0.5, 0.75, 1.25, 1.5, 2.0, 0.25, 4.0]:
                    w_new = w.copy()
                    w_new[i] = old_wi * factor
                    if np.any(w_new < -1e-9):
                        continue
                    P_new = _compute_P(s, w_new)
                    if P_new > best_P + 1e-12:
                        best_P = P_new
                        best_weights = w_new.copy()
                        best_support = s.copy()
                        w = w_new.copy()
                        improved = True
                        break
            if not improved:
                break
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
