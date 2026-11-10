# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly."""
    n = len(support)
    w = weights / weights.sum()
    # Build all pairwise sums s_i + s_j
    # P = sum_{k,l} w_k w_l * sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i w_j
    # For efficiency, precompute pairwise sums
    pair_sums = support[:, None] + support[None, :]  # n x n
    pair_weights = w[:, None] * w[None, :]  # n x n
    # For each (k, l), we need sum of pair_weights where pair_sums < 2*s_l - s_k
    # threshold[k,l] = 2*s_l - s_k
    thresholds = 2.0 * support[None, :] - support[:, None]  # n x n, thresholds[k,l]
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            if w[k] * w[l] < 1e-15:
                continue
            thr = thresholds[k, l]
            mask = pair_sums < thr
            P += w[k] * w[l] * pair_weights[mask].sum()
    return P


def _optimize_2point():
    """Optimize over 2-point distributions: p at 0, (1-p) at M."""
    best_p = 0.5
    best_M = 1.0
    best_val = 0.0
    
    # For 2-point, P = (1-p) * p^2 * (3-2p) if we use 0 and M
    # But M doesn't matter as long as M > 0 (for the strict inequality 2M < 2M fails)
    # Actually M must be > 0. Let's just optimize p.
    # g(p) = (1-p)*p^2*(3-2p)
    # Max at p = (15-sqrt(33))/16
    p_opt = (15 - np.sqrt(33)) / 16.0
    val = (1 - p_opt) * p_opt**2 * (3 - 2 * p_opt)
    
    return p_opt, 1.0, val


def _optimize_3point():
    """Try 3-point: p0 at 0, p1 at 1, p2 at M."""
    # This gets complex. Let's use a simple grid search over proportions.
    best_val = 0.0
    best_params = (0.5, 0.3, 0.2, 10.0)
    
    # Grid search over p0, p1, M (p2 = 1-p0-p1)
    for p0 in np.linspace(0.3, 0.8, 30):
        for p1 in np.linspace(0.0, 0.5, 20):
            p2 = 1.0 - p0 - p1
            if p2 < 0.01:
                continue
            for M in [2.0, 5.0, 10.0, 20.0]:
                support = np.array([0.0, 1.0, M])
                weights = np.array([p0, p1, p2])
                val = _compute_P(support, weights)
                if val > best_val:
                    best_val = val
                    best_params = (p0, p1, p2, M)
    
    return best_params, best_val


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit, keep safety margin
    
    # Start with the 2-point optimal
    p_opt = (15 - np.sqrt(33)) / 16.0
    support_2 = np.array([0.0, 1.0], dtype=np.float64)
    weights_2 = np.array([p_opt, 1.0 - p_opt], dtype=np.float64)
    best_val = _compute_P(support_2, weights_2)
    best_support = support_2.copy()
    best_weights = weights_2.copy()
    
    # Try 3-point optimization
    if time.time() < deadline - 10.0:
        params, val3 = _optimize_3point()
        if val3 > best_val:
            p0, p1, p2, M = params
            best_support = np.array([0.0, 1.0, M], dtype=np.float64)
            best_weights = np.array([p0, p1, p2], dtype=np.float64)
            best_val = val3
    
    # Try a finer 4-point search
    if time.time() < deadline - 10.0:
        # Try distributions with points at 0, a, b, c
        for a in [0.5, 1.0, 2.0]:
            for b in [1.5, 3.0, 5.0]:
                for c in [5.0, 10.0, 20.0]:
                    if not (0 <= a < b < c):
                        continue
                    # Optimize weights via simple local search
                    w0, w1, w2, w3 = 0.4, 0.2, 0.2, 0.2
                    support = np.array([0.0, a, b, c], dtype=np.float64)
                    for _ in range(50):
                        if time.time() > deadline - 5.0:
                            break
                        w = np.array([w0, w1, w2, w3])
                        w = w / w.sum()
                        val = _compute_P(support, w)
                        if val > best_val:
                            best_val = val
                            best_support = support.copy()
                            best_weights = w.copy()
                        # Perturb
                        idx = np.random.randint(4)
                        delta = np.random.uniform(-0.1, 0.1)
                        w[idx] += delta
                        if w[idx] < 0:
                            w[idx] = 0
                        w = w / w.sum()
                        w0, w1, w2, w3 = w
    
    # Validate and return
    support = best_support.astype(np.float64).flatten()
    weights = best_weights.astype(np.float64).flatten()
    
    # Ensure validity
    if len(support) != len(weights) or len(support) < 1:
        support = np.array([0.0, 1.0])
        weights = np.array([p_opt, 1.0 - p_opt])
    
    if np.any(np.isnan(support)) or np.any(np.isinf(support)):
        support = np.array([0.0, 1.0])
        weights = np.array([p_opt, 1.0 - p_opt])
    
    if np.any(support < -1e-9):
        support = np.clip(support, 0, None)
    
    if np.any(weights < -1e-9):
        weights = np.clip(weights, 0, None)
    
    if weights.sum() <= 1e-9:
        weights = np.array([p_opt, 1.0 - p_opt])
        support = np.array([0.0, 1.0])
    
    return support, weights

# EVOLVE-BLOCK-END
