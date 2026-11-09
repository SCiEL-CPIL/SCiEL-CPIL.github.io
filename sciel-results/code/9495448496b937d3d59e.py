# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def _entropy_from_values(values, weights):
    """Compute Shannon entropy in bits given values and their weights."""
    d = defaultdict(float)
    for v, w in zip(values, weights):
        d[v] += w
    total = sum(d.values())
    if total <= 1e-15:
        return 0.0
    probs = [w / total for w in d.values() if w > 1e-15]
    if not probs:
        return 0.0
    return -sum(p * np.log2(p) for p in probs)

def _compute_ratio(P):
    """Compute D/M ratio for a 10x10 distribution P[y,x]."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0
    P = P / s
    
    y_idx, x_idx = np.indices((10, 10))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    w = P.flatten()
    
    # D = H of (x - y)
    d_vals = x_flat - y_flat
    D = _entropy_from_values(d_vals, w)
    
    # M = max of H of x, x+y, x+2y
    h0 = _entropy_from_values(x_flat, w)
    h1 = _entropy_from_values(x_flat + y_flat, w)
    h2 = _entropy_from_values(x_flat + 2 * y_flat, w)
    M = max(h0, h1, h2)
    
    if M <= 1e-15:
        return 0.0
    return D / M

def _compute_ratio_vectorized(P):
    """Faster vectorized computation of D/M ratio."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0
    P = P / s
    
    y_idx, x_idx = np.indices((10, 10))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    w = P.flatten()
    
    def _entropy_vec(vals, weights):
        # Use np.unique for fast pooling
        unique_vals, inverse = np.unique(vals, return_inverse=True)
        pooled = np.zeros(len(unique_vals))
        np.add.at(pooled, inverse, weights)
        total = pooled.sum()
        if total <= 1e-15:
            return 0.0
        probs = pooled[pooled > 1e-15] / total
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    D = _entropy_vec(x_flat - y_flat, w)
    h0 = _entropy_vec(x_flat, w)
    h1 = _entropy_vec(x_flat + y_flat, w)
    h2 = _entropy_vec(x_flat + 2 * y_flat, w)
    M = max(h0, h1, h2)
    
    if M <= 1e-15:
        return 0.0
    return D / M

def _optimize_P():
    """
    Optimize P to maximize D/M.
    
    Structural change: Instead of generic Nelder-Mead on log-probabilities,
    use a structured approach based on the arithmetic structure of the problem.
    
    Key insight: The ratio D/M is maximized when the distribution of (x-y)
    is as uniform as possible (high D) while at least one of the projections
    x, x+y, x+2y has low entropy (low M).
    
    Strategy:
    1. Try distributions concentrated on lines/arithmetic progressions
    2. Use coordinate ascent on the 100-cell grid
    3. Use simulated annealing with structured perturbations
    """
    deadline = time.time() + 590  # safety margin
    
    n = 10
    best_P = np.ones((n, n)) / (n * n)
    best_score = _compute_ratio_vectorized(best_P)
    
    # Precompute the value matrices for fast evaluation
    y_idx, x_idx = np.indices((n, n))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    
    # D values: x - y, ranges from -9 to 9 (19 values)
    d_vals = x_flat - y_flat
    # H0 values: x, ranges from 0 to 9 (10 values)
    h0_vals = x_flat
    # H1 values: x + y, ranges from 0 to 18 (19 values)
    h1_vals = x_flat + y_flat
    # H2 values: x + 2y, ranges from 0 to 27 (28 values)
    h2_vals = x_flat + 2 * y_flat
    
    def _entropy_from_pooled(pooled):
        total = pooled.sum()
        if total <= 1e-15:
            return 0.0
        probs = pooled[pooled > 1e-15] / total
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    def _ratio_from_P(P_flat):
        """Compute ratio from flat array, using precomputed value indices."""
        P_flat = np.clip(P_flat, 0, None)
        s = P_flat.sum()
        if s <= 1e-15:
            return 0.0
        P_flat = P_flat / s
        
        # Pool by value using np.bincount for speed
        # d_vals ranges -9..9, offset by 9
        d_pooled = np.bincount(d_vals + 9, weights=P_flat, minlength=19)
        D = _entropy_from_pooled(d_pooled)
        
        h0_pooled = np.bincount(h0_vals, weights=P_flat, minlength=10)
        h0 = _entropy_from_pooled(h0_pooled)
        
        h1_pooled = np.bincount(h1_vals, weights=P_flat, minlength=19)
        h1 = _entropy_from_pooled(h1_pooled)
        
        h2_pooled = np.bincount(h2_vals, weights=P_flat, minlength=28)
        h2 = _entropy_from_pooled(h2_pooled)
        
        M = max(h0, h1, h2)
        if M <= 1e-15:
            return 0.0
        return D / M
    
    best_score = _ratio_from_P(best_P.flatten())
    
    # Phase 1: Try structured distributions
    # Concentrate on diagonals (x - y = const) to maximize D
    # but this makes H0, H1, H2 potentially high too
    
    # Try: uniform on a single diagonal x - y = c
    for c in range(-9, 10):
        P_try = np.zeros((n, n))
        for y in range(n):
            x = y + c
            if 0 <= x < n:
                P_try[y, x] = 1.0
        s = P_try.sum()
        if s > 0:
            P_try /= s
            score = _ratio_from_P(P_try.flatten())
            if score > best_score:
                best_score = score
                best_P = P_try.copy()
    
    # Try: uniform on anti-diagonals (x + y = const) to minimize H1
    for c in range(0, 19):
        P_try = np.zeros((n, n))
        for y in range(n):
            x = c - y
            if 0 <= x < n:
                P_try[y, x] = 1.0
        s = P_try.sum()
        if s > 0:
            P_try /= s
            score = _ratio_from_P(P_try.flatten())
            if score > best_score:
                best_score = score
                best_P = P_try.copy()
    
    # Try: uniform on columns (x = const) to minimize H0
    for c in range(n):
        P_try = np.zeros((n, n))
        P_try[:, c] = 1.0
        P_try /= n
        score = _ratio_from_P(P_try.flatten())
        if score > best_score:
            best_score = score
            best_P = P_try.copy()
    
    # Try: uniform on rows (y = const)
    for c in range(n):
        P_try = np.zeros((n, n))
        P_try[c, :] = 1.0
        P_try /= n
        score = _ratio_from_P(P_try.flatten())
        if score > best_score:
            best_score = score
            best_P = P_try.copy()
    
    # Phase 2: Coordinate ascent with random restarts
    # Each step: pick a random cell, adjust its weight
    rng = np.random.RandomState(42)
    
    P_current = best_P.flatten().copy()
    T = 0.1  # temperature for simulated annealing
    T_min = 0.001
    cooling = 0.999
    
    for iteration in range(200000):
        if time.time() > deadline:
            break
        
        # Pick a random cell to modify
        i = rng.randint(0, 100)
        
        # Try increasing or decreasing the weight
        delta = 0.05 * rng.random()
        if rng.random() < 0.5:
            delta = -delta
        
        P_new = P_current.copy()
        P_new[i] = max(0, P_new[i] + delta)
        
        # Renormalize
        s = P_new.sum()
        if s <= 1e-15:
            continue
        P_new /= s
        
        score_new = _ratio_from_P(P_new)
        score_cur = _ratio_from_P(P_current)
        
        # Accept if better or with probability exp((score_new - score_cur) / T)
        if score_new > score_cur:
            P_current = P_new
            if score_new > best_score:
                best_score = score_new
                best_P = P_new.reshape(10, 10).copy()
        else:
            if T > T_min:
                prob = np.exp((score_new - score_cur) / T)
                if rng.random() < prob:
                    P_current = P_new
        
        T *= cooling
    
    # Phase 3: Local search - try small perturbations around best
    P_work = best_P.flatten().copy()
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        i = rng.randint(0, 100)
        j = rng.randint(0, 100)
        if i == j:
            continue
        
        # Move mass from j to i
        amount = 0.02 * rng.random()
        if P_work[j] < amount:
            amount = P_work[j]
        
        P_new = P_work.copy()
        P_new[i] += amount
        P_new[j] -= amount
        P_new = np.clip(P_new, 0, None)
        s = P_new.sum()
        if s <= 1e-15:
            continue
        P_new /= s
        
        score_new = _ratio_from_P(P_new)
        if score_new > _ratio_from_P(P_work):
            P_work = P_new
            if score_new > best_score:
                best_score = score_new
                best_P = P_work.reshape(10, 10).copy()
    
    # Phase 4: Try more structured patterns
    # Checkerboard patterns
    for offset in [0, 1]:
        P_try = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if (x + y + offset) % 2 == 0:
                    P_try[y, x] = 1.0
        s = P_try.sum()
        if s > 0:
            P_try /= s
            score = _ratio_from_P(P_try.flatten())
            if score > best_score:
                best_score = score
                best_P = P_try.copy()
    
    # Try: concentrate on a 2D pattern like x = f(y)
    # e.g., x = (2*y) % 10
    for a in range(1, 10):
        for b in range(10):
            P_try = np.zeros((n, n))
            for y in range(n):
                x = (a * y + b) % n
                P_try[y, x] = 1.0
            P_try /= n
            score = _ratio_from_P(P_try.flatten())
            if score > best_score:
                best_score = score
                best_P = P_try.copy()
    
    return best_P

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    start_time = time.time()
    
    P = _optimize_P()
    
    # Final safety checks
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-9:
        # Fallback to uniform
        P = np.ones((10, 10)) / 100.0
    else:
        P = P / s
    
    # Ensure finite
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    
    return P

# EVOLVE-BLOCK-END
