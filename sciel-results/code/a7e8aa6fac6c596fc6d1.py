# EVOLVE-BLOCK-START
import numpy as np
import time

def _entropy_from_pooled(pooled):
    total = pooled.sum()
    if total <= 1e-15:
        return 0.0
    probs = pooled[pooled > 1e-15] / total
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

def _optimize_P():
    """
    Optimize P to maximize D/M.
    """
    deadline = time.time() + 590  # safety margin

    n = 10
    best_P = np.ones((n, n)) / (n * n)

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

    # Precompute bincount indices
    d_idx = d_vals + 9  # 0..18

    def _ratio_from_P(P_flat):
        """Compute ratio from flat array, using precomputed value indices."""
        P_flat = np.clip(P_flat, 0, None)
        s = P_flat.sum()
        if s <= 1e-15:
            return 0.0
        P_flat = P_flat / s

        # Pool by value using np.bincount for speed
        d_pooled = np.bincount(d_idx, weights=P_flat, minlength=19)
        h0_pooled = np.bincount(h0_vals, weights=P_flat, minlength=10)
        h1_pooled = np.bincount(h1_vals, weights=P_flat, minlength=19)
        h2_pooled = np.bincount(h2_vals, weights=P_flat, minlength=28)

        # Compute entropies vectorized
        def _ent(pooled):
            mask = pooled > 1e-15
            if not mask.any():
                return 0.0
            p = pooled[mask]
            return -np.sum(p * np.log2(p))

        D = _ent(d_pooled)
        h0 = _ent(h0_pooled)
        h1 = _ent(h1_pooled)
        h2 = _ent(h2_pooled)

        M = max(h0, h1, h2)
        if M <= 1e-15:
            return 0.0
        return D / M

    best_score = _ratio_from_P(best_P.flatten())

    # Phase 1: Try structured distributions
    # Concentrate on diagonals (x - y = const) to maximize D
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

    # Phase 2: Log-space hill climbing (targets H2 minimization via concentrating mass)
    rng = np.random.RandomState(42)

    # Initialize in log-space: start from best_P, convert to log with small floor
    log_P = np.log(np.maximum(best_P.flatten(), 1e-10))
    log_T = 0.5  # temperature in log-space
    log_T_min = 0.01
    log_cooling = 0.9995

    for iteration in range(400000):
        if time.time() > deadline:
            break

        # Pick a random cell to modify in log-space
        i = rng.randint(0, 100)
        delta = 0.3 * rng.randn()
        if rng.random() < 0.3:
            delta *= 2.0  # occasional larger step

        log_new = log_P.copy()
        log_new[i] += delta

        # Convert back to probability space
        P_new = np.exp(log_new)
        P_new = np.clip(P_new, 0, None)
        s = P_new.sum()
        if s <= 1e-15:
            continue
        P_new /= s

        score_new = _ratio_from_P(P_new)

        # Also evaluate current for comparison (cache it)
        P_cur = np.exp(log_P)
        P_cur = np.clip(P_cur, 0, None)
        s_cur = P_cur.sum()
        if s_cur <= 1e-15:
            continue
        P_cur /= s_cur
        score_cur = _ratio_from_P(P_cur)

        if score_new > score_cur:
            log_P = log_new
            if score_new > best_score:
                best_score = score_new
                best_P = P_new.reshape(10, 10).copy()
        else:
            if log_T > log_T_min:
                prob = np.exp((score_new - score_cur) / log_T)
                if rng.random() < prob:
                    log_P = log_new

        log_T *= log_cooling

    # Phase 3: Coordinate ascent with random restarts (original, now faster)
    P_current = best_P.flatten().copy()
    T = 0.1  # temperature for simulated annealing
    T_min = 0.001
    cooling = 0.999

    for iteration in range(300000):
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

    # Phase 4: Local search - try small perturbations around best
    P_work = best_P.flatten().copy()
    for iteration in range(100000):
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

    # Phase 5: Try more structured patterns
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

    # Phase 6: Final polish - fine-grained perturbations
    P_polish = best_P.flatten().copy()
    for iteration in range(50000):
        if time.time() > deadline:
            break

        i = rng.randint(0, 100)
        j = rng.randint(0, 100)
        if i == j:
            continue

        # Very small move
        amount = 0.005 * rng.random()
        if P_polish[j] < amount:
            amount = P_polish[j]

        P_new = P_polish.copy()
        P_new[i] += amount
        P_new[j] -= amount
        P_new = np.clip(P_new, 0, None)
        s = P_new.sum()
        if s <= 1e-15:
            continue
        P_new /= s

        score_new = _ratio_from_P(P_new)
        if score_new > _ratio_from_P(P_polish):
            P_polish = P_new
            if score_new > best_score:
                best_score = score_new
                best_P = P_polish.reshape(10, 10).copy()

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
