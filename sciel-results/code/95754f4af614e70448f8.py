# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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
    Optimize P to maximize D/M using gradient-based L-BFGS-B with multiple restarts.
    """
    deadline = time.time() + 590  # safety margin

    n = 10
    best_P = np.ones((n, n)) / (n * n)
    best_score = 0.0

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

    def _compute_ratio(P_flat):
        """Compute ratio D/M from a flat probability vector."""
        P_flat = np.clip(P_flat, 0, None)
        s = P_flat.sum()
        if s <= 1e-15:
            return 0.0
        P_flat = P_flat / s

        d_pooled = np.bincount(d_idx, weights=P_flat, minlength=19)
        h0_pooled = np.bincount(h0_vals, weights=P_flat, minlength=10)
        h1_pooled = np.bincount(h1_vals, weights=P_flat, minlength=19)
        h2_pooled = np.bincount(h2_vals, weights=P_flat, minlength=28)

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

    def _soft_ratio(P_flat, beta=50.0):
        """
        Smooth surrogate: replace max(h0,h1,h2) with soft-max via log-sum-exp.
        This makes the objective differentiable.
        """
        P_flat = np.clip(P_flat, 0, None)
        s = P_flat.sum()
        if s <= 1e-15:
            return 0.0
        P_flat = P_flat / s

        d_pooled = np.bincount(d_idx, weights=P_flat, minlength=19)
        h0_pooled = np.bincount(h0_vals, weights=P_flat, minlength=10)
        h1_pooled = np.bincount(h1_vals, weights=P_flat, minlength=19)
        h2_pooled = np.bincount(h2_vals, weights=P_flat, minlength=28)

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

        # Soft max: (1/beta) * log(sum(exp(beta * h_i)))
        vals = np.array([h0, h1, h2])
        max_val = np.max(vals)
        if max_val <= 1e-15:
            return 0.0
        soft_max = max_val + np.log(np.sum(np.exp(beta * (vals - max_val)))) / beta

        return D / soft_max

    # Phase 1: Try structured distributions to get good initial points
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
            score = _compute_ratio(P_try.flatten())
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
            score = _compute_ratio(P_try.flatten())
            if score > best_score:
                best_score = score
                best_P = P_try.copy()

    # Try: uniform on columns (x = const) to minimize H0
    for c in range(n):
        P_try = np.zeros((n, n))
        P_try[:, c] = 1.0
        P_try /= n
        score = _compute_ratio(P_try.flatten())
        if score > best_score:
            best_score = score
            best_P = P_try.copy()

    # Try: uniform on rows (y = const)
    for c in range(n):
        P_try = np.zeros((n, n))
        P_try[c, :] = 1.0
        P_try /= n
        score = _compute_ratio(P_try.flatten())
        if score > best_score:
            best_score = score
            best_P = P_try.copy()

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
            score = _compute_ratio(P_try.flatten())
            if score > best_score:
                best_score = score
                best_P = P_try.copy()

    # Try: concentrate on a 2D pattern like x = f(y)
    for a in range(1, 10):
        for b in range(10):
            P_try = np.zeros((n, n))
            for y in range(n):
                x = (a * y + b) % n
                P_try[y, x] = 1.0
            P_try /= n
            score = _compute_ratio(P_try.flatten())
            if score > best_score:
                best_score = score
                best_P = P_try.copy()

    # Phase 2: L-BFGS-B optimization with log-space parameterization
    # Use log-space to handle non-negativity: P_i = exp(z_i) / sum(exp(z_j))
    # This automatically ensures P_i >= 0 and sum(P) = 1 (softmax-like)
    
    def _soft_ratio_from_log(z):
        """Compute soft ratio from log-space parameters."""
        # Softmax to get probabilities
        z_max = np.max(z)
        exp_z = np.exp(z - z_max)
        P_flat = exp_z / np.sum(exp_z)
        return _soft_ratio(P_flat)

    def _neg_soft_ratio_from_log(z):
        """Negative soft ratio for minimization."""
        return -_soft_ratio_from_log(z)

    rng = np.random.RandomState(42)
    
    # Multiple restarts from different initial points
    initial_points = [best_P.flatten().copy()]
    
    # Add random perturbations around best
    for _ in range(10):
        P_rand = best_P.flatten().copy()
        noise = rng.randn(100) * 0.1
        P_rand += noise
        P_rand = np.clip(P_rand, 1e-6, None)
        P_rand /= P_rand.sum()
        initial_points.append(P_rand)
    
    # Add some random structured starts
    for c in range(-9, 10, 3):
        P_try = np.zeros((n, n))
        for y in range(n):
            x = y + c
            if 0 <= x < n:
                P_try[y, x] = 1.0
        s = P_try.sum()
        if s > 0:
            P_try /= s
            initial_points.append(P_try.flatten().copy())

    # L-BFGS-B optimization
    for init_P in initial_points:
        if time.time() > deadline:
            break
        
        # Convert to log-space
        z0 = np.log(np.maximum(init_P, 1e-10))
        
        # Run L-BFGS-B
        result = minimize(
            _neg_soft_ratio_from_log,
            z0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success:
            z_opt = result.x
            # Convert back to probability space
            z_max = np.max(z_opt)
            exp_z = np.exp(z_opt - z_max)
            P_opt = exp_z / np.sum(exp_z)
            
            # Evaluate with exact (non-soft) ratio
            score = _compute_ratio(P_opt)
            if score > best_score:
                best_score = score
                best_P = P_opt.reshape(10, 10).copy()

    # Phase 3: Fine-tuning with smaller softness parameter (more accurate)
    def _soft_ratio_fine(P_flat, beta=200.0):
        P_flat = np.clip(P_flat, 0, None)
        s = P_flat.sum()
        if s <= 1e-15:
            return 0.0
        P_flat = P_flat / s

        d_pooled = np.bincount(d_idx, weights=P_flat, minlength=19)
        h0_pooled = np.bincount(h0_vals, weights=P_flat, minlength=10)
        h1_pooled = np.bincount(h1_vals, weights=P_flat, minlength=19)
        h2_pooled = np.bincount(h2_vals, weights=P_flat, minlength=28)

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

        vals = np.array([h0, h1, h2])
        max_val = np.max(vals)
        if max_val <= 1e-15:
            return 0.0
        soft_max = max_val + np.log(np.sum(np.exp(beta * (vals - max_val)))) / beta

        return D / soft_max

    def _neg_soft_ratio_fine(z):
        z_max = np.max(z)
        exp_z = np.exp(z - z_max)
        P_flat = exp_z / np.sum(exp_z)
        return -_soft_ratio_fine(P_flat)

    # Fine-tune from best
    z_best = np.log(np.maximum(best_P.flatten(), 1e-10))
    result = minimize(
        _neg_soft_ratio_fine,
        z_best,
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    
    if result.success:
        z_opt = result.x
        z_max = np.max(z_opt)
        exp_z = np.exp(z_opt - z_max)
        P_opt = exp_z / np.sum(exp_z)
        score = _compute_ratio(P_opt)
        if score > best_score:
            best_score = score
            best_P = P_opt.reshape(10, 10).copy()

    # Phase 4: Final local search with small perturbations
    P_work = best_P.flatten().copy()
    for iteration in range(50000):
        if time.time() > deadline:
            break

        i = rng.randint(0, 100)
        j = rng.randint(0, 100)
        if i == j:
            continue

        amount = 0.005 * rng.random()
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

        score_new = _compute_ratio(P_new)
        if score_new > _compute_ratio(P_work):
            P_work = P_new
            if score_new > best_score:
                best_score = score_new
                best_P = P_work.reshape(10, 10).copy()

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
