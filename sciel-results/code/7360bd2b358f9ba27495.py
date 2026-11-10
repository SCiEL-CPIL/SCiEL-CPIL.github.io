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
    Optimize P to maximize D/M using gradient-based L-BFGS-B with multiple restarts.
    """
    deadline = time.time() + 580  # safety margin
    from scipy.optimize import minimize

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

    def _entropy_from_pooled(pooled):
        total = pooled.sum()
        if total <= 1e-15:
            return 0.0
        probs = pooled[pooled > 1e-15] / total
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))

    def _compute_ratio(P_flat):
        """Compute D/M ratio from a probability vector."""
        P_flat = np.clip(P_flat, 0, None)
        s = P_flat.sum()
        if s <= 1e-15:
            return 0.0
        P_flat = P_flat / s

        d_pooled = np.bincount(d_idx, weights=P_flat, minlength=19)
        h0_pooled = np.bincount(h0_vals, weights=P_flat, minlength=10)
        h1_pooled = np.bincount(h1_vals, weights=P_flat, minlength=19)
        h2_pooled = np.bincount(h2_vals, weights=P_flat, minlength=28)

        D = _entropy_from_pooled(d_pooled)
        h0 = _entropy_from_pooled(h0_pooled)
        h1 = _entropy_from_pooled(h1_pooled)
        h2 = _entropy_from_pooled(h2_pooled)

        M = max(h0, h1, h2)
        if M <= 1e-15:
            return 0.0
        return D / M

    def _compute_ratio_components(P_flat):
        """Compute D, h0, h1, h2 and M separately."""
        P_flat = np.clip(P_flat, 0, None)
        s = P_flat.sum()
        if s <= 1e-15:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        P_flat = P_flat / s

        d_pooled = np.bincount(d_idx, weights=P_flat, minlength=19)
        h0_pooled = np.bincount(h0_vals, weights=P_flat, minlength=10)
        h1_pooled = np.bincount(h1_vals, weights=P_flat, minlength=19)
        h2_pooled = np.bincount(h2_vals, weights=P_flat, minlength=28)

        D = _entropy_from_pooled(d_pooled)
        h0 = _entropy_from_pooled(h0_pooled)
        h1 = _entropy_from_pooled(h1_pooled)
        h2 = _entropy_from_pooled(h2_pooled)

        M = max(h0, h1, h2)
        return D, h0, h1, h2, M

    def _soft_ratio(P_flat, beta=200.0):
        """
        Smooth surrogate for D/M using soft-max on the three entropies.
        Higher beta -> closer to true max.
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

        D = _entropy_from_pooled(d_pooled)
        h0 = _entropy_from_pooled(h0_pooled)
        h1 = _entropy_from_pooled(h1_pooled)
        h2 = _entropy_from_pooled(h2_pooled)

        # Soft-max: M_soft = (1/beta) * log(sum(exp(beta * h_i)))
        hs = np.array([h0, h1, h2])
        hs_max = hs.max()
        M_soft = hs_max + np.log(np.sum(np.exp(beta * (hs - hs_max)))) / beta

        if M_soft <= 1e-15:
            return 0.0
        return D / M_soft

    def _objective_neg(theta, beta=200.0):
        """Negative of soft ratio for minimization. theta is in log-space."""
        P = np.exp(theta)
        P = np.clip(P, 0, None)
        s = P.sum()
        if s <= 1e-15:
            return 0.0
        P = P / s
        return -_soft_ratio(P, beta=beta)

    def _objective_neg_true(theta):
        """Negative of true ratio (for final evaluation)."""
        P = np.exp(theta)
        P = np.clip(P, 0, None)
        s = P.sum()
        if s <= 1e-15:
            return 0.0
        P = P / s
        return -_compute_ratio(P)

    # Phase 1: Generate diverse starting points
    rng = np.random.RandomState(42)
    starts = []

    # Uniform
    starts.append(np.ones(100) / 100.0)

    # Diagonals (x - y = const)
    for c in range(-9, 10):
        P_try = np.zeros(100)
        for y in range(n):
            x = y + c
            if 0 <= x < n:
                P_try[y * n + x] = 1.0
        s = P_try.sum()
        if s > 0:
            starts.append(P_try / s)

    # Anti-diagonals (x + y = const)
    for c in range(0, 19):
        P_try = np.zeros(100)
        for y in range(n):
            x = c - y
            if 0 <= x < n:
                P_try[y * n + x] = 1.0
        s = P_try.sum()
        if s > 0:
            starts.append(P_try / s)

    # Columns
    for c in range(n):
        P_try = np.zeros(100)
        P_try[c::n] = 1.0 / n
        starts.append(P_try)

    # Rows
    for c in range(n):
        P_try = np.zeros(100)
        P_try[c * n:(c + 1) * n] = 1.0 / n
        starts.append(P_try)

    # Linear patterns x = (a*y + b) % n
    for a in range(1, 10):
        for b in range(10):
            P_try = np.zeros(100)
            for y in range(n):
                x = (a * y + b) % n
                P_try[y * n + x] = 1.0 / n
            starts.append(P_try)

    # Random sparse starts
    for _ in range(30):
        P_try = rng.rand(100)
        P_try = np.where(P_try > 0.5, P_try, 0.0)
        s = P_try.sum()
        if s > 0:
            starts.append(P_try / s)

    # Evaluate all starts
    for P_start in starts:
        score = _compute_ratio(P_start)
        if score > best_score:
            best_score = score
            best_P = P_start.reshape(10, 10).copy()

    # Phase 2: L-BFGS-B optimization from multiple starts
    # Convert to log-space for unconstrained optimization
    log_floor = 1e-12

    # Sort starts by score, try best ones first
    start_scores = [( _compute_ratio(P), P) for P in starts]
    start_scores.sort(key=lambda x: -x[0])

    # Use top starts + some random ones
    selected_starts = []
    for sc, P in start_scores[:20]:
        selected_starts.append(P)
    # Add a few random ones for diversity
    for _ in range(5):
        P_rand = rng.rand(100)
        P_rand = np.where(P_rand > 0.6, P_rand, 0.0)
        s = P_rand.sum()
        if s > 0:
            selected_starts.append(P_rand / s)

    for P_start in selected_starts:
        if time.time() > deadline:
            break

        # Convert to log-space with floor
        theta0 = np.log(np.maximum(P_start, log_floor))

        # Try multiple beta values for the soft-max
        for beta in [100.0, 300.0, 500.0]:
            if time.time() > deadline:
                break
            try:
                result = minimize(
                    lambda th: _objective_neg(th, beta=beta),
                    theta0,
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
                )
                P_opt = np.exp(result.x)
                P_opt = np.clip(P_opt, 0, None)
                s = P_opt.sum()
                if s <= 1e-15:
                    continue
                P_opt = P_opt / s
                score = _compute_ratio(P_opt)
                if score > best_score:
                    best_score = score
                    best_P = P_opt.reshape(10, 10).copy()
                    theta0 = result.x  # Use best result as next start
            except Exception:
                continue

    # Phase 3: Coordinate descent polish on the true (nonsmooth) objective
    # This handles the max by trying to optimize each entropy as the active max
    if time.time() < deadline - 20:
        P_work = best_P.flatten().copy()
        best_cd_score = best_score
        
        # Try each of the three entropies as the "active" max
        for active_idx in range(3):  # 0=h0, 1=h1, 2=h2
            if time.time() > deadline - 10:
                break
            
            P_cd = best_P.flatten().copy()
            for iteration in range(50000):
                if time.time() > deadline - 10:
                    break
                
                i = rng.randint(0, 100)
                j = rng.randint(0, 100)
                if i == j:
                    continue
                
                # Try multiple step sizes
                found_improve = False
                for step_mult in [1.0, 0.5, 0.25, 0.1]:
                    amount = 0.01 * step_mult * rng.random()
                    if P_cd[j] < amount:
                        amount = P_cd[j] * 0.9
                    
                    P_new = P_cd.copy()
                    P_new[i] += amount
                    P_new[j] -= amount
                    P_new = np.clip(P_new, 0, None)
                    s = P_new.sum()
                    if s <= 1e-15:
                        continue
                    P_new /= s
                    
                    score_new = _compute_ratio(P_new)
                    score_cur = _compute_ratio(P_cd)
                    
                    if score_new > score_cur + 1e-12:
                        P_cd = P_new
                        if score_new > best_cd_score:
                            best_cd_score = score_new
                            best_score = score_new
                            best_P = P_cd.reshape(10, 10).copy()
                        found_improve = True
                        break
                
                if not found_improve:
                    # No improvement, try a smaller perturbation
                    i2 = rng.randint(0, 100)
                    j2 = rng.randint(0, 100)
                    if i2 != j2:
                        amount = 0.001 * rng.random()
                        if P_cd[j2] < amount:
                            amount = P_cd[j2] * 0.9
                        P_new = P_cd.copy()
                        P_new[i2] += amount
                        P_new[j2] -= amount
                        P_new = np.clip(P_new, 0, None)
                        s = P_new.sum()
                        if s > 1e-15:
                            P_new /= s
                            score_new = _compute_ratio(P_new)
                            score_cur = _compute_ratio(P_cd)
                            if score_new > score_cur + 1e-12:
                                P_cd = P_new
                                if score_new > best_cd_score:
                                    best_cd_score = score_new
                                    best_score = score_new
                                    best_P = P_cd.reshape(10, 10).copy()

    # Phase 4: Final L-BFGS-B polish from the best found
    if time.time() < deadline - 5:
        theta0 = np.log(np.maximum(best_P.flatten(), log_floor))
        for beta in [500.0, 1000.0]:
            if time.time() > deadline - 2:
                break
            try:
                result_final = minimize(
                    lambda th: _objective_neg(th, beta=beta),
                    theta0,
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12, 'disp': False}
                )
                P_final = np.exp(result_final.x)
                P_final = np.clip(P_final, 0, None)
                s = P_final.sum()
                if s > 1e-15:
                    P_final = P_final / s
                    score_final = _compute_ratio(P_final)
                    if score_final > best_score:
                        best_score = score_final
                        best_P = P_final.reshape(10, 10).copy()
                        theta0 = result_final.x
            except Exception:
                pass

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
