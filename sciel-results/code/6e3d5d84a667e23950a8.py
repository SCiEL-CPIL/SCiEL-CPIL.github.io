import numpy as np
import math
import time

def _binomial_tree_boundaries(K, r, sigma, T, n_exercises, n_steps=400):
    """Compute optimal exercise boundaries via high-resolution binomial tree backward induction."""
    dt = T / n_steps
    # Map exercise dates to tree step indices
    exercise_steps = [int(round(j * n_steps / n_exercises)) for j in range(1, n_exercises + 1)]
    exercise_steps = sorted(set(exercise_steps))
    
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    
    # Build tree: S[i][j] = S0 * u^i * d^(j-i)
    # We'll work in log-space for the boundary
    # Value array: V[j][i] for step j, node i
    # At maturity: V[n_steps][i] = max(K - S[i], 0)
    
    # We only need to store one row at a time (backward induction)
    # S at step j: S_i = u^i * d^(j-i) (S0=1, we'll scale by K later)
    # Actually let's work with S0=1 and scale boundaries by 1 at the end
    # Or better: work directly with S0=1, boundaries will be in units of S0
    
    # For a put with K=1 (normalized), compute boundaries
    # V[j][i] = max(K_norm - S_i, disc * (p*V[j+1][i+1] + (1-p)*V[j+1][i]))
    
    K_norm = K / 1.0  # We'll use S0=1, so K_norm = K if S0=1... 
    # Actually let's just set S0=1 and use K directly. The boundary is absolute.
    # S_i at step j = u^i * d^(j-i)
    
    # Precompute log_u, log_d
    log_u = math.log(u)
    log_d = math.log(d)
    
    # V_next[i] for i in 0..j
    # Start at maturity
    j = n_steps
    i_max = j
    # V[i] = max(K - u^i * d^(j-i), 0)
    V = np.zeros(i_max + 1)
    for i in range(i_max + 1):
        S = math.exp(i * log_u + (j - i) * log_d)
        V[i] = max(K - S, 0.0)
    
    # Backward induction
    boundaries_at_steps = {}  # step -> boundary S value
    for j in range(n_steps - 1, -1, -1):
        i_max = j
        V_new = np.zeros(i_max + 1)
        for i in range(i_max + 1):
            S = math.exp(i * log_u + (j - i) * log_d)
            intrinsic = max(K - S, 0.0)
            cont = disc * (p * V[i + 1] + (1.0 - p) * V[i])
            V_new[i] = max(intrinsic, cont)
        
        # Check if this step is an exercise date
        if j in exercise_steps:
            # Find boundary: largest S where intrinsic >= continuation
            # Scan from high i (high S) to low i
            # boundary is between the highest node where we exercise and the one above
            # Find the largest i such that V_new[i] == intrinsic[i] (exercise)
            # and i+1 has V_new[i+1] > intrinsic[i+1] (continue)
            # More precisely: find the crossover
            # For put: exercise when S <= boundary
            # So we want the largest S where exercise is optimal
            # Scan i from 0 upward, find where it switches from exercise to continue
            boundary = K  # default: always exercise
            for i in range(i_max + 1):
                S = math.exp(i * log_u + (j - i) * log_d)
                intrinsic = max(K - S, 0.0)
                cont = disc * (p * V[i + 1] + (1.0 - p) * V[i]) if i < i_max else 0.0
                if intrinsic >= cont:
                    boundary = S  # This node exercises, so boundary is at least S
                else:
                    # This node continues, so boundary is below S
                    # Interpolate between this node and previous
                    if i > 0:
                        S_prev = math.exp((i - 1) * log_u + (j - (i - 1)) * log_d)
                        # Linear interpolation in log-S space
                        boundary = 0.5 * (S_prev + S)
                    break
            boundaries_at_steps[j] = boundary
        
        V = V_new
    
    # Map exercise steps back to exercise indices
    # exercise_steps[k] corresponds to exercise date k+1 (1-indexed)
    # We need boundaries for exercise dates 1..n_exercises
    # exercise_steps is sorted, so exercise_steps[k] is the tree step for exercise date k+1
    n_ex = len(exercise_steps)
    bnds = np.zeros(n_ex + 1)  # 1-indexed: bnds[1]..bnds[n_ex]
    for k, step in enumerate(exercise_steps):
        if step in boundaries_at_steps:
            bnds[k + 1] = boundaries_at_steps[step]
        else:
            bnds[k + 1] = K * 0.95
    
    # At maturity (last exercise), always exercise if ITM
    bnds[n_ex] = K
    
    # Ensure monotonicity: for a put, boundary should be non-decreasing as we go back in time
    # (earlier dates have higher or equal boundaries)
    # bnds[1] >= bnds[2] >= ... >= bnds[n_ex]
    # Wait: for a put, you're more likely to exercise early if deep ITM
    # Actually for American/Bermudan puts, the boundary typically increases as time to maturity increases
    # So bnds[1] (earliest) >= bnds[2] >= ... >= bnds[n_ex] (maturity)
    # But at maturity, boundary = K (exercise if S < K)
    # So we need bnds[1] >= bnds[2] >= ... >= bnds[n_ex-1] >= bnds[n_ex] = K? No...
    # Actually at maturity, you exercise if S < K, so boundary = K.
    # At earlier dates, you might exercise even if S is slightly above K? No, for a put,
    # you only exercise if S < K (intrinsic > 0). The boundary is always < K.
    # The boundary should be non-decreasing as time to maturity increases.
    # So bnds[1] (most time remaining) >= bnds[2] >= ... >= bnds[n_ex] (least time, = K at maturity)
    # Hmm, but at maturity boundary = K, and earlier boundaries should be <= K.
    # Let me reconsider: for a put, the optimal exercise boundary B(t) satisfies:
    # B(0) > B(T) = K? No, B(T) = K means at maturity you exercise if S < K.
    # Actually B(T) should be K (or slightly below). And B(0) < K typically.
    # The boundary increases toward K as maturity approaches.
    # So bnds[1] <= bnds[2] <= ... <= bnds[n_ex] = K
    # Let me enforce this.
    for j in range(1, n_ex):
        if bnds[j] > bnds[j + 1]:
            bnds[j] = bnds[j + 1]
    
    return bnds


def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = t_start + 140.0  # 10s safety margin before 150s kill

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12

    n = n_exercises
    dt = T / n

    # Convert training_paths to array
    paths = np.asarray(training_paths, dtype=np.float64)
    if paths.ndim == 1:
        paths = paths.reshape(1, -1)
    n_paths, n_steps = paths.shape

    # Determine number of exercise dates from paths
    if n_steps > n_exercises:
        paths = paths[:, -n_exercises - 1:] if n_steps >= n_exercises + 1 else paths
    n_steps = paths.shape[1]
    n_ex = min(n_exercises, n_steps - 1) if n_steps > 1 else n_exercises
    n = n_ex
    dt = T / n

    # Precompute discount factors
    disc_factors = np.array([math.exp(-r * j * dt) for j in range(n + 1)])

    # Precompute intrinsic values for all paths and all dates
    intrinsic = np.maximum(K - paths[:, 1:n+1], 0.0)  # shape (n_paths, n)
    S_all = paths[:, 1:n+1]  # shape (n_paths, n)

    # --- Step 1: Initialize boundaries using high-resolution binomial tree ---
    tree_bnds = _binomial_tree_boundaries(K, r, sigma, T, n, n_steps=400)
    boundaries = tree_bnds.copy()

    # --- Step 2: Fast payoff computation ---
    def compute_payoff_fast(bnds):
        """Compute mean discounted payoff using vectorized operations."""
        total_payoff = np.zeros(n_paths, dtype=np.float64)
        exercised = np.zeros(n_paths, dtype=bool)

        for j in range(n):
            active = ~exercised
            if not np.any(active):
                break

            # Exercise condition: S_j <= boundary
            exercise_mask = (S_all[:, j] <= bnds[j + 1]) & active

            if np.any(exercise_mask):
                total_payoff[exercise_mask] += disc_factors[j + 1] * intrinsic[exercise_mask, j]
                exercised[exercise_mask] = True

        return np.mean(total_payoff)

    def compute_payoff_batch(bnds, batch_idx):
        """Compute mean discounted payoff on a subset of paths."""
        S_sub = S_all[batch_idx]
        intr_sub = intrinsic[batch_idx]
        total_payoff = np.zeros(len(batch_idx), dtype=np.float64)
        exercised = np.zeros(len(batch_idx), dtype=bool)

        for j in range(n):
            active = ~exercised
            if not np.any(active):
                break

            exercise_mask = (S_sub[:, j] <= bnds[j + 1]) & active

            if np.any(exercise_mask):
                total_payoff[exercise_mask] += disc_factors[j + 1] * intr_sub[exercise_mask, j]
                exercised[exercise_mask] = True

        return np.mean(total_payoff)

    # --- Step 3: Coordinate ascent optimization with fast evaluation ---
    best_bnds = boundaries.copy()
    best_payoff = compute_payoff_fast(best_bnds)
    
    # Also track tree payoff as baseline
    tree_payoff = best_payoff

    # Multi-scale coordinate ascent
    scales = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for scale in scales:
        if time.time() > deadline - 15.0:
            break
        
        improved = True
        while improved:
            if time.time() > deadline - 15.0:
                break
            improved = False
            for j in range(1, n + 1):
                if time.time() > deadline - 15.0:
                    break
                current = best_bnds[j]
                step = max(0.1, current * scale)
                for delta in [-step, step]:
                    trial_val = max(1.0, min(K, current + delta))
                    if abs(trial_val - current) < 1e-6:
                        continue
                    trial = best_bnds.copy()
                    trial[j] = trial_val
                    payoff = compute_payoff_fast(trial)
                    if payoff > best_payoff + 1e-10:
                        best_bnds[j] = trial_val
                        best_payoff = payoff
                        improved = True
                        current = trial_val

    # --- Step 4: Fine-tuning with smaller steps ---
    for scale in [0.001, 0.0005, 0.0002, 0.0001]:
        if time.time() > deadline - 5.0:
            break
        for j in range(1, n + 1):
            if time.time() > deadline - 5.0:
                break
            current = best_bnds[j]
            step = max(0.05, current * scale)
            for delta in [-step, step]:
                trial_val = max(1.0, min(K, current + delta))
                if abs(trial_val - current) < 1e-7:
                    continue
                trial = best_bnds.copy()
                trial[j] = trial_val
                payoff = compute_payoff_fast(trial)
                if payoff > best_payoff + 1e-12:
                    best_bnds[j] = trial_val
                    best_payoff = payoff

    # Final check: use the better of tree and MC-optimized
    tree_bnds_final = _binomial_tree_boundaries(K, r, sigma, T, n, n_steps=600)
    tree_payoff_final = compute_payoff_fast(tree_bnds_final)
    
    if tree_payoff_final > best_payoff:
        best_bnds = tree_bnds_final
        best_payoff = tree_payoff_final

    boundaries = best_bnds

    # Final fallback
    for j in range(1, n + 1):
        if boundaries[j] <= 0:
            boundaries[j] = K * 0.95

    class ExercisePolicy:
        def __init__(self, bnds, K_val):
            self.boundaries = bnds
            self.K = K_val

        def exercise(self, current_state, time_param):
            S_arr = np.asarray(current_state, dtype=np.float64)

            if isinstance(time_param, (int, float, np.integer, np.floating)):
                t = int(time_param)
                if t < 1:
                    t = 1
                elif t > 12:
                    t = 12
                return S_arr <= self.boundaries[t]
            else:
                t_arr = np.asarray(time_param)
                t_clipped = np.clip(t_arr, 1, 12).astype(int)
                b = np.array([self.boundaries[t] for t in t_clipped])
                if S_arr.ndim == 1 and b.ndim == 0:
                    return S_arr <= float(b)
                elif S_arr.ndim == 1 and b.ndim == 1:
                    return S_arr <= b
                else:
                    return S_arr <= b

    return ExercisePolicy(boundaries, K)
