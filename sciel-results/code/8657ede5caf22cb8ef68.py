import numpy as np
import math
import time

def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = t_start + 110.0  # 110s safety margin before 120s limit

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12

    # Convert training_paths to array
    paths = np.asarray(training_paths, dtype=np.float64)
    if paths.ndim == 1:
        paths = paths.reshape(1, -1)
    n_paths, n_steps = paths.shape

    # Determine number of exercise dates from paths
    if n_steps > n_exercises + 1:
        paths = paths[:, -n_exercises - 1:]
    n_steps = paths.shape[1]
    n = min(n_exercises, n_steps - 1) if n_steps > 1 else n_exercises

    dt = T / n

    # --- High-resolution binomial tree backward induction ---
    # Use a fine grid with many steps to accurately compute the continuation value
    # The key insight: use a high-resolution tree (e.g., 1200 steps) for accurate
    # European-style continuation values, then use those to determine exercise boundaries
    
    # Number of fine steps for the tree (high resolution for accuracy)
    n_fine = 1200
    dt_fine = T / n_fine
    u_fine = math.exp(sigma * math.sqrt(dt_fine))
    d_fine = 1.0 / u_fine
    p_fine = (math.exp(r * dt_fine) - d_fine) / (u_fine - d_fine)
    df_fine = math.exp(-r * dt_fine)

    # Grid for the fine tree
    S_min = 1.0
    S_max = 500.0
    n_grid_fine = 10000
    S_grid = np.geomspace(S_min, S_max, n_grid_fine)

    # Backward induction on the fine tree to get accurate continuation values
    # At each exercise date, we need the continuation value (value if we don't exercise)
    
    # Start from maturity
    V_fine = np.maximum(K - S_grid, 0.0)
    
    # Store continuation values at each exercise date
    # continuation_values[j] = continuation value at exercise date j (for j=1..n)
    continuation_values = [None] * (n + 1)
    continuation_values[n] = V_fine.copy()
    
    # For each exercise date going backward, compute the continuation value
    # The continuation value at date j is the discounted expected value at date j+1
    # assuming no exercise at date j (but optimal exercise from j+1 onward)
    
    # We need to track the value function through the fine steps
    # Exercise dates are at times: j*dt for j=1,...,n
    # In the fine tree, these correspond to steps: j * (n_fine / n)
    
    fine_steps_per_exercise = n_fine // n  # Should be 100 if n=12, n_fine=1200
    
    # Backward induction
    V_current = np.maximum(K - S_grid, 0.0)  # Value at maturity
    
    # We'll iterate backward through exercise dates
    # For each exercise date, we need to propagate the value function backward
    # through the fine steps to get the continuation value at that date
    
    for j in range(n - 1, 0, -1):
        if time.time() > deadline - 30.0:
            # Fallback: use simple boundary
            for jj in range(j, 0, -1):
                if continuation_values[jj] is None:
                    continuation_values[jj] = np.maximum(K - S_grid, 0.0)
            break
        
        # Propagate V_current backward through fine_steps_per_exercise fine steps
        # to get the continuation value at exercise date j
        V_prop = V_current.copy()
        
        for step in range(fine_steps_per_exercise):
            if time.time() > deadline - 30.0:
                break
            Su = np.clip(S_grid * u_fine, S_grid[0], S_grid[-1])
            Sd = np.clip(S_grid * d_fine, S_grid[0], S_grid[-1])
            V_up = np.interp(Su, S_grid, V_prop)
            V_down = np.interp(Sd, S_grid, V_prop)
            V_prop = df_fine * (p_fine * V_up + (1 - p_fine) * V_down)
        
        # V_prop is now the continuation value at exercise date j
        # (value if we don't exercise at j, assuming optimal exercise from j+1 onward)
        continuation_values[j] = V_prop
        
        # Now, for the next iteration (j-1), we need the value function that
        # includes the optimal exercise decision at date j
        # Value at date j = max(intrinsic, continuation)
        intrinsic = np.maximum(K - S_grid, 0.0)
        V_current = np.maximum(intrinsic, V_prop)
    
    # Now we have continuation values at each exercise date
    # Determine exercise boundaries: exercise when intrinsic >= continuation
    
    boundaries = [0.0] * (n + 1)
    boundaries[n] = K  # At maturity, always exercise if in the money
    
    for j in range(n - 1, 0, -1):
        if continuation_values[j] is None:
            boundaries[j] = K * 0.95
            continue
        
        intrinsic = np.maximum(K - S_grid, 0.0)
        diff = intrinsic - continuation_values[j]
        exercise_mask = diff >= 0
        
        if np.any(exercise_mask):
            idx = np.where(exercise_mask)[0]
            last_idx = idx[-1]
            if last_idx < n_grid_fine - 1:
                diff0 = diff[last_idx]
                diff1 = diff[last_idx + 1]
                if diff0 - diff1 != 0:
                    t = diff0 / (diff0 - diff1)
                    t = max(0.0, min(1.0, t))
                    boundaries[j] = S_grid[last_idx] + t * (S_grid[last_idx + 1] - S_grid[last_idx])
                else:
                    boundaries[j] = S_grid[last_idx]
            else:
                boundaries[j] = S_grid[-1]
        else:
            boundaries[j] = 0.0
    
    # Ensure all boundaries are set
    for j in range(1, n + 1):
        if boundaries[j] <= 0:
            boundaries[j] = K * 0.95

    # --- Step 2: Refine using Longstaff-Schwartz regression on training paths ---
    # Use the fine tree continuation values as a guide for regression
    
    for j in range(n - 1, 0, -1):
        if time.time() > deadline - 20.0:
            break

        S_j = paths[:, j]
        intrinsic_j = np.maximum(K - S_j, 0.0)

        # Interpolate continuation value from the fine tree
        S_j_clipped = np.clip(S_j, S_grid[0], S_grid[-1])
        V_cont_grid = np.interp(S_j_clipped, S_grid, continuation_values[j])

        # Use regression to smooth the continuation value estimate
        logS = np.log(S_j + 1e-10)
        X = np.column_stack([
            np.ones_like(S_j),
            S_j,
            S_j ** 2,
            logS,
            np.maximum(K - S_j, 0.0),
            np.maximum(K - S_j, 0.0) * logS
        ])

        weights = np.ones_like(S_j)
        itm_mask = S_j < K * 1.1
        weights[itm_mask] = 2.0

        try:
            Xw = X * np.sqrt(weights)[:, None]
            yw = V_cont_grid * np.sqrt(weights)
            coeffs, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
            V_cont_reg = X @ coeffs
            V_cont_reg = np.maximum(V_cont_reg, 0.0)
        except:
            V_cont_reg = V_cont_grid

        diff = intrinsic_j - V_cont_reg
        sort_idx = np.argsort(S_j)
        S_sorted = S_j[sort_idx]
        diff_sorted = diff[sort_idx]

        positive_mask = diff_sorted >= 0
        if np.any(positive_mask):
            last_pos = np.where(positive_mask)[0][-1]
            if last_pos < len(S_sorted) - 1:
                S0 = S_sorted[last_pos]
                S1 = S_sorted[last_pos + 1]
                d0 = diff_sorted[last_pos]
                d1 = diff_sorted[last_pos + 1]
                if d0 - d1 != 0:
                    t = d0 / (d0 - d1)
                    t = max(0.0, min(1.0, t))
                    new_boundary = S0 + t * (S1 - S0)
                else:
                    new_boundary = S0
            else:
                new_boundary = S_sorted[-1]
        else:
            new_boundary = 0.0

        if new_boundary > 0:
            # Blend with the fine tree boundary
            boundaries[j] = 0.6 * boundaries[j] + 0.4 * new_boundary
        else:
            boundaries[j] = K * 0.95

    # --- Step 3: Coordinate ascent optimization of boundaries ---
    def expected_payoff(bnds):
        total = 0.0
        for j in range(n, 0, -1):
            S_j = paths[:, j]
            intrinsic = np.maximum(K - S_j, 0.0)
            exercise_mask = S_j <= bnds[j]
            payoff = np.where(exercise_mask, intrinsic, 0.0)
            disc = math.exp(-r * j * dt)
            total += disc * np.mean(payoff)
        return total

    best_bnds = boundaries[:]
    best_payoff = expected_payoff(best_bnds)

    for iteration in range(3):
        if time.time() > deadline - 5.0:
            break

        for j in range(n - 1, 0, -1):
            if time.time() > deadline - 5.0:
                break

            # Try a few perturbed boundaries around the current one
            current_b = best_bnds[j]
            if current_b <= 0:
                continue

            # Search in log space for better boundary
            candidates = []
            for factor in [0.85, 0.92, 0.97, 1.03, 1.08, 1.15]:
                candidates.append(current_b * factor)

            best_candidate = current_b
            best_candidate_payoff = best_payoff

            for cand in candidates:
                test_bnds = best_bnds[:]
                test_bnds[j] = cand
                cand_payoff = expected_payoff(test_bnds)
                if cand_payoff > best_candidate_payoff:
                    best_candidate_payoff = cand_payoff
                    best_candidate = cand

            if best_candidate != current_b:
                best_bnds[j] = best_candidate
                best_payoff = best_candidate_payoff

    # Final boundaries
    for j in range(1, n + 1):
        if best_bnds[j] <= 0:
            best_bnds[j] = K * 0.95

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

    return ExercisePolicy(best_bnds, K)
