def fit_exercise_policy(training_paths, contract):
    import numpy as np
    import time as _time
    from scipy.optimize import brentq

    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12
    dt = 1.0 / 12.0
    disc = np.exp(-r * dt)

    # Determine shape of training_paths
    # Expected: 4096 paths, 12 steps.
    # Could be (4096, 12) or (12, 4096).
    if training_paths.ndim != 2:
        raise ValueError("training_paths must be 2D")
    
    if training_paths.shape[0] == 4096:
        paths = training_paths
    elif training_paths.shape[1] == 4096:
        paths = training_paths.T
    else:
        # Fallback: assume first dim is paths if it's larger, or just take as is
        # But contract says 4096.
        paths = training_paths

    n_paths, n_steps = paths.shape
    
    # Ensure we have 12 steps
    if n_steps != 12:
        # If mismatch, try to handle or raise
        # For this task, we assume 12.
        pass

    # V_next stores the value at step k+1 for all paths
    # We will iterate k from 11 down to 0 (indices for steps 12 down to 1)
    # Note: paths[:, 11] is S at t=12 (index 11 if 0-indexed steps 0..11 correspond to t=1/12..1)
    # Wait, standard convention:
    # training_paths[:, 0] -> t=1/12
    # training_paths[:, 11] -> t=12/12
    
    # Let's verify. "12 monthly exercise dates".
    # Usually path simulation gives S at t_1, ..., t_12.
    # So paths[:, 0] is S(t_1), ..., paths[:, 11] is S(t_12).
    
    V_next = np.maximum(K - paths[:, 11], 0.0)
    
    thresholds = np.zeros(n_dates)
    thresholds[-1] = K # At maturity, exercise if S < K
    
    # active_mask: which paths have NOT exercised yet
    active_mask = np.ones(n_paths, dtype=bool)
    
    # We will store the fitted coefficients for each step to extract thresholds later
    # Or extract thresholds immediately.
    # Let's extract immediately.
    
    # Basis function: polynomial of degree 4 on scaled S
    deg = 4
    
    for k_idx in range(10, -1, -1):
        # k_idx corresponds to step index in the array (0..11)
        # This is exercise date k_idx + 1
        # We are calculating value at step k_idx, using values from k_idx + 1
        
        S_curr = paths[:, k_idx]
        S_next = paths[:, k_idx + 1] # Not directly used for regression, V_next is
        
        # Regression data:
        # X: S_curr for active paths
        # Y: V_next for active paths
        # We want to fit E[V_next | S_curr]
        
        X = S_curr[active_mask]
        Y = V_next[active_mask]
        
        if len(X) == 0:
            # No active paths, threshold is 0? Or K?
            # If no paths are active, it means all exercised in later steps.
            # This is unlikely for a put with these params, but handle it.
            thresholds[k_idx] = 0.0
            continue
            
        # Scale X for regression
        X_scaled = X / K
        
        # Design matrix
        # Columns: 1, x, x^2, x^3, x^4
        basis = np.vander(X_scaled, N=deg + 1, increasing=True)
        
        # Solve least squares
        # Use np.linalg.lstsq
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(basis, Y, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback to simple threshold if regression fails
            thresholds[k_idx] = K
            continue
            
        # Define the continuation value function
        # C(S) = disc * poly(S/K)
        
        def continuation_val(S_val):
            S_s = S_val / K
            # Evaluate polynomial
            # coeffs are [c0, c1, c2, c3, c4] for 1, x, x^2, x^3, x^4
            # np.polyval expects highest power first, so reverse
            return disc * np.dot(np.power(S_s, np.arange(deg + 1)), coeffs)
            
        # Find threshold: K - S = continuation_val(S)
        # f(S) = K - S - continuation_val(S)
        
        def f(S_val):
            return K - S_val - continuation_val(S_val)
            
        # We know f(0) = K - 0 - C(0) > 0 (usually)
        # f(K) = 0 - C(K) < 0 (usually)
        # Check bounds
        f_low = f(0.0)
        f_high = f(K)
        
        if f_low * f_high > 0:
            # No root in [0, K]?
            # If f(0) < 0, never exercise (threshold 0)
            # If f(K) > 0, always exercise (threshold K)
            if f_low < 0:
                thresholds[k_idx] = 0.0
            else:
                thresholds[k_idx] = K
        else:
            try:
                thresh = brentq(f, 0.0, K, xtol=1e-6)
                thresholds[k_idx] = thresh
            except ValueError:
                # Bisection failed, fallback
                # Find where sign change happens in grid
                S_grid = np.linspace(0, K, 100)
                vals = f(S_grid)
                sign_changes = np.where(np.diff(np.sign(vals)) != 0)[0]
                if len(sign_changes) > 0:
                    idx = sign_changes[0]
                    # Linear interpolation
                    S1, S2 = S_grid[idx], S_grid[idx+1]
                    V1, V2 = vals[idx], vals[idx+1]
                    thresh = S1 - V1 * (S2 - S1) / (V2 - V1)
                    thresholds[k_idx] = np.clip(thresh, 0, K)
                else:
                    thresholds[k_idx] = K if f_low > 0 else 0.0

        # Update active_mask for the next iteration (earlier step)
        # Exercise if Immediate Payoff >= Continuation Value
        # Immediate Payoff at S_curr
        imm_payoff = np.maximum(K - S_curr, 0.0)
        
        # Calculate continuation value for all active paths
        # Re-use the fitted coeffs
        S_curr_scaled = S_curr[active_mask] / K
        cont_vals = disc * np.dot(np.power(S_curr_scaled, np.arange(deg + 1)), coeffs)
        
        # Exercise decision
        ex_mask_local = imm_payoff[active_mask] >= cont_vals
        
        # Update global active_mask
        # Paths that exercised at this step are NO LONGER active for previous steps
        # We need to update the mask for the indices that are currently active
        # active_mask is a boolean array.
        # We want to set active_mask[i] = False for i where active_mask[i] is True and ex_mask_local is True.
        
        # Create a temporary mask for all paths
        # active_mask is currently True for paths that survived up to step k+1
        # We check exercise at step k.
        # If exercised, they drop out.
        
        # Efficient update:
        # indices_active = np.where(active_mask)[0]
        # indices_exercised = indices_active[ex_mask_local]
        # active_mask[indices_exercised] = False
        
        # This is O(N).
        
        indices_active = np.where(active_mask)[0]
        indices_exercised = indices_active[ex_mask_local]
        active_mask[indices_exercised] = False
        
        # Update V_next for the next iteration (which will be step k)
        # V_next for step k is max(imm, cont) for active paths
        # For inactive paths, value is 0 (or irrelevant)
        
        V_curr = np.zeros(n_paths)
        # For active paths (before this step's exercise decision)
        # V_curr[active] = max(imm[active], cont[active])
        # Note: active_mask has just been updated to EXCLUDE those who exercised.
        # So we need to use the OLD active_mask to compute V_curr for the NEXT step's regression?
        # No.
        # The regression at step k-1 will use V_curr (value at step k) for paths that are active at step k-1.
        # Paths active at step k-1 are those that did NOT exercise at step k, k+1, ..., 12.
        # So, V_curr should be the value at step k for all paths.
        # For paths that exercised at step k, the "value" is the payoff, but they are gone.
        # For paths that did NOT exercise at step k, the value is the continuation value.
        # The regression at step k-1 will only use paths that are active at step k-1.
        # These are a subset of paths that were active at step k AND did not exercise at step k.
        # So, for the regression at k-1, we need V_curr to be the "continuation value" for those paths.
        # Which is exactly what we computed as `cont_vals` for the paths that remained active.
        # However, `V_next` for the next loop iteration (k-1) needs to be the value at step k for ALL paths that might be active at k-1.
        # The set of paths active at k-1 is `active_mask` (after update).
        # For these paths, V_curr should be the continuation value calculated above.
        # For paths that exercised at step k, they are not in `active_mask`, so their V_curr doesn't matter for the next regression.
        
        # So, we can just set V_curr[active_mask] = cont_vals (which are the values for the remaining active paths)
        # Wait, `cont_vals` was computed for `indices_active` (the OLD active set).
        # `indices_exercised` are a subset of `indices_active`.
        # The new `active_mask` has False for `indices_exercised`.
        # The paths remaining in `active_mask` are `indices_active` minus `indices_exercised`.
        # We need to map `cont_vals` back to the full array for the remaining active paths.
        
        # Let's do it cleanly:
        V_curr = np.zeros(n_paths)
        # For the paths that are still active (did not exercise at k):
        # We need their continuation values.
        # We already have `cont_vals` for the OLD active set.
        # We can just assign V_curr[indices_active] = np.maximum(imm_payoff[indices_active], cont_vals)
        # This is correct for all paths that were active at k.
        # The next regression will filter by the NEW active_mask.
        
        V_curr[indices_active] = np.maximum(imm_payoff[indices_active], cont_vals)
        
        V_next = V_curr
        
    class ExercisePolicy:
        def __init__(self, thresholds, K):
            self.thresholds = thresholds
            self.K = K

        def exercise(self, current_state, time):
            if isinstance(time, (int, float)):
                if isinstance(time, float) and 0 < time <= 1.0:
                    idx = int(np.round(time * 12)) - 1
                    if idx < 0: idx = 0
                    if idx >= 12: idx = 11
                elif isinstance(time, int):
                    if 1 <= time <= 12: idx = time - 1
                    elif 0 <= time <= 11: idx = time
                    else: idx = min(max(time, 0), 11)
                else: idx = 0
            else: idx = 0
            
            state = np.asarray(current_state)
            return state < self.thresholds[idx]

    return ExercisePolicy(thresholds, K)
