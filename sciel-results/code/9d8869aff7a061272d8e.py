"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12

    tp = np.asarray(training_paths, dtype=np.float64)

    # Ensure paths shape is (n_paths, n_times)
    if tp.ndim == 2:
        if tp.shape[0] == 4096:
            paths = tp
        elif tp.shape[1] == 4096:
            paths = tp.T
        else:
            paths = tp
    else:
        paths = tp

    n_paths, n_times = paths.shape
    
    # If n_times is 13, it might include time 0. We only exercise at 1..12.
    # The contract says "12 monthly exercise dates (no time-0 exercise)".
    # If the input includes time 0, indices 1..12 correspond to exercise dates.
    # If the input is just 12 dates, indices 0..11 correspond to exercise dates.
    # Let's assume standard convention: if shape[1] == 13, drop first column.
    if n_times == 13:
        paths = paths[:, 1:]
        n_times = 12
    elif n_times != 12:
        # Fallback: assume last 12 columns
        if n_times > 12:
            paths = paths[:, -12:]
            n_times = 12
        else:
            # Pad or error? Assume it's 12.
            pass

    dt = T / n_dates
    discount = np.exp(-r * dt)
    
    # 1. Compute Optimal Policy via Backward Induction on Training Paths
    # V[i] is the value at time step i (1-indexed in our loop logic, 0-indexed in array)
    # We store decisions for each step.
    
    # Value at maturity (last step, index 11)
    S_last = paths[:, 11]
    V_next = np.maximum(K - S_last, 0.0)
    
    # decisions[i] corresponds to time step i+1 (since we loop backwards from 11 to 0)
    # Let's use 0-indexed steps for the array, but map to 1..12 for the policy.
    # Step 11 is t=1.0 (12th month)
    # Step 0 is t=1/12 (1st month)
    
    decisions = [None] * n_dates
    
    for i in range(n_dates - 1, -1, -1):
        if time.time() > deadline - 5.0:
            break
            
        S_i = paths[:, i]
        immediate = np.maximum(K - S_i, 0.0)
        
        if i == n_dates - 1:
            # At maturity, always exercise if in the money
            decisions[i] = immediate > 0
            V_current = immediate
        else:
            cont = discount * V_next
            decisions[i] = immediate >= cont
            V_current = np.maximum(immediate, cont)
        
        V_next = V_current

    # 2. Extract Initial Thresholds
    # For each step, find the highest S where we exercise.
    # This gives a "crude" boundary.
    thresholds = np.zeros(n_dates)
    for i in range(n_dates):
        if decisions[i] is not None:
            S_i = paths[:, i]
            exercised = decisions[i]
            if np.any(exercised):
                thresholds[i] = float(np.max(S_i[exercised]))
            else:
                # If no one exercised, threshold is very low
                thresholds[i] = 0.0
        else:
            # Fallback if time ran out during backward induction
            thresholds[i] = K * 0.8

    # 3. Refine Thresholds using Coordinate Ascent on Training Paths
    # Objective: Maximize mean discounted payoff
    # We evaluate the policy on the training paths.
    
    def evaluate_policy(thr):
        """
        thr: array of shape (n_dates,)
        Returns mean discounted payoff.
        """
        # Vectorized simulation of the policy
        # active: boolean array of paths that haven't exercised yet
        active = np.ones(n_paths, dtype=bool)
        total_payoff = np.zeros(n_paths)
        
        # Pre-compute discount factors for each step
        # Step i corresponds to time (i+1)*dt
        # Discount factor for payoff at step i is exp(-r * (i+1) * dt)
        
        for i in range(n_dates):
            if not np.any(active):
                break
                
            S_i = paths[:, i]
            # Exercise if active and S_i <= thr[i]
            mask = active & (S_i <= thr[i])
            
            if np.any(mask):
                # Payoff is (K - S_i) at time t_i
                # Discount factor: exp(-r * (i+1) * dt)
                df = np.exp(-r * (i + 1) * dt)
                total_payoff[mask] += df * (K - S_i[mask])
                active[mask] = False
                
        return np.mean(total_payoff)

    best_thr = thresholds.copy()
    best_val = evaluate_policy(best_thr)
    
    # Coordinate Ascent
    # We will iterate through each dimension and try to improve it.
    # Use a fine grid around the current value.
    
    # Determine search range for each threshold.
    # Thresholds for puts are typically between 0 and K.
    # Early steps (low i) have lower thresholds. Late steps (high i) have higher thresholds.
    
    # We'll do a few passes.
    max_passes = 20
    pass_count = 0
    
    while pass_count < max_passes and time.time() < deadline - 10.0:
        improved = False
        for i in range(n_dates):
            # Define search range
            # Based on current value, search +/- 10% or so.
            # But ensure we stay within [0, K].
            
            # A more robust search: try a set of candidate values.
            # Since we don't know the exact shape, let's try a local grid.
            current_val = best_thr[i]
            
            # Step size
            step_size = 0.5
            # Try values: current - 2*step, current - step, current + step, current + 2*step
            # Also try 0 and K if close.
            
            candidates = [
                current_val - 2*step_size,
                current_val - step_size,
                current_val + step_size,
                current_val + 2*step_size,
            ]
            
            # Add boundaries if we are close
            if current_val < 5.0:
                candidates.append(0.0)
                candidates.append(2.5)
            if current_val > K - 5.0:
                candidates.append(K)
                candidates.append(K - 2.5)
                
            # Filter valid candidates
            valid_candidates = [c for c in candidates if 0.0 <= c <= K]
            
            best_local_val = best_val
            best_local_thr = current_val
            
            for c in valid_candidates:
                if c == current_val:
                    continue
                old_val = best_thr[i]
                best_thr[i] = c
                val = evaluate_policy(best_thr)
                if val > best_local_val:
                    best_local_val = val
                    best_local_thr = c
                best_thr[i] = old_val
            
            if best_local_thr != current_val:
                best_thr[i] = best_local_thr
                best_val = best_local_val
                improved = True
                
        if not improved:
            # Try finer step size
            step_size *= 0.5
            if step_size < 0.01:
                break
        else:
            # Keep step size or slightly reduce to converge
            step_size *= 0.8
            
        pass_count += 1
        
        # Check time
        if time.time() > deadline - 10.0:
            break

    # 4. Create Policy Object
    class ExercisePolicy:
        def __init__(self, thresholds, K):
            self.thresholds = thresholds
            self.K = K
            self.n_dates = len(thresholds)

        def exercise(self, current_state, time):
            cs = np.asarray(current_state, dtype=np.float64)
            # Map time to index
            # time is in [0, 1].
            # Exercise dates are at 1/12, 2/12, ..., 12/12.
            # If time is 0.0, it's before first date. Return False?
            # The contract says "no time-0 exercise".
            # If time is exactly on a date, use that index.
            
            # Calculate index
            # t_i = (i+1)/12
            # We want i such that t_i is closest to time?
            # Or is 'time' always passed as one of the discrete dates?
            # "candidate receives current states only during scoring... exercise(current_state, time)"
            # Usually in these tasks, 'time' is the specific exercise time.
            
            # Let's assume 'time' is one of the 12 dates.
            # i = round(time * 12) - 1
            # Clamp to [0, n_dates-1]
            
            idx = int(round(time * 12)) - 1
            idx = max(0, min(self.n_dates - 1, idx))
            
            threshold = self.thresholds[idx]
            return (cs <= threshold).astype(bool)

    return ExercisePolicy(best_thr, K)
# EVOLVE-BLOCK-END
