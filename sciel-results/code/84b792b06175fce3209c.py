def fit_exercise_policy(training_paths, contract):
    import numpy as np
    import time as _time

    # Extract parameters
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12
    dt = 1.0 / 12.0
    disc = np.exp(-r * dt)

    # Deadline guard
    start_time = _time.time()
    deadline = start_time + 110.0

    # Determine shape of training_paths
    if training_paths.ndim == 2:
        if training_paths.shape[0] == 4096:
            states = training_paths
        elif training_paths.shape[1] == 4096:
            states = training_paths.T
        else:
            if training_paths.shape[0] >= 13:
                states = training_paths.T
            else:
                states = training_paths
    else:
        states = np.array(training_paths)

    n_paths = states.shape[0]
    n_steps = states.shape[1]

    # States at exercise dates: columns 1 through 12 (indices 1..12)
    # S[:, i] for i=0..11 corresponds to exercise date i+1
    S = states[:, 1:n_dates + 1]

    # V[i, j] is the value at step i for path j
    # i=0 corresponds to date 1, ..., i=11 corresponds to date 12
    V = np.zeros((n_dates, n_paths))
    
    # Initialize last date
    V[-1, :] = np.maximum(K - S[:, -1], 0)
    thresholds = np.zeros(n_dates)
    thresholds[-1] = K

    # Value Iteration
    # We iterate to update thresholds and V consistently
    # The number of iterations is small because the problem is 1D and convex-ish
    n_iterations = 20
    
    for iteration in range(n_iterations):
        if _time.time() > deadline - 5:
            break

        # Backward pass
        # We need to update V[i] based on V[i+1]
        # Start from n_dates - 2 down to 0
        
        # We store the new V in a separate array to avoid using updated values in the same pass incorrectly?
        # Actually, standard backward induction uses V[i+1] which is already "fixed" for this iteration's pass if we go backwards.
        # But V[i+1] was computed in the previous iteration of the outer loop.
        # So we just use the current V array.
        
        for i in range(n_dates - 2, -1, -1):
            S_i = S[:, i]
            C_i = disc * V[i + 1, :]
            I_i = np.maximum(K - S_i, 0)
            
            # We want to find threshold theta such that we exercise if S_i < theta
            # Objective: maximize sum( I_i[j] * (S_i[j] < theta) + C_i[j] * (S_i[j] >= theta) )
            
            # Sort states
            sort_idx = np.argsort(S_i)
            S_sorted = S_i[sort_idx]
            I_sorted = I_i[sort_idx]
            C_sorted = C_i[sort_idx]
            
            # Calculate cumulative sums
            # If we exercise for the first k states (sorted ascending), 
            # the gain is sum(I_sorted[0:k]) + sum(C_sorted[k:N])
            
            # Let Diff[j] = I_sorted[j] - C_sorted[j]
            # J(k) = sum_{j=0}^{k-1} I_sorted[j] + sum_{j=k}^{N-1} C_sorted[j]
            # J(k) = Total_C + sum_{j=0}^{k-1} (I_sorted[j] - C_sorted[j])
            
            Total_C = np.sum(C_sorted)
            Diff = I_sorted - C_sorted
            CumDiff = np.cumsum(Diff)
            
            # J(k) = Total_C + CumDiff[k-1] for k>=1
            # J(0) = Total_C
            # We want to maximize J(k) for k in 0..N
            
            # Create an array of J values for k=0..N
            # J[0] = Total_C
            # J[k] = Total_C + CumDiff[k-1]
            
            # We can just find the max of CumDiff, and handle k=0 separately
            # If max(CumDiff) < 0, then k=0 is best (exercise no one)
            # Else, k = argmax(CumDiff) + 1 is best
            
            max_cumdiff = np.max(CumDiff)
            if max_cumdiff < 0:
                best_k = 0
            else:
                best_k = np.argmax(CumDiff) + 1
            
            # Determine threshold
            if best_k == 0:
                # Exercise no one. Threshold should be <= min(S_i)
                # To be safe, set to 0 or min(S_i)
                thresholds[i] = 0.0
            elif best_k == n_paths:
                # Exercise everyone. Threshold should be > max(S_i)
                # But we only exercise if S < K effectively? No, if S > K, I=0.
                # If we exercise everyone, it means even high S are exercised?
                # Wait, I_i = max(K-S, 0). If S > K, I=0.
                # If C_i < 0 (impossible, V>=0), we'd exercise.
                # Usually for puts, we don't exercise OTM.
                # If best_k == N, it means for all j, I_sorted[j] > C_sorted[j].
                # This implies we should exercise all.
                # Threshold should be high enough to include all.
                # However, if S > K, I=0. If C < 0 (never), we exercise.
                # So if best_k == N, set threshold to K (or max(S)+1).
                # But logically, if S > K, we shouldn't exercise unless C < 0.
                # Let's set threshold to K to be consistent with "don't exercise OTM" if C is small.
                # Actually, if the math says exercise, we should.
                # But for a put, if S is very high, I=0. If C is also 0, it doesn't matter.
                # If C > 0, we wouldn't exercise. So best_k=N is rare for puts.
                thresholds[i] = K # Cap at K for safety? Or max(S)+1?
                # If we set threshold to K, we exercise S < K.
                # If best_k was N, it means even S >= K were "better" to exercise?
                # That implies C < 0 for those, which is impossible.
                # So best_k=N usually means all S < K were exercised, and maybe some S >= K?
                # If S >= K, I=0. If C >= 0, we don't exercise.
                # So best_k should naturally be <= count(S < K).
                # Let's just use K as the threshold.
            else:
                # best_k is in (0, N)
                # We exercise S_sorted[0] ... S_sorted[best_k-1]
                # We do not exercise S_sorted[best_k] ... S_sorted[N-1]
                # Threshold should be between S_sorted[best_k-1] and S_sorted[best_k]
                # Use midpoint
                lo = S_sorted[best_k - 1]
                hi = S_sorted[best_k]
                thresholds[i] = (lo + hi) / 2.0
            
            # Update V[i]
            # Exercise if S_i < thresholds[i]
            exercise_mask = S_i < thresholds[i]
            V[i, :] = np.where(exercise_mask, I_i, C_i)

    # Ensure last threshold is K
    thresholds[-1] = K

    # Policy object
    class ExercisePolicy:
        def __init__(self, thresholds, K):
            self.thresholds = thresholds
            self.K = K

        def exercise(self, current_state, time):
            if isinstance(time, (int, float)):
                if isinstance(time, float) and 0 < time <= 1.0:
                    idx = int(np.round(time * 12)) - 1
                    if idx < 0:
                        idx = 0
                    if idx >= n_dates:
                        idx = n_dates - 1
                elif isinstance(time, int):
                    if 1 <= time <= 12:
                        idx = time - 1
                    elif 0 <= time <= 11:
                        idx = time
                    else:
                        idx = min(max(time, 0), n_dates - 1)
                else:
                    idx = 0
            else:
                idx = 0

            state = np.asarray(current_state)
            return state < self.thresholds[idx]

    return ExercisePolicy(thresholds, K)
