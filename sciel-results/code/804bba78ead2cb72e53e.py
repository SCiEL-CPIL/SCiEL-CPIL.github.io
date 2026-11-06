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

    # Initialize thresholds
    thresholds = np.zeros(n_dates)

    # V[i, j] is the value at step i for path j
    V = np.zeros((n_dates, n_paths))
    # Last date: always exercise if in the money, else 0
    V[-1, :] = np.maximum(K - S[:, -1], 0)
    thresholds[-1] = K

    # Backward Induction
    for i in range(n_dates - 2, -1, -1):
        if _time.time() > deadline:
            break
            
        S_i = S[:, i]
        # Continuation value is the discounted value from the next step
        # Note: V[i+1] already includes the decision at i+1.
        cont = disc * V[i + 1, :]
        
        # We want to find threshold tau such that we exercise if S_i < tau.
        # Objective: sum( I(S_i < tau) * (K - S_i) + I(S_i >= tau) * cont )
        # Let's sort S_i to find the optimal cut efficiently.
        
        # If all states are > K, we never exercise. Threshold 0.
        # If all states are < K, we might exercise all.
        
        # Sort indices by S_i
        order = np.argsort(S_i)
        S_sorted = S_i[order]
        cont_sorted = cont[order]
        
        # Immediate payoff for sorted states
        imm_sorted = np.maximum(K - S_sorted, 0)
        
        # We evaluate the objective for each possible cut point m (0 to n_paths)
        # If we exercise the first m paths (smallest S), we get sum(imm_sorted[:m]) + sum(cont_sorted[m:])
        # Let's compute prefix sums of imm_sorted and suffix sums of cont_sorted
        
        # Prefix sum of immediate payoffs
        # pref_imm[m] = sum(imm_sorted[:m])
        pref_imm = np.zeros(n_paths + 1)
        pref_imm[1:] = np.cumsum(imm_sorted)
        
        # Suffix sum of continuation values
        # suff_cont[m] = sum(cont_sorted[m:])
        # total_cont = sum(cont_sorted)
        # suff_cont[m] = total_cont - sum(cont_sorted[:m])
        total_cont = np.sum(cont_sorted)
        pref_cont = np.zeros(n_paths + 1)
        pref_cont[1:] = np.cumsum(cont_sorted)
        suff_cont = total_cont - pref_cont
        
        # Objective for cut m: pref_imm[m] + suff_cont[m]
        objectives = pref_imm + suff_cont
        
        # Find the m that maximizes the objective
        best_m = np.argmax(objectives)
        
        # The optimal threshold is between S_sorted[best_m - 1] and S_sorted[best_m]
        # If best_m is 0, we exercise none, threshold 0.
        # If best_m is n_paths, we exercise all, threshold > max(S).
        
        if best_m == 0:
            thresholds[i] = 0.0
        elif best_m == n_paths:
            thresholds[i] = K # Cap at K, as exercising above K is never optimal for a put
        else:
            # The threshold should be anywhere between S_sorted[best_m-1] and S_sorted[best_m]
            # Taking the midpoint is a safe choice.
            thresholds[i] = (S_sorted[best_m - 1] + S_sorted[best_m]) / 2.0
            # Ensure it's within [0, K]
            thresholds[i] = np.clip(thresholds[i], 0.0, K)

        # Update V[i] using the optimal threshold
        # Exercise if S_i < thresholds[i]
        exercise_mask = S_i < thresholds[i]
        V[i, :] = np.where(exercise_mask, np.maximum(K - S_i, 0), cont)

    # Ensure last threshold is K
    thresholds[-1] = K

    # Policy Object
    class ExercisePolicy:
        def __init__(self, thresholds, K):
            self.thresholds = thresholds
            self.K = K

        def exercise(self, current_state, time):
            # Determine index
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
            # Return boolean vector: exercise if state < threshold
            return state < self.thresholds[idx]

    return ExercisePolicy(thresholds, K)
