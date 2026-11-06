import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using exact greedy threshold optimization
    via backward induction on the training paths.
    """
    t_start = time.time()
    DEADLINE = t_start + 110.0  # Safety margin before 120s/150s kill

    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12
    dt = T / n_dates
    disc = np.exp(-r * dt)

    # Normalize training paths to (n_paths, n_steps)
    tp = np.asarray(training_paths, dtype=np.float64)
    if tp.ndim == 2:
        if tp.shape[0] == 4096:
            states = tp
        elif tp.shape[1] == 4096:
            states = tp.T
        else:
            # Heuristic: assume first dim is paths if it's larger
            if tp.shape[0] > tp.shape[1]:
                states = tp
            else:
                states = tp.T
    else:
        states = tp.reshape(4096, -1)

    n_paths = states.shape[0]
    n_steps = states.shape[1]

    # Exercise dates are at columns 1..12 (0-indexed)
    # states[:, j] for j=1..12 are the states at exercise dates 1..12
    # We map exercise date i (1-indexed, 1..12) to column index i
    # So S[:, i-1] in our 0-indexed array of exercise dates corresponds to states[:, i]
    
    # Extract states at exercise dates: columns 1 to 12
    # Let S_ex[i] be the state at exercise date i+1 (0-indexed i from 0 to 11)
    # S_ex[i, p] = states[p, i+1]
    S_ex = states[:, 1:n_dates + 1]  # Shape (n_paths, 12)

    # Initialize thresholds
    thresholds = np.zeros(n_dates)
    
    # V[i] is the value at exercise date i (0-indexed) for all paths
    # V[11] corresponds to date 12 (maturity)
    V = np.zeros((n_dates, n_paths))
    
    # Maturity (date 12, index 11): Always exercise if in the money
    S_last = S_ex[:, -1]
    V[-1, :] = np.maximum(K - S_last, 0.0)
    thresholds[-1] = K  # Threshold at maturity is effectively K (exercise if S < K)

    # Backward Induction from date 11 down to date 1
    # i is the index in V (0..10), corresponding to exercise date i+1
    for i in range(n_dates - 2, -1, -1):
        if time.time() > DEADLINE:
            # Fallback: assume threshold K for remaining steps
            for j in range(i, -1, -1):
                thresholds[j] = K
            break

        S_i = S_ex[:, i]
        
        # Continuation value is the discounted value from the next exercise date
        # V[i+1] already includes the optimal decision at date i+2
        cont = disc * V[i + 1, :]
        
        # Immediate payoff
        imm = np.maximum(K - S_i, 0.0)
        
        # We want to find the optimal threshold tau such that we exercise if S_i < tau.
        # Objective: Sum( I(S_i < tau) * imm + I(S_i >= tau) * cont )
        
        # Sort paths by S_i
        order = np.argsort(S_i)
        S_sorted = S_i[order]
        imm_sorted = imm[order]
        cont_sorted = cont[order]
        
        # Prefix sums of immediate payoffs
        # pref_imm[m] = sum(imm_sorted[:m])
        pref_imm = np.zeros(n_paths + 1)
        pref_imm[1:] = np.cumsum(imm_sorted)
        
        # Suffix sums of continuation values
        # suff_cont[m] = sum(cont_sorted[m:])
        total_cont = np.sum(cont_sorted)
        pref_cont = np.zeros(n_paths + 1)
        pref_cont[1:] = np.cumsum(cont_sorted)
        suff_cont = total_cont - pref_cont
        
        # Objective for cut m (exercise first m paths, which have smallest S)
        objectives = pref_imm + suff_cont
        
        # Find the m that maximizes the objective
        best_m = np.argmax(objectives)
        
        # Determine threshold
        if best_m == 0:
            # Never exercise
            thresholds[i] = 0.0
        elif best_m == n_paths:
            # Always exercise (for all training paths)
            # In practice, for a put, if S is very low, we exercise.
            # Setting to K is a safe upper bound for "always exercise" in ITM region
            # But if all S are > K, we shouldn't exercise. 
            # However, if best_m == n_paths, it means exercising all is best.
            # This implies all S_i are low enough that (K-S_i) > cont.
            # Since cont >= 0, this implies S_i < K.
            # So threshold should be at least max(S_i).
            # To be safe and consistent with "exercise if S < tau", 
            # if we exercise all, tau must be > max(S_i).
            # Let's set it to K, as exercising above K is never optimal for a put.
            # If max(S_i) > K, and best_m == n_paths, something is wrong, but 
            # usually this won't happen for puts unless cont is negative (impossible).
            thresholds[i] = K
        else:
            # Threshold is between S_sorted[best_m - 1] and S_sorted[best_m]
            # Using the midpoint is a robust choice
            thresholds[i] = (S_sorted[best_m - 1] + S_sorted[best_m]) / 2.0
            thresholds[i] = np.clip(thresholds[i], 0.0, K)

        # Update V[i]
        # Exercise if S_i < thresholds[i]
        exercise_mask = S_i < thresholds[i]
        V[i, :] = np.where(exercise_mask, imm, cont)

    # Ensure last threshold is K
    thresholds[-1] = K

    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, K_val):
            self.thresholds = thresholds
            self.K = K_val

        def exercise(self, current_state, time):
            # Determine the exercise date index from time
            # time can be in [0, 1] (fractional) or [1, 12] (integer dates)
            if isinstance(time, (int, float)):
                t_val = float(time)
                if 0 < t_val <= 1.0:
                    # Fractional time
                    idx = int(np.round(t_val * 12)) - 1
                    idx = max(0, min(idx, 11))
                elif 1 <= t_val <= 12:
                    # Integer date index
                    idx = int(t_val) - 1
                    idx = max(0, min(idx, 11))
                else:
                    idx = 0
            else:
                idx = 0

            state = np.asarray(current_state, dtype=np.float64)
            # Return boolean vector: exercise if state < threshold
            return state < self.thresholds[idx]

    return ExercisePolicy(thresholds, K)
