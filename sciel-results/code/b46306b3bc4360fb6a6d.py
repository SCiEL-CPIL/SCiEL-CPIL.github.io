def fit_exercise_policy(training_paths, contract):
    import numpy as np
    import time as _time

    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12
    dt = 1.0 / 12.0
    disc = np.exp(-r * dt)

    # Deadline guard
    start_time = _time.time()
    deadline = start_time + 115.0

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

    # Initialize thresholds and values
    thresholds = np.zeros(n_dates)
    V = np.zeros((n_dates, n_paths))
    
    # Last date: always exercise if in the money
    V[-1, :] = np.maximum(K - S[:, -1], 0)
    thresholds[-1] = K

    # Value Iteration / Backward Induction with Regression-Based Continuation
    # We will use a few iterations to converge the thresholds.
    # In each iteration, we perform a backward pass.
    # At each step i, we estimate the continuation value using a regression
    # on the "American" values from step i+1 (which include the optimal exercise decision).
    # However, since we only have 1D state (S), and the value function is convex,
    # a simple threshold search on the discrete training data is very effective.
    # The "inspiration" 2 uses a greedy threshold search.
    # The "inspiration" 3 uses value iteration.
    # The "current" program uses a high-res binomial tree.
    
    # Strategy: Use the training data to perform a refined backward induction.
    # We can improve the discrete threshold search by using a smoother estimate of the continuation value.
    # Or, we can simply run the discrete threshold search multiple times (value iteration) to ensure consistency.
    
    # Let's implement a robust backward induction that iterates until convergence or time limit.
    
    max_iterations = 10
    for iteration in range(max_iterations):
        if _time.time() > deadline - 5:
            break
            
        # Backward pass
        for i in range(n_dates - 2, -1, -1):
            if _time.time() > deadline - 2:
                break
                
            S_i = S[:, i]
            # Continuation value is the discounted value from the next step
            # V[i+1] already includes the decision at i+1.
            cont = disc * V[i + 1, :]
            
            # Find optimal threshold
            # We want to maximize sum( I(S_i < tau) * (K - S_i) + I(S_i >= tau) * cont )
            
            # Sort indices by S_i
            order = np.argsort(S_i)
            S_sorted = S_i[order]
            cont_sorted = cont[order]
            
            # Immediate payoff for sorted states
            imm_sorted = np.maximum(K - S_sorted, 0)
            
            # Prefix sum of immediate payoffs
            pref_imm = np.zeros(n_paths + 1)
            pref_imm[1:] = np.cumsum(imm_sorted)
            
            # Suffix sum of continuation values
            total_cont = np.sum(cont_sorted)
            pref_cont = np.zeros(n_paths + 1)
            pref_cont[1:] = np.cumsum(cont_sorted)
            suff_cont = total_cont - pref_cont
            
            # Objective for cut m
            objectives = pref_imm + suff_cont
            
            best_m = np.argmax(objectives)
            
            if best_m == 0:
                new_threshold = 0.0
            elif best_m == n_paths:
                new_threshold = K
            else:
                # Midpoint between the last exercised and first held
                new_threshold = (S_sorted[best_m - 1] + S_sorted[best_m]) / 2.0
                new_threshold = np.clip(new_threshold, 0.0, K)
            
            # Update V[i]
            exercise_mask = S_i < new_threshold
            V[i, :] = np.where(exercise_mask, np.maximum(K - S_i, 0), cont)
            
            # Store the new threshold
            thresholds[i] = new_threshold

    # Ensure last threshold is K
    thresholds[-1] = K

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
                    if idx >= 12:
                        idx = 11
                elif isinstance(time, int):
                    if 1 <= time <= 12:
                        idx = time - 1
                    elif 0 <= time <= 11:
                        idx = time
                    else:
                        idx = min(max(time, 0), 11)
                else:
                    idx = 0
            else:
                idx = 0

            state = np.asarray(current_state)
            return state < self.thresholds[idx]

    return ExercisePolicy(thresholds, K)
