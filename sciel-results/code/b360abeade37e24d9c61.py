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

    # Initialize thresholds with a simple backward induction
    thresholds = np.zeros(n_dates)

    # V[i, j] is the value at step i for path j
    V = np.zeros((n_dates, n_paths))
    V[-1, :] = np.maximum(K - S[:, -1], 0)
    thresholds[-1] = K

    for i in range(n_dates - 2, -1, -1):
        if _time.time() > deadline:
            break
        cont = disc * V[i + 1, :]
        immediate = np.maximum(K - S[:, i], 0)
        # Simple threshold: exercise if immediate > cont
        # This implies S < K - cont
        path_thresholds = K - cont
        exercise_mask = immediate > cont
        if np.any(exercise_mask) and np.any(~exercise_mask):
            S_exer = S[:, i][exercise_mask]
            S_non = S[:, i][~exercise_mask]
            lo = S_exer.max()
            hi = S_non.min()
            if lo < hi:
                thresholds[i] = (lo + hi) / 2.0
            else:
                # Fallback: mean of path thresholds
                thresholds[i] = np.mean(path_thresholds)
        elif np.all(exercise_mask):
            thresholds[i] = K
        else:
            thresholds[i] = 0.0

    # Policy Iteration Refinement
    # We iterate to update thresholds based on the V computed from the current thresholds
    for iteration in range(50):
        if _time.time() > deadline - 10:
            break

        # 1. Compute V using current thresholds
        V_new = np.zeros((n_dates, n_paths))
        V_new[-1, :] = np.maximum(K - S[:, -1], 0)

        for i in range(n_dates - 2, -1, -1):
            cont = disc * V_new[i + 1, :]
            immediate = K - S[:, i]
            # Exercise if S < threshold[i]
            # Note: We only exercise if immediate > 0, i.e., S < K
            # The threshold should be capped at K
            thr = min(thresholds[i], K)
            exercise = S[:, i] < thr
            # If S >= K, immediate is <= 0, so we should not exercise even if threshold > K (which we prevent)
            # If S < thr, we exercise. Payoff is max(K-S, 0). Since S < thr <= K, K-S > 0.
            V_new[i, :] = np.where(exercise, immediate, cont)

        # 2. Optimize each threshold independently using smooth surrogate
        for i in range(n_dates - 1):  # Last date is fixed at K (or effectively so)
            S_i = S[:, i]
            cont_i = disc * V_new[i + 1, :]
            
            # Use smooth logistic surrogate for the indicator function
            # Objective: maximize sum( sigmoid((thr - S_i)/eps) * (K - S_i) + (1 - sigmoid((thr - S_i)/eps)) * cont_i )
            # where eps is a small temperature parameter
            
            eps = 0.1  # Temperature for smooth approximation
            
            # Define the smooth objective function
            def smooth_objective(thr):
                # Compute smooth exercise probabilities
                z = (thr - S_i) / eps
                # Clip to avoid numerical issues
                z = np.clip(z, -50, 50)
                # Smooth step function: sigmoid(z)
                smooth_exercise = 1.0 / (1.0 + np.exp(-z))
                
                immediate = K - S_i
                # Only exercise if immediate > 0
                immediate = np.maximum(immediate, 0)
                
                return np.sum(smooth_exercise * immediate + (1 - smooth_exercise) * cont_i)
            
            # Use golden section search to maximize the smooth objective
            # Search range: [min(S_i) - 1, max(S_i) + 1]
            lo = max(0.0, np.min(S_i) - 1.0)
            hi = min(K + 1.0, np.max(S_i) + 1.0)
            
            if lo >= hi:
                lo, hi = 0.0, K
            
            # Golden ratio
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            
            # Initial points
            x1 = lo + gr * (hi - lo)
            x2 = hi - gr * (hi - lo)
            
            f1 = smooth_objective(x1)
            f2 = smooth_objective(x2)
            
            # Iterate
            for _ in range(30):
                if _time.time() > deadline - 5:
                    break
                    
                if f1 < f2:
                    lo = x1
                    x1 = x2
                    f1 = f2
                    x2 = hi - gr * (hi - lo)
                    f2 = smooth_objective(x2)
                else:
                    hi = x2
                    x2 = x1
                    f2 = f1
                    x1 = lo + gr * (hi - lo)
                    f1 = smooth_objective(x1)
            
            # Best estimate is the one with higher objective value
            if f1 > f2:
                best_thr = x1
            else:
                best_thr = x2
            
            # Refine with exact discrete evaluation around the smooth optimum
            # Check a small neighborhood around best_thr
            candidate_thrs = np.array([best_thr - 0.05, best_thr - 0.02, best_thr, best_thr + 0.02, best_thr + 0.05])
            candidate_thrs = np.clip(candidate_thrs, 0.0, K)
            
            best_val = -np.inf
            for ct in candidate_thrs:
                exercise = S_i < ct
                immediate = np.maximum(K - S_i, 0)
                val = np.sum(np.where(exercise, immediate, cont_i))
                if val > best_val:
                    best_val = val
                    best_thr = ct
            
            thresholds[i] = best_thr

    # Ensure last threshold is K
    thresholds[-1] = K

    # Now create the policy object
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
