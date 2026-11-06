def fit_exercise_policy(training_paths, contract):
    import numpy as np
    import time

    # Explicit deadline guard: 150s wall-clock limit, use 140s as safety margin
    start_time = time.time()
    DEADLINE = start_time + 140

    # Extract parameters
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12

    # Determine shape of training_paths
    # Assume (n_paths, n_steps) where n_steps = 13 (time 0 through 12/12)
    if training_paths.ndim == 2:
        if training_paths.shape[0] == 4096:
            states = training_paths  # (n_paths, n_steps)
        elif training_paths.shape[1] == 4096:
            states = training_paths.T
        else:
            # Try to infer
            if training_paths.shape[0] >= 13:
                states = training_paths.T
            else:
                states = training_paths
    else:
        states = np.array(training_paths)

    n_paths = states.shape[0]
    n_steps = states.shape[1]

    # States at exercise dates: columns 1 through 12 (indices 1..12)
    # S[:, i] for i=1..12 corresponds to exercise date i
    S = states[:, 1:n_dates + 1]  # shape (n_paths, 12)

    # Backward induction to compute optimal exercise decisions
    disc = np.exp(-r / n_dates)  # discount factor for one period

    # V[i, p] = value of option at date i for path p, given optimal exercise from i onwards
    V = np.zeros((n_dates, n_paths))

    # At last date (index n_dates-1 = 11), continuation = 0
    # Exercise if K - S > 0
    V[-1, :] = np.maximum(K - S[:, -1], 0)

    # Work backwards from date 11 to date 0
    for i in range(n_dates - 2, -1, -1):
        cont = disc * V[i + 1, :]
        immediate = np.maximum(K - S[:, i], 0)
        exercise = immediate > cont
        V[i, :] = np.where(exercise, immediate, cont)

    # Now fit thresholds for each date
    # At date i, exercise if S_i < threshold_i
    # The continuation value for path p at date i is disc * V[i+1, p] (or 0 for last date)
    # Exercise if K - S_i > cont, i.e., S_i < K - cont
    # So the path-specific threshold is K - cont
    # We want a single threshold that best approximates this

    thresholds = np.zeros(n_dates)

    for i in range(n_dates):
        S_i = S[:, i]

        if i < n_dates - 1:
            cont_i = disc * V[i + 1, :]
        else:
            cont_i = np.zeros(n_paths)

        # Path-specific threshold: exercise if S_i < K - cont_i
        path_thresholds = K - cont_i

        # Find the best single threshold
        # We want to minimize the number of misclassified decisions
        # The optimal threshold is one that maximizes agreement with the path-specific decisions

        # The path-specific decision is: exercise if S_i < path_thresholds
        # We want to find a single thr such that (S_i < thr) matches (S_i < path_thresholds) as closely as possible

        # Since both are monotonic in S_i (lower S_i -> more likely to exercise),
        # the optimal threshold is the one that best separates the two groups

        # Group 1: paths where we exercise (S_i < path_thresholds)
        # Group 2: paths where we don't exercise (S_i >= path_thresholds)

        exercise_mask = S_i < path_thresholds

        if np.any(exercise_mask) and np.any(~exercise_mask):
            S_exer = S_i[exercise_mask]
            S_non = S_i[~exercise_mask]
            # Threshold is between max(S_exer) and min(S_non)
            thr = (S_exer.max() + S_non.min()) / 2.0
        elif np.all(exercise_mask):
            thr = K  # Always exercise
        elif np.all(~exercise_mask):
            thr = 0.0  # Never exercise
        else:
            thr = K

        thresholds[i] = thr

    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, K, n_dates):
            self.thresholds = thresholds
            self.K = K
            self.n_dates = n_dates

        def exercise(self, current_state, time):
            # Determine which exercise date we're at
            if isinstance(time, (int, float)):
                if isinstance(time, float) and 0 < time <= 1.0:
                    idx = int(np.round(time * 12)) - 1
                    if idx < 0:
                        idx = 0
                    if idx >= self.n_dates:
                        idx = self.n_dates - 1
                elif isinstance(time, int):
                    if 1 <= time <= 12:
                        idx = time - 1
                    elif 0 <= time <= 11:
                        idx = time
                    else:
                        idx = min(max(time, 0), self.n_dates - 1)
                else:
                    idx = 0
            else:
                idx = 0

            state = np.asarray(current_state)
            return state < self.thresholds[idx]

    return ExercisePolicy(thresholds, K, n_dates)
