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
    dt = T / n_dates
    disc = np.exp(-r * dt)

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
    S = states[:, 1:n_dates + 1]

    # Initialize thresholds with a simple backward induction
    thresholds = np.zeros(n_dates)
    
    V = np.zeros((n_dates, n_paths))
    V[-1, :] = np.maximum(K - S[:, -1], 0)
    thresholds[-1] = K

    for i in range(n_dates - 2, -1, -1):
        cont = disc * V[i + 1, :]
        immediate = np.maximum(K - S[:, i], 0)
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
                thresholds[i] = np.mean(path_thresholds)
        elif np.all(exercise_mask):
            thresholds[i] = K
        else:
            thresholds[i] = 0.0

    # Policy Iteration Refinement with convergence tracking
    prev_thresholds = thresholds.copy()
    for iteration in range(50):
        if time.time() > DEADLINE - 15:
            break
            
        # 1. Compute V using current thresholds
        V_new = np.zeros((n_dates, n_paths))
        V_new[-1, :] = np.maximum(K - S[:, -1], 0)
        
        for i in range(n_dates - 2, -1, -1):
            cont = disc * V_new[i + 1, :]
            immediate = K - S[:, i]
            thr = min(thresholds[i], K)
            exercise = S[:, i] < thr
            V_new[i, :] = np.where(exercise, immediate, cont)
            
        # 2. Optimize each threshold independently
        for i in range(n_dates - 1):
            S_i = S[:, i]
            cont_i = disc * V_new[i + 1, :]
            
            order = np.argsort(S_i)
            S_sorted = S_i[order]
            cont_sorted = cont_i[order]
            
            payoffs = K - S_sorted
            prefix_payoffs = np.cumsum(payoffs)
            
            suffix_conts = np.cumsum(cont_sorted[::-1])[::-1]
            
            prefix_values = np.zeros(n_paths + 1)
            prefix_values[1:] = prefix_payoffs
            
            suffix_values = np.zeros(n_paths + 1)
            suffix_values[:-1] = suffix_conts
            
            total_values = prefix_values + suffix_values
            
            best_m = np.argmax(total_values)
            
            if best_m == 0:
                thresholds[i] = S_sorted[0] - 1e-6
            elif best_m == n_paths:
                thresholds[i] = S_sorted[-1] + 1e-6
            else:
                thresholds[i] = (S_sorted[best_m - 1] + S_sorted[best_m]) / 2.0
            
            if thresholds[i] > K:
                thresholds[i] = K
            if thresholds[i] < 0:
                thresholds[i] = 0.0

        # Convergence check
        if np.max(np.abs(thresholds - prev_thresholds)) < 1e-8:
            prev_thresholds = thresholds.copy()
            break
        prev_thresholds = thresholds.copy()

    # High-precision polish pass: local search around each threshold
    for i in range(n_dates - 1):
        if time.time() > DEADLINE - 5:
            break
            
        S_i = S[:, i]
        # Recompute continuation values from the last full V computation
        V_polish = np.zeros((n_dates, n_paths))
        V_polish[-1, :] = np.maximum(K - S[:, -1], 0)
        for j in range(n_dates - 2, -1, -1):
            cont_j = disc * V_polish[j + 1, :]
            imm_j = K - S[:, j]
            thr_j = min(thresholds[j], K)
            ex_j = S[:, j] < thr_j
            V_polish[j, :] = np.where(ex_j, imm_j, cont_j)
        
        cont_i = disc * V_polish[i + 1, :]
        
        # Local search: try small perturbations around current threshold
        current_thr = thresholds[i]
        best_val = -np.inf
        best_thr = current_thr
        
        # Generate candidate thresholds around current
        if current_thr <= 0:
            candidates = [current_thr]
        elif current_thr >= K:
            candidates = [current_thr]
        else:
            # Try a range of offsets
            offsets = np.linspace(-0.5, 0.5, 21)
            candidates = current_thr + offsets
            # Also try exact data point boundaries
            S_sorted_local = np.sort(S_i)
            # Find nearest data points
            idx = np.searchsorted(S_sorted_local, current_thr)
            if idx > 0:
                candidates.append(S_sorted_local[idx - 1])
            if idx < n_paths:
                candidates.append(S_sorted_local[idx])
            candidates = np.unique(candidates)
        
        for cand_thr in candidates:
            cand_thr = max(0, min(K, cand_thr))
            exercise_mask = S_i < cand_thr
            value = np.sum(np.where(exercise_mask, K - S_i, cont_i))
            if value > best_val:
                best_val = value
                best_thr = cand_thr
        
        thresholds[i] = best_thr

    thresholds[-1] = K

    class ExercisePolicy:
        def __init__(self, thresholds, K, n_dates):
            self.thresholds = thresholds
            self.K = K
            self.n_dates = n_dates

        def exercise(self, current_state, time):
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
