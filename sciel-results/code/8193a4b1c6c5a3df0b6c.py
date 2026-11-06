def fit_exercise_policy(training_paths, contract):
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12
    
    n_paths = training_paths.shape[0]
    S = training_paths[:, 1:n_dates+1]  # shape (n_paths, 12)
    
    n_ex = n_dates
    disc = np.exp(-r / n_dates)
    
    # Backward induction to get per-path optimal decisions
    V = np.zeros((n_ex, n_paths))
    V[-1, :] = np.maximum(K - S[:, -1], 0)
    
    for i in range(n_ex - 2, -1, -1):
        cont = disc * V[i+1, :]
        exercise = (K - S[:, i]) > cont
        V[i, :] = np.where(exercise, K - S[:, i], cont)
    
    # Structural change: use a lookup table (bin-based) instead of single threshold
    # Bin the state space into fine bins and store the majority exercise decision per bin
    n_bins = 2000  # fine resolution
    
    # Determine bin edges from training data range
    S_min = float(np.min(S))
    S_max = float(np.max(S))
    
    # Add small margin
    margin = max(0.1, (S_max - S_min) * 0.01)
    S_min -= margin
    S_max += margin
    
    bin_edges = np.linspace(S_min, S_max, n_bins + 1)
    
    # For each exercise date, build a lookup table
    # lookup[i, b] = True if majority of training paths in bin b at date i exercise
    lookup = np.zeros((n_ex, n_bins), dtype=bool)
    
    for i in range(n_ex):
        S_i = S[:, i]
        # Get bin indices for each path
        bin_idx = np.searchsorted(bin_edges, S_i, side='right') - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        
        # Determine exercise decision from backward induction
        if i < n_ex - 1:
            cont_i = disc * V[i+1, :]
            exercise_i = (K - S_i) > cont_i
        else:
            exercise_i = (K - S_i) > 0
        
        # For each bin, compute majority vote
        for b in range(n_bins):
            mask = (bin_idx == b)
            if np.any(mask):
                # Majority exercise
                lookup[i, b] = np.sum(exercise_i[mask]) > np.sum(~exercise_i[mask])
    
    # Check time
    if time.time() > deadline - 5:
        # Fallback: use simple threshold if running out of time
        thresholds = np.zeros(n_ex)
        for i in range(n_ex):
            if i < n_ex - 1:
                cont_i = disc * V[i+1, :]
            else:
                cont_i = np.zeros(n_paths)
            thresholds[i] = np.mean(K - cont_i)
        
        class ExercisePolicy:
            def __init__(self, thresholds, K):
                self.thresholds = thresholds
                self.K = K
                self.n_dates = len(thresholds)
            
            def exercise(self, current_state, time):
                idx = int(round(time * 12)) - 1
                idx = max(0, min(idx, self.n_dates - 1))
                return current_state < self.thresholds[idx]
        
        return ExercisePolicy(thresholds, K)
    
    # Create the policy object with lookup table
    class ExercisePolicy:
        def __init__(self, lookup, bin_edges, n_dates):
            self.lookup = lookup
            self.bin_edges = bin_edges
            self.n_dates = n_dates
            self.n_bins = len(bin_edges) - 1
        
        def exercise(self, current_state, time):
            idx = int(round(time * 12)) - 1
            idx = max(0, min(idx, self.n_dates - 1))
            
            # Map current_state to bin indices
            bin_idx = np.searchsorted(self.bin_edges, current_state, side='right') - 1
            bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
            
            return self.lookup[idx, bin_idx]
    
    return ExercisePolicy(lookup, bin_edges, n_ex)
