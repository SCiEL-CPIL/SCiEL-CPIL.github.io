import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial guess: all ones
    x = np.ones(N, dtype=int)
    
    # Calculate initial autocorrelation
    # R[tau] = sum x[i] x[i+tau]
    # Use FFT for initial calc? Or direct?
    # Direct is O(N^2). 160k ops. Fast.
    
    def get_autocorr(x):
        # x is 1D array of +/- 1
        # R[0] is not needed, we look at 1..N-1
        R = np.zeros(N)
        for tau in range(1, N):
            # np.dot(x[:N-tau], x[tau:])
            R[tau] = np.dot(x[:N-tau], x[tau:])
        return R

    R = get_autocorr(x)
    current_min = np.min(R[1:])
    current_val = abs(current_min)
    best_x = x.copy()
    best_val = current_val
    
    # Local Search
    # We will iterate through indices, trying to flip.
    # To speed up, we can maintain R and update it.
    
    # Helper to update R when flipping index k
    # Delta R[tau] = -2 * x[k] * (x[k+tau] + x[k-tau])
    # Note: boundaries.
    
    def flip_and_update(k, x, R):
        # Returns new R, new x
        # x is modified in place? No, create new or copy.
        # For efficiency, we can calculate delta and apply.
        old_xk = x[k]
        # Calculate changes for all tau
        # tau ranges 1 to N-1
        # Term 1: x[k] * x[k+tau] exists if k+tau < N
        # Term 2: x[k-tau] * x[k] exists if k-tau >= 0
        
        # Vectorized update:
        # Create array of changes
        changes = np.zeros(N)
        
        # Forward shifts: tau such that k+tau < N => tau < N-k
        # R[tau] includes x[k]x[k+tau]
        # New term is -x[k]x[k+tau]. Change is -2 x[k]x[k+tau]
        # Indices in R: 1 to N-k-1
        if k < N-1:
             taus_f = np.arange(1, N-k)
             # x[k+tau] corresponds to x[k+1], x[k+2]...
             vals_f = x[k+1:k+N-k] # This slice is x[k+1] to x[N-1]
             # Wait, x[k+tau] for tau=1 is x[k+1].
             # The slice x[k+1:N] has length N-k-1.
             # taus_f has length N-k-1.
             changes[taus_f] += -2 * old_xk * vals_f
             
        # Backward shifts: tau such that k-tau >= 0 => tau <= k
        # R[tau] includes x[k-tau]x[k]
        # New term is -x[k-tau]x[k]. Change is -2 x[k-tau]x[k]
        if k > 0:
             taus_b = np.arange(1, k+1)
             # x[k-tau] for tau=1 is x[k-1].
             # We need x[k-1], x[k-2]... x[0].
             # Slice x[0:k] reversed?
             # x[k-1] is at index k-1.
             # vals_b = x[k-1::-1] ?
             # taus_b[0]=1 -> x[k-1]
             # taus_b[1]=2 -> x[k-2]
             # So vals_b[i] = x[k-1-i]
             vals_b = x[k-1::-1]
             changes[taus_b] += -2 * old_xk * vals_b
             
        R_new = R + changes
        x_new = x.copy()
        x_new[k] = -old_xk
        return x_new, R_new

    # Main Loop
    no_improvement_count = 0
    while time.time() < deadline:
        improved = False
        # Randomize order of indices to avoid bias
        indices = np.random.permutation(N)
        
        for k in indices:
            if time.time() > deadline:
                break
            
            x_trial, R_trial = flip_and_update(k, x, R)
            min_trial = np.min(R_trial[1:])
            val_trial = abs(min_trial)
            
            # Acceptance criteria
            if val_trial < current_val - 1e-9:
                x = x_trial
                R = R_trial
                current_min = min_trial
                current_val = val_trial
                improved = True
                no_improvement_count = 0
            elif val_trial == current_val:
                # Tie breaker: prefer flatter spectrum (lower sum of squares of positive parts?)
                # Or just random accept with low prob?
                # Let's check sum of squares of all R[1:]
                # If it decreases, it's generally better for future moves?
                # Actually, if val is same, we are stuck.
                # Let's try to accept if it reduces the max autocorr?
                # For now, accept 10% of the time to escape.
                if np.random.rand() < 0.05:
                    x = x_trial
                    R = R_trial
                    current_min = min_trial
                    # val_trial is same
            else:
                no_improvement_count += 1
        
        if not improved:
             # If we made a full pass with no improvement, we might be stuck.
             # Try a random restart or perturbation?
             # Given the time budget, a simple random perturbation of a few bits might help.
             if no_improvement_count > 100:
                 # Perturb
                 for _ in range(5):
                     k = np.random.randint(N)
                     x, R = flip_and_update(k, x, R)
                 no_improvement_count = 0
                 
        # Update best
        if current_val < best_val:
            best_val = current_val
            best_x = x.copy()
            
    return best_x
