import numpy as np
import time

N = 400

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initialize with random signs
    s = np.random.choice([-1, 1], size=N)
    
    # Calculate initial autocorrelation R(k) for k=0..N-1
    # R(k) = sum_{i=0}^{N-1-k} s[i] * s[i+k]
    R = np.zeros(N)
    for k in range(N):
        R[k] = np.dot(s[:N-k], s[k:])
        
    # Objective: Minimize max(|R(k)|) for k > 0
    current_max_sidelobe = np.max(np.abs(R[1:]))
    
    # Helper to update R when flipping s[i]
    # Returns new R and new max sidelobe
    def flip_and_check(s, i, R):
        new_s = s.copy()
        new_s[i] = -new_s[i]
        
        # Update R efficiently
        new_R = R.copy()
        old_si = s[i]
        
        for k in range(1, N):
            delta = 0
            # Pair (i, i+k)
            if i + k < N:
                delta -= 2 * old_si * s[i+k]
            # Pair (i-k, i)
            if i - k >= 0:
                delta -= 2 * s[i-k] * old_si
            
            new_R[k] = R[k] + delta
            # Note: R[-k] is same as R[k] in magnitude, but we only store k>0
            # The max sidelobe is over k=1..N-1
            
        new_max = np.max(np.abs(new_R[1:]))
        return new_s, new_R, new_max

    # Local Search
    # We use a simple greedy approach with random restarts if stuck
    # Since N=400, one pass is ~400 flips. Each flip is O(N). Total ~160k ops per pass.
    # 115 seconds is plenty for thousands of passes.
    
    best_s = s.copy()
    best_R = R.copy()
    best_max = current_max_sidelobe
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        improved = False
        # Random order of indices to break symmetry
        indices = np.random.permutation(N)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            new_s, new_R, new_max = flip_and_check(s, i, R)
            
            if new_max < current_max_sidelobe:
                # Accept
                s = new_s
                R = new_R
                current_max_sidelobe = new_max
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        if improved:
            # Update best
            best_s = s.copy()
            best_R = R.copy()
            best_max = current_max_sidelobe
        else:
            # If no improvement in a full pass, try a few random restarts or perturbations
            # For now, just break or do a random restart
            if no_improve_count > max_no_improve:
                # Random restart
                s = np.random.choice([-1, 1], size=N)
                for k in range(N):
                    R[k] = np.dot(s[:N-k], s[k:])
                current_max_sidelobe = np.max(np.abs(R[1:]))
                no_improve_count = 0
                # Check if this restart is better
                if current_max_sidelobe < best_max:
                    best_s = s.copy()
                    best_R = R.copy()
                    best_max = current_max_sidelobe

    return best_s

# EVOLVE-BLOCK-END
