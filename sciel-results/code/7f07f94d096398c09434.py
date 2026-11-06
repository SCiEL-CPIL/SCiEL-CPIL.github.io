"""Low-discrepancy on arithmetic progressions via simulated annealing with efficient delta computation."""
# EVOLVE-BLOCK-START
import time
import numpy as np

N = 256

def construct():
    t0 = time.time()
    deadline = t0 + 55.0  # safety margin before 60s limit
    
    # Precompute APs efficiently: for each (start, step), store the list of indices
    # Use a more memory-efficient representation
    aps = []
    for step in range(1, N + 1):
        for start in range(1, N + 1):
            # Generate indices for this AP
            idx = []
            pos = start - 1
            while pos < N:
                idx.append(pos)
                pos += step
            if idx:
                aps.append(idx)
    
    num_aps = len(aps)
    
    # Build position-to-AP mapping for efficient updates
    pos_to_aps = [[] for _ in range(N)]
    for ap_idx, idx in enumerate(aps):
        for i in idx:
            pos_to_aps[i].append(ap_idx)
    
    # Compute initial sums for a given sign configuration
    def compute_sums(signs):
        sums = np.zeros(num_aps, dtype=np.int32)
        for i in range(N):
            s = signs[i]
            for ap_idx in pos_to_aps[i]:
                sums[ap_idx] += s
        return sums
    
    # Get max absolute sum
    def get_max_abs(sums):
        return int(np.max(np.abs(sums)))
    
    # Simulated annealing
    best_signs = None
    best_obj = 10**9
    
    rng = np.random.default_rng(171)
    
    # Try multiple restarts
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() >= deadline:
            break
        
        # Random initialization
        signs = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
        sums = compute_sums(signs)
        cur_obj = get_max_abs(sums)
        
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_signs = signs.copy()
        
        # SA parameters
        T = 5.0
        T_min = 0.01
        cooling = 0.999
        steps_per_restart = 2000
        
        for step in range(steps_per_restart):
            if time.time() >= deadline:
                break
            
            # Pick random position to flip
            i = int(rng.integers(N))
            old_sign = signs[i]
            affected = pos_to_aps[i]
            
            # Compute delta: flipping sign changes sum by -2*old_sign for each affected AP
            delta = -2 * old_sign
            
            # Update sums
            for ap_idx in affected:
                sums[ap_idx] += delta
            
            # Update sign
            signs[i] = -old_sign
            
            # Compute new objective
            new_obj = get_max_abs(sums)
            
            # Acceptance criterion
            d = new_obj - cur_obj
            if d <= 0 or (T > T_min and rng.random() < np.exp(-d / T)):
                cur_obj = new_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_signs = signs.copy()
            else:
                # Revert
                signs[i] = old_sign
                for ap_idx in affected:
                    sums[ap_idx] -= delta
            
            T *= cooling
            if T < T_min:
                break
    
    if best_signs is None:
        best_signs = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
    
    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
