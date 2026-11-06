import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # 1. Precompute APs
    # An AP is defined by start and step.
    # We consider all APs with length >= 2. Length 1 APs have sum +/-1, so max is 1.
    # If we achieve max 1, we are optimal.
    
    aps_indices = []
    aps_containing = [[] for _ in range(N)]
    
    for d in range(1, N + 1):
        for start in range(0, N - d + 1):
            terms = list(range(start, N, d))
            if len(terms) > 1:
                ap_id = len(aps_indices)
                aps_indices.append(np.array(terms, dtype=np.int32))
                for idx in terms:
                    aps_containing[idx].append(ap_id)
    
    # Convert aps_containing to numpy arrays for fast indexing
    aps_containing_np = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in aps_containing]
    
    num_aps = len(aps_indices)
    
    # 2. Helper functions
    
    def compute_ap_sums(signs):
        sums = np.zeros(num_aps, dtype=np.int16)
        for ap_idx, indices in enumerate(aps_indices):
            sums[ap_idx] = np.sum(signs[indices])
        return sums

    def get_max_abs(sums):
        return np.max(np.abs(sums))

    # 3. Greedy Initialization
    def greedy_init():
        signs = np.zeros(N, dtype=np.int8)
        ap_sums = np.zeros(num_aps, dtype=np.int16)
        
        for i in range(N):
            best_sign = 1
            best_max = N + 1
            best_sums = None
            
            for s in [1, -1]:
                # Calculate new sums
                new_sums = ap_sums.copy()
                # Update sums for APs containing i
                ap_ids = aps_containing_np[i]
                if len(ap_ids) > 0:
                    new_sums[ap_ids] += s
                
                m = get_max_abs(new_sums)
                if m < best_max:
                    best_max = m
                    best_sign = s
                    best_sums = new_sums
            
            signs[i] = best_sign
            ap_sums = best_sums
            
        return signs, ap_sums, get_max_abs(ap_sums)

    # 4. Local Search with Simulated Annealing
    def local_search(signs, ap_sums, current_max, time_limit):
        best_signs = signs.copy()
        best_max = current_max
        
        T = 5.0  # Initial temperature
        cool_rate = 0.999
        min_T = 0.01
        
        while time.time() < time_limit and T > min_T:
            # Pick a random index to flip
            i = random.randint(0, N - 1)
            
            # Calculate delta
            delta = -2 * signs[i]
            
            # Get affected APs
            ap_ids = aps_containing_np[i]
            
            if len(ap_ids) == 0:
                # No APs affected, flip is free
                signs[i] = -signs[i]
                continue
            
            # Apply change
            old_sums = ap_sums[ap_ids].copy()
            ap_sums[ap_ids] += delta
            
            new_max = get_max_abs(ap_sums)
            
            # Acceptance criterion
            accepted = False
            if new_max < current_max:
                accepted = True
            elif new_max == current_max:
                # Accept to move around plateau
                if random.random() < 0.05:
                    accepted = True
            else:
                # new_max > current_max
                delta_e = new_max - current_max
                if random.random() < np.exp(-delta_e / T):
                    accepted = True
            
            if accepted:
                signs[i] = -signs[i]
                current_max = new_max
                
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            else:
                # Revert
                ap_sums[ap_ids] = old_sums
            
            T *= cool_rate
            
        return best_signs, best_max

    # 5. Main Loop
    best_val = N + 1
    best_signs = None
    
    # Run multiple restarts
    while time.time() < deadline - 5.0:
        # Greedy Init
        signs, ap_sums, current_max = greedy_init()
        
        if current_max < best_val:
            best_val = current_max
            best_signs = signs.copy()
            
        if best_val <= 2:
            break
            
        # Local Search
        time_left = deadline - 5.0 - time.time()
        if time_left <= 0:
            break
            
        # Run SA for a portion of the time
        sa_time = max(1.0, time_left * 0.4)
        signs, best_max_ls = local_search(signs, ap_sums, current_max, time.time() + sa_time)
        
        if best_max_ls < best_val:
            best_val = best_max_ls
            best_signs = signs.copy()
            
    return best_signs.tolist()
