import numpy as np
import time

def construct():
    n = 128
    # A is a boolean array of size n+1
    A = np.zeros(n + 1, dtype=bool)
    A[64:129] = True
    
    def get_mults(A):
        # Compute multiplicities for sums 128 to 256
        # conv[S] is the number of ordered pairs (a,b) in A such that a+b=S
        conv = np.convolve(A.astype(np.int32), A.astype(np.int32), mode='full')
        S_range = np.arange(128, 257)
        # Base multiplicity is conv[S] / 2 for unordered pairs
        # But for even S, if S/2 is in A, the pair (S/2, S/2) is counted once in conv,
        # but represents 1 unordered pair.
        # conv[S] counts (a,b) and (b,a).
        # If a != b, there are 2 ordered pairs for 1 unordered pair.
        # If a == b, there is 1 ordered pair for 1 unordered pair.
        # So unordered count = (conv[S] - delta) / 2 + delta = (conv[S] + delta) / 2
        # where delta = 1 if S is even and S/2 in A, else 0.
        
        mults = conv[S_range] / 2.0
        
        even_mask = (S_range % 2 == 0)
        if np.any(even_mask):
            S_even = S_range[even_mask]
            k = S_even // 2
            # k is in range 64..128, which is valid for A
            diag_present = A[k].astype(np.float64)
            mults[even_mask] += diag_present / 2.0
            
        return mults

    current_mults = get_mults(A)
    current_max = np.max(current_mults)
    current_sq_sum = np.sum(current_mults**2)
    
    best_A = A.copy()
    best_max = current_max
    best_sq_sum = current_sq_sum
    
    start_time = time.time()
    timeout = 55  # Safety margin for 60s limit
    
    # Precompute list of indices for A and not A for faster random selection?
    # We can just use np.where each time, it's fast enough for 129 elements.
    
    iterations = 0
    no_improve_count = 0
    
    while time.time() - start_time < timeout:
        # Find hot sums
        max_val = np.max(current_mults)
        hot_sums = np.where(current_mults == max_val)[0]
        S_hot = hot_sums[np.random.randint(len(hot_sums))]
        
        # Find a pair (a, b) in A such that a + b = S_hot
        # a ranges from max(0, S_hot - n) to min(n, S_hot)
        a_start = max(0, S_hot - n)
        a_end = min(n, S_hot)
        
        valid_a = []
        # Iterate through possible a
        for a in range(a_start, a_end + 1):
            b = S_hot - a
            if A[a] and A[b]:
                valid_a.append(a)
                
        if not valid_a:
            continue
            
        a_remove = valid_a[np.random.randint(len(valid_a))]
        
        # Pick a new location a_add
        candidates = np.where(~A)[0]
        if len(candidates) == 0:
            break
        a_add = candidates[np.random.randint(len(candidates))]
        
        # Perform swap
        A[a_remove] = False
        A[a_add] = True
        
        new_mults = get_mults(A)
        
        # Check coverage: all sums must be >= 1
        if np.min(new_mults) < 1:
            A[a_remove] = True
            A[a_add] = False
            continue
        
        new_max = np.max(new_mults)
        new_sq_sum = np.sum(new_mults**2)
        
        # Acceptance criteria
        accept = False
        if new_max < current_max:
            accept = True
        elif new_max == current_max:
            if new_sq_sum < current_sq_sum:
                accept = True
            elif new_sq_sum == current_sq_sum:
                # Accept neutral moves with some probability to explore
                if np.random.rand() < 0.2:
                    accept = True
            
            # If stuck at the same max and sq_sum, try a "kick" occasionally
            # But simple random walk on the plateau is handled by the 0.2 prob.
            
        if accept:
            current_mults = new_mults
            current_max = new_max
            current_sq_sum = new_sq_sum
            no_improve_count = 0 # Reset? Or only reset on max improvement?
            
            if current_max < best_max:
                best_max = current_max
                best_A = A.copy()
                best_sq_sum = current_sq_sum
        else:
            # Revert
            A[a_remove] = True
            A[a_add] = False
            no_improve_count += 1
            
        # If stuck for a long time, try a random kick
        if no_improve_count > 1000:
            # Randomly swap 2 elements
            idx_in = np.where(A)[0]
            idx_out = np.where(~A)[0]
            if len(idx_in) > 0 and len(idx_out) > 0:
                i1 = idx_in[np.random.randint(len(idx_in))]
                i2 = idx_out[np.random.randint(len(idx_out))]
                A[i1] = False
                A[i2] = True
                kick_mults = get_mults(A)
                if np.min(kick_mults) >= 1:
                    current_mults = kick_mults
                    current_max = np.max(current_mults)
                    current_sq_sum = np.sum(current_mults**2)
                else:
                    A[i1] = True
                    A[i2] = False
            no_improve_count = 0

    return list(np.where(best_A)[0])
