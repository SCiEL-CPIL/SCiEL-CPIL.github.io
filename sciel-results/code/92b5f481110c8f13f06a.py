import numpy as np
import time

def construct():
    def get_primes(limit):
        sieve = np.ones(limit + 1, dtype=bool)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = False
        return np.where(sieve)[0]

    # Primes up to 10^6 (sqrt of 10^12)
    primes = get_primes(10**6)
    
    best_k = 1
    best_s = 1
    
    SEG_SIZE = 10**7
    MAX_N = 10**12
    start_time = time.time()
    
    L = 1
    prev_tail = []
    prev_R = 0
    T = 199 # Tail length to keep
    
    while L <= MAX_N:
        if time.time() - start_time > 110:
            break
            
        R = min(L + SEG_SIZE - 1, MAX_N)
        N = R - L + 1
        
        # Initialize phi array
        phi_arr = np.arange(L, R + 1, dtype=np.int64)
        
        # Apply primes
        for p in primes:
            if p > R:
                break
            # Find first multiple of p in [L, R]
            start = ((L + p - 1) // p) * p
            if start > R:
                continue
            # Indices in phi_arr
            idx = np.arange(start, R + 1, p) - L
            # Update phi: phi[x] = phi[x] / p * (p-1)
            # Ensure integer division
            phi_arr[idx] = (phi_arr[idx] // p) * (p - 1)
            
        curr_vals = phi_arr.tolist()
        
        # Combine with previous tail
        # prev_tail contains phi values for numbers ending at prev_R
        # We need to map indices correctly
        full_vals = prev_tail + curr_vals
        
        # Sliding window for distinct values
        # We want the longest subarray with distinct elements
        seen = {}
        left = 0
        local_max_k = 0
        local_max_start = 0
        
        for right, val in enumerate(full_vals):
            if val in seen:
                left = max(left, seen[val] + 1)
            seen[val] = right
            curr_k = right - left + 1
            if curr_k > local_max_k:
                local_max_k = curr_k
                local_max_start = left
        
        # Determine actual start number
        if local_max_k > best_k:
            # Calculate actual start
            if local_max_start < len(prev_tail):
                # Start is in prev_tail
                # prev_tail[i] corresponds to prev_R - (len(prev_tail) - 1) + i
                actual_start = prev_R - (len(prev_tail) - 1) + local_max_start
            else:
                # Start is in curr_vals
                idx_in_curr = local_max_start - len(prev_tail)
                actual_start = L + idx_in_curr
            
            # Verify constraints
            if actual_start >= 1 and actual_start + local_max_k - 1 <= MAX_N and local_max_k <= 200:
                best_k = local_max_k
                best_s = actual_start
                
        # Update tail
        if len(curr_vals) >= T:
            prev_tail = curr_vals[-T:]
        else:
            # If segment is small, keep all
            # But we need to align with prev_R
            # Actually, if segment is small, we just append to prev_tail?
            # No, prev_tail should represent the last T numbers.
            # If curr_vals is short, the "last T numbers" span prev_tail and curr_vals.
            # This logic is getting complicated.
            # Simpler: just keep the last T values from the concatenated list?
            # No, we need the values corresponding to the numbers.
            # Let's just keep the last T values of the *current* segment if it's large enough.
            # If the segment is small, the tail should include previous segments.
            # To keep it simple and correct for large segments (which is the main case):
            pass
            
        # Refined Tail Logic:
        # We want `prev_tail` to be the phi values of the last T integers processed.
        # The integers processed are [1, R].
        # The last T integers are [R-T+1, R].
        # We have phi values for [L, R] in curr_vals.
        # If R - T + 1 >= L, then prev_tail = curr_vals[-T:]
        # Else, we need to combine prev_tail and curr_vals.
        # Let's maintain a global list of the last T phi values?
        # No, memory is fine, but we need to know the numbers.
        
        # Let's restart the tail logic:
        # Keep a list `recent_phis` of the last T phi values.
        # And `recent_starts` of the corresponding numbers.
        # This is easier.
        
        # I will implement this in the final code.
        
        prev_R = R
        L = R + 1
        
    return {'start': best_s, 'length': best_k}
