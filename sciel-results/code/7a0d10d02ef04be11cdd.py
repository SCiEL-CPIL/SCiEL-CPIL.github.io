import math
import time

def construct():
    deadline = time.time() + 110  # Leave 10s margin for 120s limit
    
    # 1. Generate primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # 2. Function to compute phi for a segment [start, start+length)
    def compute_phi_segment(start, length):
        # Initialize phi values with the numbers themselves
        phi_vals = [start + i for i in range(length)]
        
        # For each prime, update phi for its multiples
        for p in primes:
            if p > start + length:
                break
            # Find the first multiple of p in the range [start, start+length)
            # first_multiple = ((start + p - 1) // p) * p
            first_multiple = ((start + p - 1) // p) * p
            if first_multiple < start:
                first_multiple += p # Should not happen with ceil logic, but safe
                
            # Iterate through multiples
            for m in range(first_multiple, start + length, p):
                idx = m - start
                if phi_vals[idx] % p == 0:
                    phi_vals[idx] -= phi_vals[idx] // p
                    
        return phi_vals

    # 3. Search for the longest run
    # We know k <= 200. We want to maximize k.
    # We suspect k=200 is possible near 10^12.
    
    # Let's check a segment near 10^12.
    # Size of segment: Let's take 500 to be safe.
    segment_size = 500
    end_limit = 10**12
    
    # Start from the end and go backwards
    # s is the start of the segment. The segment is [s, s+segment_size)
    # We need s + segment_size - 1 <= 10^12
    # So s <= 10^12 - segment_size + 1
    
    best_start = 1
    best_len = 1
    
    # We will check a few segments if the first one fails, but it shouldn't.
    # Let's just check the very last possible segment of size 200 first.
    # If that works, we return 200.
    
    # Actually, let's compute a larger segment to find the max run robustly.
    # If we find a run of 200, we can stop.
    
    s_start = end_limit - segment_size + 1
    if s_start < 1:
        s_start = 1
        segment_size = end_limit - s_start + 1
        
    phi_vals = compute_phi_segment(s_start, segment_size)
    
    # Find longest run of distinct values
    # We are looking for max k <= 200
    max_k_found = 0
    best_s_in_segment = s_start
    
    # Rolling window or simple scan
    # Since segment is small (500), we can do a O(N^2) scan or just check windows of size 200.
    # Better: Iterate and maintain a set.
    
    # We want the max k.
    # Let's just scan for the longest distinct run in the whole segment.
    current_run_start = 0
    seen = set()
    current_len = 0
    
    for i in range(len(phi_vals)):
        val = phi_vals[i]
        if val in seen:
            # Reset
            # We need to find the new start.
            # For simplicity and given small N, we can just restart the set from i.
            # But that's O(N^2). N=500, so 250k ops. Trivial.
            seen = set()
            current_run_start = i
            current_len = 0
        seen.add(val)
        current_len += 1
        
        if current_len > max_k_found:
            max_k_found = current_len
            best_s_in_segment = s_start + current_run_start
            
        # Optimization: if we hit 200, we can break early if we are sure we found the max possible?
        # The max possible is 200.
        if max_k_found == 200:
            break
            
    if max_k_found == 200:
        return {'start': best_s_in_segment, 'length': 200}
        
    # If we didn't find 200 in the last segment, try a segment further back?
    # Or just return what we found.
    # But 200 is extremely likely.
    # Let's add a fallback: if time allows, check another segment.
    
    if time.time() < deadline:
        # Try a segment in the middle? Or just previous one.
        s_start2 = s_start - segment_size
        if s_start2 >= 1:
            phi_vals2 = compute_phi_segment(s_start2, segment_size)
            
            seen = set()
            current_run_start = 0
            current_len = 0
            max_k_found2 = 0
            best_s2 = s_start2
            
            for i in range(len(phi_vals2)):
                val = phi_vals2[i]
                if val in seen:
                    seen = set()
                    current_run_start = i
                    current_len = 0
                seen.add(val)
                current_len += 1
                if current_len > max_k_found2:
                    max_k_found2 = current_len
                    best_s2 = s_start2 + current_run_start
                if max_k_found2 == 200:
                    break
                    
            if max_k_found2 > max_k_found:
                max_k_found = max_k_found2
                best_s_in_segment = best_s2
                
    # Ensure we don't return k > 200 (though logic above caps it effectively by segment size and break)
    if max_k_found > 200:
        max_k_found = 200
        # Adjust start if needed? 
        # If the run was longer than 200, we just take the first 200 of that run.
        # The logic `best_s_in_segment` points to the start of the run.
        # So returning length 200 is valid.
        
    return {'start': best_s_in_segment, 'length': max_k_found}
