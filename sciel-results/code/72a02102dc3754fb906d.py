<<<<<<< SEARCH
import numpy as np
import time
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 10000000
    
    # 1. Compute sigma(n) for all n up to N
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    if time.time() > deadline:
        return [11, 6]
=======
import numpy as np
import time
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 10000000
    
    # 1. Compute sigma(n) for all n up to N using sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    if time.time() > deadline:
        return [11, 6]
    
    # 2. Sort indices by sigma(n)
    n_values = np.arange(1, N + 1, dtype=np.int64)
    s_values = sigma[1:N+1]
    
    # Sort by sigma, then by n
    sort_idx = np.lexsort((n_values, s_values))
    sorted_n = n_values[sort_idx]
    sorted_s = s_values[sort_idx]
    
    if time.time() > deadline:
        return [11, 6]
    
    # 3. Find best pair using vectorized approach
    best_score = float('inf')
    best_a, best_b = 11, 6
    
    i = 0
    while i < N:
        current_sigma = sorted_s[i]
        next_idx = np.searchsorted(sorted_s, current_sigma, side='right')
        
        if next_idx - i >= 2:
            group = sorted_n[i:next_idx]
            n_l = len(group)
            
            if n_l >= 2:
                # Vectorized: for each b in group, compute target_a = 1.5*b
                # We want to find a in group that minimizes |a/b - 3/2| = |2a - 3b| / (2b)
                # So we want a close to 1.5*b
                
                # For each element b in group, find the closest a in group to 1.5*b
                # Use searchsorted for efficiency
                targets = (3 * group).astype(np.int64)  # 3*b
                # We want a such that 2*a is close to 3*b, i.e., a close to 1.5*b
                
                # Find positions where 2*a is closest to 3*b
                # For each b, we want to find a in group minimizing |2a - 3b|
                
                # Create 2*group for comparison
                two_group = 2 * group
                
                # For each target (3*b), find the closest value in two_group
                # Use searchsorted
                pos = np.searchsorted(two_group, targets)
                
                # Check positions and pos-1
                # Clip to valid range
                pos_clipped = np.clip(pos, 1, n_l - 1)
                
                # Get candidate a values
                a_candidates = group[pos_clipped]
                b_candidates = group
                diff1 = np.abs(2 * a_candidates - 3 * b_candidates)
                
                # Also check pos-1
                pos_minus = np.clip(pos - 1, 0, n_l - 1)
                a_candidates2 = group[pos_minus]
                diff2 = np.abs(2 * a_candidates2 - 3 * b_candidates2)
                
                # Combine and find minimum
                all_diffs = np.minimum(diff1, diff2)
                # Mask out cases where a == b (diff would be |2b-3b| = b, not 0)
                # Actually we need a != b, so mask where a_candidates == b_candidates
                mask1 = (a_candidates != b_candidates) & (diff1 > 0)
                mask2 = (a_candidates2 != b_candidates2) & (diff2 > 0)
                
                valid_diffs = np.where(mask1, diff1, np.where(mask2, diff2, np.iinfo(np.int64).max))
                
                min_idx = np.argmin(valid_diffs)
                if valid_diffs[min_idx] < np.iinfo(np.int64).max:
                    b = int(b_candidates[min_idx])
                    if mask1[min_idx]:
                        a = int(a_candidates[min_idx])
                    else:
                        a = int(a_candidates2[min_idx])
                    
                    if a != b:
                        diff = abs(2 * a - 3 * b)
                        if diff > 0:
                            current_score = diff / (2.0 * b)
                            if current_score < best_score:
                                best_score = current_score
                                best_a = a
                                best_b = b
        
        i = next_idx
        
        if time.time() > deadline:
            break
    
    # 4. Local polish: try to improve around the best pair found
    if best_score < float('inf') and time.time() < deadline - 5:
        # Try nearby values
        for da in range(-100, 101):
            for db in range(-100, 101):
                a_new = best_a + da
                b_new = best_b + db
                if a_new <= 0 or b_new <= 0 or a_new > N or b_new > N:
                    continue
                if a_new == b_new:
                    continue
                if sigma[a_new] == sigma[b_new]:
                    diff = abs(2 * a_new - 3 * b_new)
                    if diff > 0:
                        score = diff / (2.0 * b_new)
                        if score < best_score:
                            best_score = score
                            best_a = a_new
                            best_b = b_new
    
    return [int(best_a), int(best_b)]
>>>>>>> REPLACE

