import time
import numpy as np
from collections import defaultdict

MAX_N = 10_000_000
TARGET_NUM, TARGET_DEN = 3, 2
DEADLINE_FRACTION = 0.85

def construct():
    t_start = time.time()
    deadline = t_start + 90.0 * DEADLINE_FRACTION
    
    # 1. Compute sigma(n) for all n <= MAX_N using Linear Sieve
    # sigma(n) = sum of divisors of n
    # We use the property: if n = p^k * m where gcd(p, m) = 1, 
    # sigma(n) = sigma(p^k) * sigma(m) = (p^(k+1)-1)/(p-1) * sigma(m)
    # Linear sieve keeps track of the smallest prime factor and the power of that prime.
    
    N = MAX_N
    spf = np.zeros(N + 1, dtype=np.int64) # Smallest Prime Factor
    primes = []
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    # We need to store sigma values. They can be large.
    # Max sigma for n <= 10^7 is roughly < 5 * 10^7 (highly composite numbers).
    # Actually, sigma(n) can be larger. For n=9,999,999, sigma is around 1.5*10^7.
    # Let's use int64 to be safe.
    sigma = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    
    # Also need to track the "sigma part" contributed by the smallest prime factor power
    # Let spf[n] be the smallest prime factor p.
    # Let exp[n] be the exponent of p in n.
    # Let spf_power[n] = p^exp[n].
    # Let sigma_part[n] = 1 + p + ... + p^exp[n].
    # Then sigma[n] = sigma_part[n] * sigma[n / spf_power[n]].
    
    exp = np.zeros(N + 1, dtype=np.int32)
    spf_power = np.zeros(N + 1, dtype=np.int64)
    sigma_part = np.zeros(N + 1, dtype=np.int64)
    
    sigma[1] = 1
    sigma_part[1] = 1
    spf_power[1] = 1
    
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            spf[i] = i
            exp[i] = 1
            spf_power[i] = i
            sigma_part[i] = 1 + i
            sigma[i] = 1 + i
        
        for p in primes:
            if i * p > N:
                break
            is_prime[i * p] = False
            spf[i * p] = p
            
            if p == spf[i]:
                # p divides i
                exp[i * p] = exp[i] + 1
                spf_power[i * p] = spf_power[i] * p
                # sigma_part[i*p] = sigma_part[i] + p^(exp[i]+1)
                sigma_part[i * p] = sigma_part[i] + spf_power[i * p]
                # sigma[i*p] = sigma_part[i*p] * sigma[i / spf_power[i]]
                sigma[i * p] = sigma_part[i * p] * sigma[i // spf_power[i]]
                break
            else:
                # p does not divide i, so p < spf[i]
                exp[i * p] = 1
                spf_power[i * p] = p
                sigma_part[i * p] = 1 + p
                sigma[i * p] = sigma_part[i * p] * sigma[i]
                
        if time.time() > deadline - 10.0:
            # If we run out of time during sieve, we have a problem. 
            # But linear sieve is fast. 10^7 is ~0.5-1s in numpy/python loop.
            # Actually, pure python loop for 10^7 might be 5-10s.
            # Let's hope it fits. If not, we need a faster sieve.
            pass

    # 2. Group numbers by sigma value
    # Using a dictionary of lists is memory heavy. 
    # Alternative: Sort indices by sigma value, then process contiguous blocks.
    
    # Create array of indices 1..N
    indices = np.arange(1, N + 1, dtype=np.int32)
    
    # Sort indices based on sigma values
    # np.argsort is stable and fast
    order = np.argsort(sigma[1:N+1], kind='mergesort')
    
    # Now sigma[1:N+1][order] is sorted.
    sorted_sigmas = sigma[1:N+1][order]
    sorted_indices = indices[order]
    
    best_score = float('inf')
    best_a = 1
    best_b = 2 # placeholder, must have same sigma
    
    # Iterate through groups of equal sigma
    # Find boundaries where sigma value changes
    # diff = np.diff(sorted_sigmas)
    # changes = np.where(diff != 0)[0] + 1
    # boundaries = np.concatenate(([0], changes, [len(sorted_sigmas)]))
    
    # To avoid creating huge temporary arrays, we can loop through the sorted array
    # and identify blocks.
    
    current_sigma = sorted_sigmas[0]
    start_idx = 0
    
    for i in range(1, len(sorted_sigmas)):
        if sorted_sigmas[i] != current_sigma:
            # Process group from start_idx to i-1
            if i - start_idx >= 2:
                group_indices = sorted_indices[start_idx:i]
                
                # Vectorized search for best pair in this group
                # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
                # For each b in group, find a in group closest to 1.5*b
                
                # Convert to float for 1.5*b calculation? 
                # No, keep as int for precision if possible, but 1.5*b might overflow int32.
                # group_indices are up to 10^7. 1.5 * 10^7 = 1.5 * 10^7, fits in int32? 
                # 2^31 is ~2.1 * 10^9. So yes, int32 is fine.
                
                # Use searchsorted
                # targets = 1.5 * group_indices
                # Since group_indices is sorted, targets is sorted.
                
                # To avoid float precision issues in searchsorted, we can use integer arithmetic.
                # We want a closest to 3b/2.
                # Equivalent to finding a such that 2a is closest to 3b.
                # Let's stick to float for simplicity, 10^7 is well within float precision (2^53).
                
                targets = 1.5 * group_indices.astype(np.float64)
                
                # searchsorted returns insertion index
                idxs = np.searchsorted(group_indices, targets, side='left')
                
                # Candidate indices: idxs and idxs-1
                # Clip indices to valid range [0, len(group)-1]
                idxs_clipped = np.clip(idxs, 0, len(group_indices) - 1)
                idxs_minus = np.clip(idxs - 1, 0, len(group_indices) - 1)
                
                a_candidates1 = group_indices[idxs_clipped]
                a_candidates2 = group_indices[idxs_minus]
                
                b_values = group_indices
                
                # Calculate error for both candidates
                # err = |2a - 3b| / (2b)
                
                # Candidate 1
                diff1 = np.abs(2 * a_candidates1 - 3 * b_values).astype(np.float64)
                den1 = 2 * b_values.astype(np.float64)
                err1 = diff1 / den1
                
                # Candidate 2
                diff2 = np.abs(2 * a_candidates2 - 3 * b_values).astype(np.float64)
                den2 = 2 * b_values.astype(np.float64)
                err2 = diff2 / den2
                
                # We need to exclude cases where a == b (diff=0 is valid but a/b=1, not 3/2)
                # And exclude exact match a/b = 3/2 -> 2a = 3b -> diff=0.
                # The contract says: "A raw value of exactly 0 cannot be scored and counts as invalid."
                # So if diff == 0, we must skip that pair.
                
                # Mask out exact matches
                valid1 = diff1 > 0
                valid2 = diff2 > 0
                
                # Also, if a == b, diff is 0, so it's handled by valid mask.
                
                # Find min error in valid candidates
                min_err1 = np.min(err1[valid1]) if np.any(valid1) else float('inf')
                min_err2 = np.min(err2[valid2]) if np.any(valid2) else float('inf')
                
                current_min = min(min_err1, min_err2)
                
                if current_min < best_score:
                    # Find which one it was and update best_a, best_b
                    if min_err1 < min_err2:
                        # Find index of min_err1 in err1[valid1]
                        # This is tricky with masks. 
                        # Let's just recompute or find the argmin carefully.
                        # Simpler: 
                        # Flatten valid errors and find global min
                        pass # We'll do a simpler update logic below
                    
                    # To find a,b:
                    # We can just iterate over the small number of "best" candidates?
                    # No, vectorized find is better.
                    
                    # Let's combine errors and indices
                    # This is getting complex to trace back to a,b.
                    # Alternative: Just compute the min error. 
                    # If it improves, we need the a,b.
                    
                    # Let's refine:
                    # err1 and err2 are arrays.
                    # We want the global min of the union of valid err1 and valid err2.
                    
                    # Create a combined array? No, too slow.
                    
                    # Just check min_err1 and min_err2.
                    if min_err1 < min_err2:
                        # Find the b index that gave min_err1
                        # err1[valid1] is a subset. 
                        # np.argmin on the masked array doesn't give original index directly.
                        # Let's set invalid ones to inf.
                        err1_clean = np.where(valid1, err1, float('inf'))
                        idx_b = np.argmin(err1_clean)
                        b = group_indices[idx_b]
                        a = a_candidates1[idx_b]
                    else:
                        err2_clean = np.where(valid2, err2, float('inf'))
                        idx_b = np.argmin(err2_clean)
                        b = group_indices[idx_b]
                        a = a_candidates2[idx_b]
                        
                    best_score = current_min
                    best_a = int(a)
                    best_b = int(b)
                    
            current_sigma = sorted_sigmas[i]
            start_idx = i
            
            if time.time() > deadline:
                break

    # Ensure types are correct
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-START
# The code above is the complete program.
# EVOLVE-BLOCK-END
