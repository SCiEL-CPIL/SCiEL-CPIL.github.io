import bisect
import time

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    # 1. Generate primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
                
    primes = [i for i, prime in enumerate(is_prime) if prime]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return primes
        
    # dp[i][j] = length of longest valid subsequence ending at primes[i] 
    # with primes[j] as the immediate predecessor.
    # prev[i][j] = index k such that dp[j][k] was the max used to form dp[i][j]
    # If prev[i][j] is -1, it means the subsequence is just [primes[j], primes[i]]
    
    dp = [[0] * n for _ in range(n)]
    prev = [[-1] * n for _ in range(n)]
    
    # suffix_max[j][k] = max(dp[j][k], dp[j][k+1], ..., dp[j][j-1])
    # argmax_suffix[j][k] = index m in [k, j-1] that achieves the max
    suffix_max = [None] * n
    argmax_suffix = [None] * n
    
    best_len = 1
    best_i = 0
    best_j = -1 # dummy
    
    # We iterate i from 1 to n-1
    # For each i, we need dp[j][k] for all j < i.
    # So we can precompute suffix structures for j as we go?
    # No, dp[j][k] is fixed once we've finished all i > j?
    # Wait. dp[j][k] depends on i > j? No.
    # dp[j][k] is the length of the best chain ending at j with predecessor k.
    # This value is determined by chains ending at k.
    # So dp[j][k] is fully determined before we start calculating any dp[i][j] for i > j.
    # Thus, we can precompute suffix_max for row j once all dp[j][k] (k<j) are known.
    # But dp[j][k] is calculated when the outer loop variable is j?
    # No, dp[j][k] is calculated when the outer loop variable is j?
    # Let's check the dependency.
    # dp[i][j] depends on dp[j][k].
    # So to calculate dp[i][j], we need row j of dp to be complete.
    # Row j of dp consists of dp[j][k] for k < j.
    # These values are calculated when the outer loop index is j.
    # So, at the start of outer loop iteration i, all rows 0 to i-1 are complete.
    # So we can precompute suffix_max for row j when we enter the inner loop for j?
    # Or better, precompute it right after finishing the outer loop for j.
    
    # Let's restructure:
    # Outer loop i from 1 to n-1:
    #   For j from 0 to i-1:
    #      We need suffix_max[j].
    #      Since j < i, row j was completed in previous outer loop iterations.
    #      So we can compute suffix_max[j] on the fly the first time we need it for a given j?
    #      Or just precompute all suffix_max[j] for j < i at the start of outer loop i?
    #      That would be O(n^3) if we recompute.
    #      We should compute suffix_max[j] once, after dp[j][*] is fully populated.
    #      dp[j][k] is populated when outer loop is j.
    #      So after finishing outer loop for j, we can compute suffix_max[j].
    
    # Revised Plan:
    # Initialize dp, prev.
    # For i in 1..n-1:
    #    # Before processing i, ensure suffix_max for all j < i is ready.
    #    # Actually, suffix_max[j] only depends on dp[j][0..j-1].
    #    # dp[j][k] is calculated when outer loop is j.
    #    # So after the outer loop iteration for j is done, dp[j][0..j-1] is final.
    #    # So we can compute suffix_max[j] at the END of outer loop iteration j.
    #    # But we need suffix_max[j] for calculations in outer loop i > j.
    #    # So yes, compute it at end of iteration j.
    
    # Wait, dp[j][k] is calculated when outer loop is j?
    # dp[j][k] is an entry where the "end" index is j.
    # So yes, when we process i=j in the outer loop, we calculate dp[j][k] for all k < j.
    # So at the end of iteration i=j, the entire row j of dp (for columns k<j) is ready.
    # So we can compute suffix_max[j] then.
    
    # Let's implement this.
    
    # Pre-fill suffix_max for j=0?
    # j=0 has no k < 0. So suffix_max[0] is empty.
    
    for i in range(1, n):
        # Calculate dp[i][j] for all j < i
        for j in range(0, i):
            gap = primes[i] - primes[j]
            target = primes[j] - gap
            
            # Find smallest k in [0, j-1] such that primes[k] >= target
            k_start = bisect.bisect_left(primes, target, 0, j)
            
            if k_start < j:
                # Valid k exists in [k_start, j-1]
                # Get max from suffix_max[j] starting at k_start
                M = suffix_max[j][k_start]
                k_star = argmax_suffix[j][k_start]
            else:
                # No valid k, base case
                M = 1
                k_star = -1
                
            dp[i][j] = M + 1
            prev[i][j] = k_star
            
            if dp[i][j] > best_len:
                best_len = dp[i][j]
                best_i = i
                best_j = j
                
        # After processing all j for this i, the row i of dp is complete?
        # No, dp[i][j] is calculated for j < i.
        # So row i (as the "end" index) is now complete for columns 0..i-1.
        # So we can compute suffix_max[i] now.
        
        if i > 0:
            # Compute suffix_max[i] and argmax_suffix[i]
            # Size is i (indices 0 to i-1)
            sm = [0] * i
            am = [0] * i
            
            # Start from i-1 down to 0
            sm[i-1] = dp[i][i-1]
            am[i-1] = i-1
            
            for k in range(i-2, -1, -1):
                if dp[i][k] >= sm[k+1]:
                    sm[k] = dp[i][k]
                    am[k] = k
                else:
                    sm[k] = sm[k+1]
                    am[k] = am[k+1]
                    
            suffix_max[i] = sm
            argmax_suffix[i] = am
            
        # Check deadline
        if time.time() > deadline:
            break
            
    if best_len == 1:
        return [primes[0]]
        
    # Reconstruct
    seq = []
    i = best_i
    j = best_j
    while j != -1:
        seq.append(primes[i])
        k = prev[i][j]
        i = j
        j = k
    seq.append(primes[i])
    
    seq.reverse()
    return seq
