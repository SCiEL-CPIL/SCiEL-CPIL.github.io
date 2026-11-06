"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict
import numpy as np

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using a^2 * b^3 form."""
    if limit < 1:
        return [1]
    powerful = set()
    powerful.add(1)
    
    # b^3 * a^2 <= limit
    # b up to limit^(1/3)
    max_b = int(limit ** (1/3)) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        # a^2 <= limit / b3
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    
    return sorted(powerful)

def _is_powerful(n):
    """Check if n is powerful by exact factorization."""
    if n <= 1:
        return True
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                temp //= d
                count += 1
            if count == 1:
                return False
        d += 1 if d == 2 else 2  # skip even numbers after 2
    if temp > 1:
        # temp is a prime factor appearing once
        return False
    return True

def _count_representations(N, powerful_list, powerful_set):
    """Count number of pairs (a,b) with a<=b, a+b=N, both powerful."""
    pairs = []
    # Iterate through powerful numbers a <= N//2
    # Since powerful_list is sorted, we can stop when a > N//2
    for a in powerful_list:
        if a > N // 2:
            break
        b = N - a
        if b in powerful_set:
            pairs.append([a, b])
    return pairs

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Generate powerful numbers up to a limit that allows efficient vectorized processing
    # We need M such that M^2 is manageable with numpy bincount.
    # M=10000 -> 50M sums. ~0.5-1s with numpy.
    # M=20000 -> 200M sums. ~2-4s with numpy.
    # Let's aim for M ~ 15000-20000.
    # Count of powerful numbers ~ sqrt(X). So X ~ M^2.
    # M=20000 -> X ~ 4*10^8.
    # Let's generate up to 10^9 to be safe, but limit the list size if it gets too big?
    # No, just generate up to a limit that gives ~20k numbers.
    # sqrt(4*10^8) = 20000.
    limit_gen = 4 * 10**8
    powerful_list = _generate_powerful(limit_gen)
    powerful_set = set(powerful_list)
    M = len(powerful_list)
    
    best_N = 2
    best_pairs = [[1, 1]]
    best_score = 1
    
    if M < 100:
        # Fallback if generation failed or limit too small
        return {'N': best_N, 'pairs': best_pairs}
    
    # Strategy: Use numpy to compute sums of pairs from the first K powerful numbers.
    # We will try a few values of K.
    # K=5000, 10000, 15000, 20000 (limited by M)
    
    subset_sizes = [5000, 10000, 15000, 20000]
    
    for K in subset_sizes:
        if time.time() > deadline - 20:
            break
            
        if K > M:
            K = M
            
        if K < 100:
            continue
        
        P_sub = np.array(powerful_list[:K], dtype=np.int64)
        
        # Compute all pair sums a[i] + a[j] for i <= j
        # Vectorized approach:
        # Create a matrix of sums? No, memory is K^2.
        # K=20000 -> 400M ints -> 3.2GB. Too much.
        # We need a chunked approach or a smarter way.
        
        # Alternative: Use np.bincount on flattened triangular sums?
        # Still need to generate the sums.
        
        # Let's try a smaller K first to see timing.
        # K=5000 -> 25M sums. 200MB. OK.
        # K=10000 -> 100M sums. 800MB. Risky.
        # K=15000 -> 225M sums. 1.8GB. Too much.
        
        # So we are limited to K ~ 10000 by memory if we do full matrix.
        # But we can do it in chunks to save memory?
        # Or just stick to K=10000 and maybe a second pass with random sampling for larger K?
        
        if K > 10000:
             # If K is large, we might run out of memory with full outer sum.
             # Let's cap K for the vectorized full sum method.
             K = 10000
             P_sub = np.array(powerful_list[:K], dtype=np.int64)
        
        # Compute sums
        # Use broadcasting: P_sub[:, None] + P_sub[None, :]
        # This creates a KxK matrix.
        # We only need upper triangle (i <= j).
        # But bincount counts all. We can just count all and divide? No, (a,b) and (b,a) are distinct in matrix but same pair if a!=b.
        # The task asks for distinct pairs a<=b.
        # If we count all sums, sum(a,b) and sum(b,a) are the same N.
        # So count in full matrix = 2 * count(a<b) + count(a==b).
        # We want count(a<=b).
        # So we can just compute the full matrix sums, bincount them, and then...
        # Actually, it's easier to just generate the upper triangle sums.
        
        # Efficient way to get upper triangle sums without full matrix memory?
        # We can iterate over rows and bincount each row's sums?
        # That would be O(K^2) operations but O(K) memory per step.
        
        sum_counts = defaultdict(int)
        
        # Chunked processing to avoid huge temporary arrays if needed, 
        # but for K=10000, a 10000x10000 int64 array is 800MB. 
        # Python/Numpy might handle it, but it's close to limit.
        # Let's try to do it in chunks of 1000 rows.
        
        chunk_size = 1000
        counts_arr = np.zeros(2 * P_sub[-1] + 1, dtype=np.int64) # Max sum is 2*max(P_sub)
        
        for start in range(0, K, chunk_size):
            end = min(start + chunk_size, K)
            # Take rows start:end
            # We want sums with columns >= start (to avoid double counting and ensure i<=j in global index? No, that's complex)
            # Simpler: Just compute all sums for rows start:end against ALL columns?
            # Then we count each pair twice (once as (i,j) and once as (j,i)) except when i==j.
            # If we sum over all chunks, we get total count for each N.
            # Let C[N] be the count from full matrix.
            # C[N] = 2 * (number of pairs a<b with a+b=N) + (number of pairs a==b with a+b=N)
            # We want P[N] = (number of pairs a<b) + (number of pairs a==b)
            # P[N] = (C[N] + D[N]) / 2 where D[N] is 1 if N is sum of two equal powerfuls, else 0.
            
            # So we can just compute full matrix sums in chunks.
            
            P_chunk = P_sub[start:end]
            # Sums: P_chunk[:, None] + P_sub[None, :]
            # This is (end-start) x K
            sums = P_chunk[:, None] + P_sub[None, :]
            # Flatten and bincount
            flat_sums = sums.ravel()
            # bincount
            chunk_counts = np.bincount(flat_sums, minlength=len(counts_arr))
            counts_arr += chunk_counts
            
        # Now counts_arr[N] is the count in the full KxK matrix.
        # We need to convert to a<=b count.
        # C[N] = 2 * pairs_ab + pairs_aa
        # pairs_ab = (C[N] - pairs_aa) / 2
        # P[N] = pairs_ab + pairs_aa = (C[N] + pairs_aa) / 2
        
        # How to get pairs_aa?
        # pairs_aa is 1 if N/2 is in P_sub and N is even, else 0.
        # We can check this for the top candidates.
        
        # Find top candidates in counts_arr
        # We want top counts.
        # Sort indices by count?
        # counts_arr is large (up to 2*10^9). Sorting is too slow.
        # Use argpartition or just find max?
        # We want top 20.
        
        # Get indices of non-zero counts
        non_zero_idx = np.nonzero(counts_arr)[0]
        if len(non_zero_idx) == 0:
            continue
            
        # Get counts for non-zero
        non_zero_counts = counts_arr[non_zero_idx]
        
        # Find top 20 indices in non_zero_counts
        # Use argpartition for top K
        k_top = min(20, len(non_zero_counts))
        top_indices_in_nonzero = np.argpartition(non_zero_counts, -k_top)[-k_top:]
        
        # Sort these top indices by count descending
        top_indices_sorted = top_indices_in_nonzero[np.argsort(non_zero_counts[top_indices_in_nonzero])[::-1]]
        
        # Get actual N values
        top_Ns = non_zero_idx[top_indices_sorted]
        
        for N_cand in top_Ns:
            if time.time() > deadline - 10:
                break
            
            N_cand = int(N_cand)
            
            # Calculate expected count for a<=b
            # C = counts_arr[N_cand]
            C = int(counts_arr[N_cand])
            
            # Check if N_cand/2 is powerful and in P_sub
            # If N_cand is even, check if N_cand/2 is in powerful_set and <= P_sub[-1]
            D = 0
            if N_cand % 2 == 0:
                half = N_cand // 2
                if half in powerful_set and half <= P_sub[-1]:
                    D = 1
            
            # P[N] = (C + D) / 2
            # Note: C is the count in the full KxK matrix of P_sub.
            # This count includes pairs where a or b might be > P_sub[-1]? No, P_sub is the set.
            # So C counts pairs (a,b) where a,b in P_sub.
            # The actual number of representations for N using ALL powerful numbers might be higher.
            # We must verify with the full powerful_list.
            
            # But first, let's see if this candidate is promising.
            # If (C+D)/2 is high, it's a good candidate.
            
            # Verify with full set
            pairs = _count_representations(N_cand, powerful_list, powerful_set)
            
            if len(pairs) > best_score:
                best_score = len(pairs)
                best_N = N_cand
                best_pairs = pairs

    return {'N': best_N, 'pairs': best_pairs}

# EVOLVE-BLOCK-END
