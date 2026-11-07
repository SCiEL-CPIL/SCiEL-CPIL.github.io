"""Improved program: vectorized sigma sieve, find M with many k where k*sigma(k)=M, k<=sqrt(M)."""
import time
import math
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def _sigma_sieve_numpy(n):
    """
    Compute sigma(k) for all k in 1..n using vectorized NumPy operations.
    Returns a NumPy array sigma of shape (n+1,) where sigma[0]=0.
    """
    sigma = np.zeros(n + 1, dtype=np.int64)
    # Iterate over divisors i
    # For large n, the number of iterations is n, but each step is a slice operation.
    # To avoid memory blowup or excessive Python loop overhead for very large n,
    # we might need to be careful, but for n=10^8, the loop runs 10^8 times in Python?
    # No, the loop `for i in range(1, n+1)` is still Python-level.
    # Wait, the standard sieve is:
    # for i in 1..n:
    #   for j in i, 2i, ... n:
    #     sigma[j] += i
    # 
    # Vectorized version:
    # for i in 1..n:
    #   sigma[i::i] += i
    # 
    # This loop still runs n times in Python. For n=10^8, this is too slow in Python.
    # We need a faster way or a smaller n.
    # 
    # Alternative: Use a more efficient factorization-based sigma calculation?
    # Or limit n to a size where the Python loop is acceptable?
    # 10^7 iterations in Python takes ~1-2 seconds if the body is fast.
    # sigma[i::i] += i is a NumPy operation.
    # For i=1, it touches 10^8 elements.
    # For i=2, it touches 5*10^7 elements.
    # Total operations sum(n/i) = n log n.
    # For n=10^7, sum is ~10^7 * 16 = 1.6*10^8 operations.
    # NumPy can do ~10^8-10^9 simple ops per second.
    # The Python loop overhead for 10^7 iterations is the killer.
    # 
    # Let's try to optimize the loop or use a different strategy.
    # Strategy: Only compute for k up to a limit where k*sigma(k) <= 10^18.
    # Max k is roughly 10^9.
    # If we can't sieve 10^9, we can't find all solutions.
    # 
    # However, the "cell findings" said: "Searching up to k=10^7 allows finding M with more than 3 preimages... Measured: 1 -> 1.667."
    # This implies that solutions with many preimages might be found in smaller ranges or that the previous attempt at 10^7 was the bottleneck.
    # 
    # Let's try to make the sieve faster by reducing the number of Python loop iterations.
    # We can't easily vectorize the outer loop.
    # 
    # Maybe we can use a different algorithm for sigma?
    # sigma(n) = product (p^(e+1)-1)/(p-1).
    # We can sieve primes first, then compute sigma for each number using its prime factorization.
    # This is O(n log log n) for prime sieve + O(n * number of prime factors) for sigma.
    # 
    # Let's implement a linear sieve for smallest prime factor (SPF) and then compute sigma.
    # This is much faster than the divisor summation method for large n in Python if implemented carefully, 
    # but actually, the divisor summation `sigma[i::i] += i` is very fast per iteration.
    # The problem is the 10^7 Python loop iterations.
    # 
    # Let's test with a smaller N first, say 2*10^6, to ensure it runs fast, then scale up if time permits.
    # Or, use a hybrid: Sieve up to N.
    
    # Actually, there is a trick. We can group the updates? No, they are independent.
    # 
    # Let's stick to the NumPy sieve but keep N reasonable, e.g., 5*10^6 or 10^7.
    # 10^7 iterations in Python is ~1s-2s overhead + computation.
    # Computation: sum of n/i for i=1..n is n log n.
    # For n=10^7, n log n ~ 1.6 * 10^8 additions.
    # NumPy addition is ~10^8/sec. So ~1.6s.
    # Total ~3-4s. This is safe.
    
    # Let's try N = 10^7.
    
    # Optimization: Use int64.
    # The loop:
    for i in range(1, n + 1):
        sigma[i::i] += i
        
    return sigma

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_M = 0
    best_k_list = []
    
    # We will try to sieve up to N.
    # Let's start with N = 10^7. If it's fast, we can try larger.
    # But we must finish in time.
    
    N = 10_000_000
    
    t0 = time.time()
    # Check if we have time to do the sieve
    # Sieve takes roughly 2-4 seconds for 10^7.
    # If we are already close to deadline, skip? No, we just started.
    
    try:
        sigma = _sigma_sieve_numpy(N)
    except MemoryError:
        # Fallback to smaller N
        N = 5_000_000
        sigma = _sigma_sieve_numpy(N)
    
    t1 = time.time()
    
    # Create array of k values: 1 to N
    k_vals = np.arange(1, N + 1, dtype=np.int64)
    
    # Compute M = k * sigma(k)
    # Note: sigma[0] is 0, but k starts at 1.
    # sigma array is 0-indexed, so sigma[k] is correct.
    M_vals = k_vals * sigma[1:N+1]
    
    # Filter M <= 10^18
    mask = M_vals <= 10**18
    valid_Ms = M_vals[mask]
    valid_ks = k_vals[mask]
    
    # Find the M with the most occurrences
    # np.unique returns unique values and their counts
    unique_Ms, counts, inverse_indices = np.unique(valid_Ms, return_counts=True, return_inverse=True)
    
    if len(unique_Ms) == 0:
        return {'M': 1, 'k': [1]}
        
    max_count_idx = np.argmax(counts)
    best_M_candidate = unique_Ms[max_count_idx]
    max_count = counts[max_count_idx]
    
    # Get the k values corresponding to this M
    # inverse_indices tells us which unique value each valid_M belongs to.
    # We need valid_ks where inverse_indices == max_count_idx
    best_k_candidates = valid_ks[inverse_indices == max_count_idx]
    
    # Update best
    if max_count > len(best_k_list):
        best_M = int(best_M_candidate)
        best_k_list = [int(k) for k in best_k_candidates]
    
    # If we have time, try a larger N?
    # The previous attempt failed at 10^7? No, it failed due to timeout, likely because 10^7 was too slow in pure Python, 
    # but NumPy should handle it. 
    # However, if 10^7 only finds 3-4 solutions, maybe a larger range is needed?
    # The "cell finding" says 10^7 found 1.667 score. 
    # Let's try to extend to 2*10^7 if time permits.
    
    if time.time() < deadline - 20:
        N2 = 20_000_000
        try:
            sigma2 = _sigma_sieve_numpy(N2)
            k_vals2 = np.arange(1, N2 + 1, dtype=np.int64)
            M_vals2 = k_vals2 * sigma2[1:N2+1]
            mask2 = M_vals2 <= 10**18
            valid_Ms2 = M_vals2[mask2]
            valid_ks2 = k_vals2[mask2]
            
            unique_Ms2, counts2, inverse_indices2 = np.unique(valid_Ms2, return_counts=True, return_inverse=True)
            
            if len(unique_Ms2) > 0:
                max_count_idx2 = np.argmax(counts2)
                best_M_candidate2 = unique_Ms2[max_count_idx2]
                max_count2 = counts2[max_count_idx2]
                
                if max_count2 > len(best_k_list):
                    best_M = int(best_M_candidate2)
                    best_k_candidates2 = valid_ks2[inverse_indices2 == max_count_idx2]
                    best_k_list = [int(k) for k in best_k_candidates2]
        except Exception:
            pass # Keep previous best

    # Final verification step is implicit in the construction, but let's ensure types.
    # The problem states k_i must be <= sqrt(M).
    # M = k * sigma(k). Since sigma(k) >= k+1 for k>1, M > k^2, so k < sqrt(M).
    # For k=1, M=2, sqrt(2)=1.41, k=1 <= 1.41.
    # So the condition is naturally satisfied for all k where M=k*sigma(k).
    
    # Ensure distinctness (np.unique on M doesn't guarantee distinct k in the list? 
    # No, k_vals are distinct indices. If two different k have same M, they are both in the list.
    # The task requires distinct k_i. They are distinct by construction from arange.)
    
    # Check if best_k_list is empty (shouldn't happen)
    if not best_k_list:
        best_M = 2
        best_k_list = [1]
        
    return {'M': int(best_M), 'k': best_k_list}

# EVOLVE-BLOCK-END
