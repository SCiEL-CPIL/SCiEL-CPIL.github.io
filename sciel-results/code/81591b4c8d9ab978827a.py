import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Strategy:
    # 1. Construct N <= 10^18 that maximizes the number of representations as a sum of two squares.
    #    This is a subset of powerful numbers (squares are powerful).
    #    The number of representations of N as a sum of two squares is 4 * prod(e_i + 1)
    #    where e_i are exponents of primes p_i ≡ 1 (mod 4) in the factorization of N,
    #    provided all primes ≡ 3 (mod 4) have even exponents.
    #    We will only use primes ≡ 1 (mod 4) and 2.
    #    Primes ≡ 1 (mod 4): 5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, ...
    #    We want to maximize prod(e_i + 1) subject to sum(e_i * log(p_i)) <= log(10^18).
    #    This is an integer knapsack problem.
    
    # Primes p ≡ 1 (mod 4)
    primes_1_4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193]
    
    # Maximize prod(e_i + 1) subject to sum(e_i * log(p_i)) <= L
    # L = log(10^18)
    L = 18 * np.log(10)
    
    # We can use dynamic programming or greedy search.
    # Since the number of primes is small (~20) and exponents are small (max ~41 for p=5),
    # we can do a bounded search.
    # Actually, e_i is at most L / log(p_i).
    # For p=5, max e = 41 / 1.6 = 25.
    # For p=13, max e = 41 / 2.5 = 16.
    # This is a multi-dimensional knapsack.
    # We can use a recursive search with memoization or just iterate.
    # Let's use a simple heuristic: assign higher exponents to smaller primes.
    # Or use DP on the log-value? No, log is continuous.
    # Let's use a recursive branch and bound.
    
    best_score = 0
    best_exponents = {}
    
    def search(idx, current_log, current_score, exponents):
        nonlocal best_score, best_exponents
        if time.time() > deadline - 5:
            return
        
        if idx == len(primes_1_4):
            if current_score > best_score:
                best_score = current_score
                best_exponents = exponents.copy()
            return
        
        p = primes_1_4[idx]
        log_p = np.log(p)
        max_e = int((L - current_log) // log_p)
        
        # Try all possible exponents from max_e down to 0
        # Heuristic: higher exponents for smaller primes are usually better
        # But we need to check all to be sure, or use a bound.
        # Bound: even if we use all remaining log for this prime, what's the max score?
        # This is complex. Let's just iterate, but prune if current_score * (max_e+1) < best_score?
        # No, because future primes might contribute.
        # Given the small number of primes, we can just iterate.
        
        for e in range(max_e, -1, -1):
            new_log = current_log + e * log_p
            new_score = current_score * (e + 1)
            # Pruning: if new_score is already less than best_score, and we can't improve it?
            # Actually, score is multiplicative.
            # If new_score <= best_score, it might still be worth it if we can add more factors?
            # No, the score is the product. If the current product is already lower than the best,
            # and we are multiplying by (e+1) which is >= 1, it will only stay lower or get lower?
            # Wait, if e=0, score stays same. If e>0, score increases.
            # So if new_score < best_score, it might still be possible to beat it if we add more primes?
            # No, the score is the total product.
            # If current_score * (e+1) < best_score, it doesn't mean we can't beat it later,
            # because later primes multiply the score.
            # So we can't prune based on current_score alone.
            # However, the number of primes is small.
            
            exponents[idx] = e
            search(idx + 1, new_log, new_score, exponents)
            
    # Initialize
    exponents = [0] * len(primes_1_4)
    search(0, 0.0, 1, exponents)
    
    # Construct N
    N = 1
    for i, p in enumerate(primes_1_4):
        if exponents[i] > 0:
            N *= p ** exponents[i]
            
    if N > 10**18:
        # Fallback to a smaller N
        # Just pick a known good N, e.g., 5^10
        N = 5**10
        
    # Now, find all pairs (a, b) such that a + b = N, a <= b, and a, b are powerful.
    # We know that squares are powerful.
    # So we can find all pairs (k^2, l^2) such that k^2 + l^2 = N.
    # This is equivalent to finding integer solutions to k^2 + l^2 = N.
    
    pairs = []
    # k^2 <= N/2 => k <= isqrt(N/2)
    max_k = isqrt(N // 2)
    
    for k in range(1, max_k + 1):
        k_sq = k * k
        l_sq = N - k_sq
        l = isqrt(l_sq)
        if l * l == l_sq:
            # Check if l >= k
            if l >= k:
                pairs.append([k_sq, l_sq])
                
    # Are there any other powerful pairs?
    # Powerful numbers are of the form prod p_i^{e_i} with e_i >= 2.
    # Squares are a subset.
    # We can try to find more pairs by checking powerful numbers up to N/2.
    # But generating all powerful numbers up to N/2 is expensive if N is large.
    # However, N is at most 10^18.
    # The number of powerful numbers up to X is ~ c * sqrt(X).
    # For X = 10^18, sqrt(X) = 10^9. Too many.
    # So we can only rely on the square pairs.
    # But wait, the current program finds ~6 pairs.
    # How many square pairs does our N have?
    # The number of representations as a sum of two squares is 4 * prod(e_i + 1).
    # This counts ordered pairs (k, l) with signs.
    # The number of unordered positive pairs (k, l) with k <= l is roughly (4 * prod(e_i + 1)) / 8 = prod(e_i + 1) / 2.
    # Let's estimate prod(e_i + 1).
    # If we use 5^10, 13^5, 17^3, 29^2, 37^1, ...
    # Score = 11 * 6 * 4 * 3 * 2 ...
    # 11 * 6 = 66
    # 66 * 4 = 264
    # 264 * 3 = 792
    # 792 * 2 = 1584
    # So we might have ~792 pairs!
    # This is much better than 6.
    
    # Let's verify the count.
    # The formula for the number of representations of N as a sum of two squares is r_2(N) = 4 * (d_1(N) - d_3(N)),
    # where d_1(N) is the number of divisors ≡ 1 (mod 4) and d_3(N) is the number of divisors ≡ 3 (mod 4).
    # For N = prod p_i^{e_i} with p_i ≡ 1 (mod 4), all divisors are ≡ 1 (mod 4) or ≡ 3 (mod 4)?
    # No, divisors of N are of the form prod p_i^{f_i}.
    # Since p_i ≡ 1 (mod 4), p_i^{f_i} ≡ 1 (mod 4).
    # So all divisors are ≡ 1 (mod 4).
    # Thus d_1(N) = prod(e_i + 1) and d_3(N) = 0.
    # So r_2(N) = 4 * prod(e_i + 1).
    # This r_2(N) counts all integer solutions (k, l) including negative and zero.
    # We want positive k, l with k <= l.
    # The solutions come in groups of 8: (±k, ±l), (±l, ±k).
    # If k != l, there are 8 solutions.
    # If k = l, there are 4 solutions: (k, k), (k, -k), (-k, k), (-k, -k). But k^2 + k^2 = N => 2k^2 = N.
    # So if N is not twice a square, k != l for all solutions.
    # Then the number of positive unordered pairs is r_2(N) / 8.
    # Let's check: r_2(N) = 4 * prod(e_i + 1).
    # So number of pairs = 4 * prod(e_i + 1) / 8 = prod(e_i + 1) / 2.
    # This must be an integer.
    # prod(e_i + 1) is even?
    # If any e_i is odd, e_i + 1 is even.
    # If all e_i are even, then N is a square.
    # If N is a square, say N = M^2, then k^2 + l^2 = M^2.
    # This is a Pythagorean triple.
    # In any case, the number of pairs should be prod(e_i + 1) / 2.
    
    # Let's just use the pairs we found by iterating k.
    # The iteration is O(sqrt(N)), which is 10^9 for N=10^18. Too slow.
    # We need to generate the pairs directly from the factorization.
    
    # How to generate the pairs?
    # We can use the formula for the number of representations, but we need the actual pairs.
    # We can use the fact that if N = prod p_i^{e_i}, then the representations can be generated
    # by multiplying Gaussian integers.
    # This is complex.
    # Alternative: since the number of pairs is ~ prod(e_i + 1) / 2, and this is at most ~1000,
    # we can just iterate k up to sqrt(N) if N is small enough.
    # But N can be 10^18.
    # However, the number of pairs is small.
    # Can we find the pairs without iterating up to sqrt(N)?
    # We can use the fact that the pairs are related to the divisors.
    # Actually, we can just use the FFT method on a smaller scale? No.
    # Let's check the time for iterating k.
    # If N = 10^18, sqrt(N) = 10^9. Loop takes 10 seconds in Python.
    # This is acceptable if we do it once.
    # But we have a 110s deadline.
    # Let's try it.
    
    # Optimization: we can break early if we find enough pairs?
    # No, we want all pairs to maximize the score.
    
    # Let's refine the search for exponents to ensure N is not too large,
    # or just accept the time cost.
    
    # To be safe, let's limit the search for exponents to keep N <= 10^17,
    # so sqrt(N) <= 3*10^8, which is faster.
    
    # Let's re-run the search with a tighter bound.
    L_tight = 17 * np.log(10)
    best_score_tight = 0
    best_exponents_tight = [0] * len(primes_1_4)
    
    def search_tight(idx, current_log, current_score, exponents):
        nonlocal best_score_tight, best_exponents_tight
        if time.time() > deadline - 10:
            return
        
        if idx == len(primes_1_4):
            if current_score > best_score_tight:
                best_score_tight = current_score
                best_exponents_tight = exponents.copy()
            return
        
        p = primes_1_4[idx]
        log_p = np.log(p)
        max_e = int((L_tight - current_log) // log_p)
        
        for e in range(max_e, -1, -1):
            new_log = current_log + e * log_p
            new_score = current_score * (e + 1)
            exponents[idx] = e
            search_tight(idx + 1, new_log, new_score, exponents)
            
    search_tight(0, 0.0, 1, [0] * len(primes_1_4))
    
    N = 1
    for i, p in enumerate(primes_1_4):
        if best_exponents_tight[i] > 0:
            N *= p ** best_exponents_tight[i]
            
    # Generate pairs
    pairs = []
    max_k = isqrt(N // 2)
    
    # If max_k is too large, we might time out.
    # But let's try.
    for k in range(1, max_k + 1):
        k_sq = k * k
        l_sq = N - k_sq
        l = isqrt(l_sq)
        if l * l == l_sq and l >= k:
            pairs.append([k_sq, l_sq])
            
    return {'N': N, 'pairs': pairs}
