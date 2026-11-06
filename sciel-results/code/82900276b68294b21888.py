import time
import numpy as np
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Strategy: 
    # 1. Choose a powerful number H that is highly composite in terms of powerful divisors.
    #    H = 3600 = 2^4 * 3^2 * 5^2 is powerful.
    #    If a = H*x and b = H*y, then a+b = H(x+y) = N.
    #    For a to be powerful, H*x must be powerful.
    #    Since H is powerful, H*x is powerful iff x is powerful.
    #    Proof: Let H = prod p_i^{e_i} with e_i >= 2.
    #           Let x = prod p_i^{f_i} * prod q_j^{g_j} where q_j are primes not in H.
    #           H*x = prod p_i^{e_i+f_i} * prod q_j^{g_j}.
    #           For H*x to be powerful, all exponents must be >= 2.
    #           e_i+f_i >= 2 is always true since e_i>=2, f_i>=0.
    #           g_j >= 2 is required. Thus x must be powerful.
    #    
    # 2. We want to maximize the number of pairs (x,y) of powerful numbers such that x+y = M, 
    #    where M = N/H.
    #    Then N = H*M.
    #    We can choose M to be a number with many representations as a sum of two powerful numbers.
    #    By choosing M small enough, we can use a small FFT or even O(K^2) to find the best M.
    #    But we want N to be large? No, N just needs to be <= 10^18.
    #    The score is the number of pairs.
    #    The number of pairs for N is the same as the number of pairs for M (since the mapping is bijective).
    #    So we just need to find an M that has many pairs of powerful numbers summing to it.
    #    
    # 3. How to find a good M?
    #    We can use the same FFT logic but on a smaller scale, or we can use the fact that 
    #    small M's have fewer powerful numbers, so we can brute force or use a small FFT.
    #    Actually, the density of powerful numbers is higher for small numbers? 
    #    No, the count of powerful numbers up to X is ~ c * X^{1/2}.
    #    The number of pairs for sum M is roughly proportional to the product of densities?
    #    Actually, let's just find the M that maximizes the count of pairs (x,y) with x+y=M, 
    #    where x,y are powerful.
    #    
    # 4. Let's generate powerful numbers up to some limit L.
    #    If we pick L such that 2*L is a reasonable size for FFT, we can find the best M in [1, 2L].
    #    Then N = H * M.
    #    We need N <= 10^18.
    #    If H=3600, M can be up to 10^18 / 3600 ~ 2.7 * 10^14.
    #    But we can't generate powerful numbers up to 2.7 * 10^14.
    #    
    #    Wait, the mapping a=H*x, b=H*y requires x,y to be powerful.
    #    But a,b don't have to be multiples of H! 
    #    The strategy "a=H*x, b=H*y" is a SUFFICIENT condition for finding pairs, 
    #    but it might miss pairs where a is not a multiple of H.
    #    However, if we find a set of pairs all of the form H*x, H*y, that is a valid set of pairs for N.
    #    The score is the number of pairs we CERTIFY.
    #    We don't have to find ALL pairs, just a set of them.
    #    So if we find K pairs of the form (H*x, H*y), the score is at least K.
    #    
    #    So the problem reduces to: Find M and H such that the number of pairs (x,y) of powerful numbers 
    #    with x+y=M is maximized, subject to H*M <= 10^18.
    #    Since H can be any powerful number, we can make H as large as we want (up to 10^18/M), 
    #    but H must be powerful.
    #    Actually, H doesn't affect the count of pairs (x,y).
    #    So we just need to find the M that maximizes the number of pairs (x,y) of powerful numbers 
    #    with x+y=M.
    #    And then we can pick any powerful H such that H*M <= 10^18.
    #    To be safe, we can pick H=1, then N=M.
    #    So this strategy is equivalent to finding the M that maximizes the number of pairs.
    #    Which is exactly what the current program does!
    #    
    #    So why is this different?
    #    The current program generates powerful numbers up to M_limit and does FFT.
    #    The issue is that M_limit is limited by memory/time.
    #    
    #    Is there a way to find a better M without generating all powerful numbers up to M?
    #    Maybe by using a "sparse" convolution or a number-theoretic argument?
    #    
    #    Let's look at the inspirations.
    #    Inspiration 1: "Random Sampling + Verification" -> 5.9333
    #    Inspiration 2: "Number-Theoretic Parameterized Search" -> 5.9333
    #    
    #    Let's try Inspiration 2's idea: 
    #    "Enumerate pairs of powerful numbers (a,b) and count the frequency of a+b."
    #    This is O(K^2).
    #    If K is small, this is fast.
    #    But K grows as sqrt(M).
    #    If we want to find a good N, we need to check many (a,b) pairs.
    #    
    #    What if we restrict a and b to be of the form p^2 or p^3?
    #    No, that's too restrictive.
    #    
    #    Let's try a different angle: 
    #    What if we pick N to be a sum of two powerful numbers in many ways by making N a multiple of a powerful number H, 
    #    and then finding many pairs (x,y) such that x+y = N/H?
    #    This is the same as before.
    #    
    #    Let's look at the "Collision Center" idea.
    #    Maybe we can find N by solving a system of equations?
    #    
    #    Let's try a simpler improvement to the current program:
    #    The current program uses FFT. FFT is fast, but it's an approximation (floating point).
    #    We verify the top candidates.
    #    What if we use a "Two-Pointer" or "Hash Map" approach on a smaller set?
    #    
    #    Let's try to generate powerful numbers up to a smaller limit, say 10^6.
    #    K ~ 1000.
    #    O(K^2) = 10^6 operations. Very fast.
    #    We can count the frequency of each sum.
    #    Find the sum M with the highest frequency.
    #    Then N = M.
    #    But M is small. The score is the number of pairs.
    #    For small M, the number of pairs is small.
    #    
    #    We need to go to larger M to get more pairs.
    #    
    #    Let's try to use the fact that powerful numbers are "dense" in a logarithmic sense?
    #    
    #    Actually, let's look at the best score: 5.9333.
    #    The current program gets 4.0667.
    #    The difference is ~2.
    #    
    #    Let's try to implement the "Number-Theoretic Parameterized Search" from Inspiration 2.
    #    Idea: Instead of finding N by convolution, we can pick a powerful number H, 
    #    and then find many pairs (x,y) such that x+y = M, where M is chosen to maximize the count.
    #    
    #    Wait, I already established that this is equivalent to finding the best M.
    #    
    #    Let's try a different strategy: 
    #    **Hybrid Search**: 
    #    1. Generate powerful numbers up to a moderate limit (e.g., 10^7).
    #    2. Use FFT to find top candidates for N.
    #    3. For each candidate N, verify the count.
    #    4. Additionally, try to "extend" the pairs by scaling?
    #       If (a,b) is a pair for N, is (a*k, b*k) a pair for N*k?
    #       Only if a*k and b*k are powerful.
    #       If k is a square, and a,b are powerful, then a*k and b*k are powerful.
    #       So if we find a pair (a,b) for N, we can generate pairs (a*k^2, b*k^2) for N*k^2.
    #       This doesn't increase the number of pairs for a fixed N, but it allows us to find 
    #       representations for larger N's based on smaller ones.
    #    
    #    But the score is for a SINGLE N.
    #    So we want to find an N with many pairs.
    #    
    #    Let's try to increase the limit M in the current program as much as possible.
    #    The current program uses M=16,000,000.
    #    FFT size is 2^25 = 33,554,432.
    #    This takes ~1-2 seconds.
    #    We can try M=30,000,000.
    #    FFT size 2^26 = 67,108,864.
    #    This might take 4-8 seconds and use ~500MB.
    #    Let's try it.
    
    M = 30_000_000
    
    # Check if we have enough time
    if time.time() > deadline - 20:
        M = 10_000_000

    # 1. Generate powerful numbers up to M
    powerful_set = set()
    max_v = int(M ** (1/3)) + 2
    
    for v in range(1, max_v + 1):
        v_cubed = v ** 3
        if v_cubed > M:
            break
        max_u = isqrt(M // v_cubed)
        for u in range(1, max_u + 1):
            p = u * u * v_cubed
            if p <= M:
                powerful_set.add(p)
            
    powerful_list = sorted(powerful_set)
    K = len(powerful_list)
    
    # 2. Prepare for FFT
    size = 1
    while size < 2 * M + 1:
        size *= 2
        
    # Check time
    if time.time() > deadline - 10:
        # Fallback to smaller M
        M = 10_000_000
        powerful_set = set()
        max_v = int(M ** (1/3)) + 2
        for v in range(1, max_v + 1):
            v_cubed = v ** 3
            if v_cubed > M:
                break
            max_u = isqrt(M // v_cubed)
            for u in range(1, max_u + 1):
                p = u * u * v_cubed
                if p <= M:
                    powerful_set.add(p)
        powerful_list = sorted(powerful_set)
        size = 1
        while size < 2 * M + 1:
            size *= 2

    # Create indicator array
    A = np.zeros(size, dtype=np.float64)
    for p in powerful_list:
        A[p] = 1.0
        
    # 3. FFT Convolution
    F = np.fft.rfft(A)
    F_sq = F * F
    counts_float = np.fft.irfft(F_sq, n=size)
    
    # We only care about N up to 2*M
    counts_int = np.round(counts_float[:2*M + 1]).astype(np.int64)
    
    # 4. Find candidates
    max_val = np.max(counts_int)
    if max_val == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    top_k = 200
    top_indices = np.argpartition(counts_int, -top_k)[-top_k:]
    
    # 5. Verify exact counts for these candidates
    best_N = 0
    best_exact_count = 0
    best_pairs = []
    
    powerful_set_local = powerful_set
    
    for idx in top_indices:
        N = int(idx)
        count = 0
        pairs = []
        for a in powerful_list:
            if a > N // 2:
                break
            b = N - a
            if b in powerful_set_local:
                count += 1
                pairs.append([a, b])
                
        if count > best_exact_count:
            best_exact_count = count
            best_N = N
            best_pairs = pairs
            
    if time.time() > deadline - 5:
        pass

    return {'N': best_N, 'pairs': best_pairs}
