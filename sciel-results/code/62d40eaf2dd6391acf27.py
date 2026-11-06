import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    def is_powerful(n):
        if n <= 1:
            return True
        # Check if all prime factors have exponent >= 2
        # Trial division up to cube root, then handle remainder
        # For speed, we can use a precomputed set for small numbers, 
        # but for large N-a, we need a fast check.
        # A number is powerful iff n = x^2 * y^3.
        # Alternatively, check prime factors.
        # Since we are checking N-a where N is fixed and a is small powerful,
        # N-a is large. 
        # Efficient check: 
        # 1. If n is not divisible by p^2 for any p, return False.
        # This is hard to do without factorization.
        # However, we can use the property: n is powerful iff 
        # rad(n) (product of distinct primes) divides n.
        # Or simply: for each prime p, if p|n then p^2|n.
        
        # Let's use a simpler heuristic first: 
        # Precompute powerful numbers up to a limit L.
        # If N-a <= L, check set.
        # If N-a > L, we need a check.
        
        # Actually, the most robust way in Python without sympy is:
        # Trial division by primes up to n^(1/3).
        # If remaining n' > 1:
        #   if n' is prime -> not powerful (exp 1) unless n'=1
        #   if n' is square of prime -> powerful
        #   if n' is cube of prime -> powerful
        #   if n' is square of square? i.e. p^4? -> powerful
        #   if n' = p^k with k>=2 -> powerful
        #   if n' = p*q ... -> not powerful
        
        # This is slow for many calls. 
        # Alternative: Generate powerful numbers up to max_a + something? No.
        
        # Let's stick to the "N is multiple of H" idea.
        # If we pick N = H * K, and H is powerful, and we ensure K is powerful?
        # No, a+b=N.
        
        # Let's optimize the check function using a cache and a fast path.
        # For this implementation, I will use a precomputed set of powerful numbers 
        # up to a reasonably large limit, and for larger numbers, I will use 
        # a probabilistic or approximate check? No, must be exact.
        
        # Exact check for n:
        # Factorize n.
        # This is the bottleneck.
        
        # Strategy adjustment: 
        # Instead of checking N-a for arbitrary a, we can choose N such that 
        # N-a is likely to be powerful? 
        # Or, we can just return the pairs found by the convolution method.
        # The convolution method finds pairs (a,b) where a,b are in the precomputed list.
        # If we make the precomputed list large enough, we find many pairs.
        # The "search for extra" part is risky if factorization is slow.
        
        # Let's focus on making the convolution step as large as possible.
        # M = 50,000,000? 
        # FFT size 2^26 = 67M. 
        # Memory: 2 arrays of size 2^26 complex128 -> 2 * 67M * 16 bytes ~ 2GB. 
        # This might be too much for the environment. 
        # Let's try M = 20,000,000. Size 2^25 = 33M. 2 * 33M * 16 = 1GB. 
        # Still risky.
        # M = 10,000,000. Size 2^24 = 16M. 2 * 16M * 16 = 512MB. Safe.
        
        pass

    def generate_powerful(M):
        pw = set()
        # a = u^2 v^3
        max_v = int(M ** (1/3)) + 2
        for v in range(1, max_v + 1):
            v3 = v ** 3
            if v3 > M:
                break
            max_u = math.isqrt(M // v3)
            for u in range(1, max_u + 1):
                a = u * u * v3
                if a <= M:
                    pw.add(a)
        return sorted(pw)

    # Step 1: Generate powerful numbers up to M
    # We want to maximize the number of pairs.
    # The number of pairs for sum S is roughly proportional to the number of powerful numbers <= S/2.
    # To get a high count, we need a large M.
    
    # Let's estimate the time for FFT.
    # M = 20,000,000.
    # P length ~ 20,000? (Density of powerful numbers is ~ x^(1/2) * const? No, count is ~ x^(1/2) / (2*pi) ? 
    # Actually, count of powerful numbers <= x is ~ x^(1/2) / (2*pi) * sqrt(6/pi)? 
    # Empirically, for x=10^7, count is ~ 10k.
    # For x=10^8, count is ~ 30k?
    
    # Let's try M = 5,000,000 first to be safe, then increase if time permits.
    # Actually, the current program uses M=10^7 and takes ~10-20s for the O(K^2) part.
    # FFT should be much faster.
    
    M = 10_000_000
    
    P = generate_powerful(M)
    P_set = set(P)
    K = len(P)
    
    if time.time() > deadline - 10:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Create indicator array
    # Size must be at least 2*M + 1 to capture sums up to 2*M.
    # Next power of 2.
    size = 1
    while size < 2 * M + 1:
        size *= 2
        
    # Check memory
    # size * 16 bytes * 2 (real/imag or complex)
    # For M=10^7, size=2^24=16,777,216.
    # 16M * 16 bytes = 256 MB per array. Two arrays = 512 MB.
    # This is acceptable.
    
    a = np.zeros(size, dtype=np.float64)
    for p in P:
        a[p] = 1.0
        
    # FFT
    fa = np.fft.rfft(a)
    fa_sq = fa * fa
    conv = np.fft.irfft(fa_sq, n=size)
    
    # conv[i] is the number of pairs (x, y) such that x + y = i and x, y are in P.
    # Note: This counts (x, y) and (y, x) as distinct if x != y.
    # We want distinct pairs a <= b.
    # Count(a,b) with a<b is (conv[S] - conv[S/2 if S even]) / 2 ?
    # Let C[S] = sum_{x+y=S} 1.
    # If S is even, one pair is (S/2, S/2) if S/2 in P.
    # Let D[S] be number of distinct pairs a<=b.
    # D[S] = (C[S] - (1 if S even and S/2 in P else 0)) / 2 + (1 if S even and S/2 in P else 0)
    # D[S] = (C[S] + (1 if S even and S/2 in P else 0)) / 2
    
    # We want to maximize D[S].
    # Let's compute D for all S up to 2*M.
    
    # Efficiently find max D[S].
    # We can just look at the top few S values from conv.
    
    # Get indices of top 100 sums
    # conv is length size. We only care up to 2*M.
    max_sum = 2 * M
    conv_valid = conv[:max_sum + 1]
    
    # Find top 100 indices
    top_indices = np.argpartition(conv_valid, -100)[-100:]
    
    best_N = 0
    best_count = 0
    best_pairs = []
    
    for idx in top_indices:
        S = int(idx)
        if S < 2:
            continue
            
        # Calculate exact distinct pair count
        # C = int(conv_valid[S])
        # Check if S/2 is in P
        if S % 2 == 0:
            half = S // 2
            if half in P_set:
                distinct_count = (C + 1) // 2
            else:
                distinct_count = C // 2
        else:
            distinct_count = C // 2
            
        if distinct_count > best_count:
            best_count = distinct_count
            best_N = S
            # Generate pairs
            pairs = []
            for a_val in P:
                if a_val > S // 2:
                    break
                b_val = S - a_val
                if b_val in P_set:
                    pairs.append([a_val, b_val])
            best_pairs = pairs
            
    if time.time() > deadline - 5:
        if best_N > 0:
            return {'N': best_N, 'pairs': best_pairs}
        else:
            return {'N': 2, 'pairs': [[1, 1]]}
    
    # Now, can we lift?
    # The current program lifts by multiplying N and pairs by k^2.
    # N_final = N_small * k^2
    # This works because if a+b=N, then k^2*a + k^2*b = k^2*N.
    # And if a is powerful, k^2*a is powerful?
    # a = prod p_i^{e_i}. k^2 = prod q_j^{2 f_j}.
    # k^2 * a = prod p_i^{e_i} prod q_j^{2 f_j}.
    # Exponents are e_i (>=2) and 2 f_j (>=2).
    # So yes, scaling by a square preserves powerfulness.
    
    # We want to maximize the score. The score is the number of pairs.
    # Scaling does not change the number of pairs.
    # So why scale? 
    # The task says: "Integers up to 10^18".
    # "The score is the number of pairs".
    # So scaling doesn't help the score directly, unless we can find MORE pairs in the larger N?
    # The current program just scales the pairs. It doesn't find new ones.
    # However, if we return a small N, the score is the same.
    # But maybe the evaluator prefers larger N? No, "score is the number of pairs".
    # "Maximize."
    
    # Wait, if I can find more pairs for a larger N without scaling, that would be better.
    # But finding pairs for large N is hard.
    # The "lift" strategy allows us to use a small N to generate a valid large N.
    # But the score is the same.
    
    # Is there a way to get MORE pairs?
    # If we choose M larger, we might find a small N with more pairs.
    # Let's see if we can increase M.
    
    # Current M = 10^7.
    # If we use M = 5 * 10^7, size = 2^26 = 67M.
    # Memory: 67M * 16 * 2 = 2GB. Might fail.
    # Let's try M = 2 * 10^7. Size 2^25 = 33M. 33M * 16 * 2 = 1GB.
    # Let's try to detect if we have enough memory/time.
    
    # For now, let's stick with M=10^7 to be safe, but use the FFT result.
    # The FFT result should be much faster than the O(K^2) loop.
    # This saves time. Can we use the saved time to do a "search for extra pairs" on the scaled N?
    
    # N_final = best_N * k^2
    # k = isqrt(10^18 // best_N)
    
    k = int(math.isqrt(10**18 // best_N))
    if k == 0: k = 1
    N_final = best_N * k * k
    
    # Scale pairs
    pairs_final = []
    for a, b in best_pairs:
        pairs_final.append([a * k * k, b * k * k])
    
    # Can we find more pairs for N_final?
    # We have ~10 seconds left.
    # We can try to find a, b powerful such that a + b = N_final.
    # We know the scaled ones.
    # Are there others?
    # a must be powerful.
    # If a is not a multiple of k^2, it might still be powerful.
    # Example: N=10, pairs (1,9). Scale by 2^2=4. N=40. Pairs (4,36).
    # Are there other pairs for 40?
    # 40 = 1 + 39 (39 not powerful)
    # 40 = 4 + 36 (yes)
    # 40 = 9 + 31 (no)
    # 40 = 16 + 24 (24 not powerful, 24=2^3*3, exp of 3 is 1)
    # 40 = 25 + 15 (no)
    # 40 = 36 + 4 (dup)
    
    # It is unlikely to find many extra pairs quickly without a powerful number generator for large numbers.
    # But we can check small powerful numbers a.
    # Generate powerful numbers up to, say, 10^6.
    # For each a, check if N_final - a is powerful.
    
    # How to check if X is powerful quickly?
    # If X is large, factorization is slow.
    # But we can use the property: X is powerful iff X = y^2 z^3.
    # This doesn't help with direct lookup.
    
    # Let's just return the scaled pairs. The structural change is the FFT method.
    
    return {'N': N_final, 'pairs': pairs_final}

# To ensure we actually try a larger M if possible, we could wrap the main logic 
# in a try block or check time more carefully. 
# But the prompt asks for a complete program. 
# I will stick to M=10^7 for safety, as FFT is the key improvement over O(K^2).
