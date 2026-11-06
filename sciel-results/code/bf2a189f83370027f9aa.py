import numpy as np
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def is_powerful(n):
        if n <= 1:
            return True
        # Check if every prime factor appears with exponent >= 2
        # This is fast for n up to 10^18 if we use trial division up to cube root
        # and handle the remainder carefully.
        # Actually, simpler: 
        # n is powerful iff n = u^2 v^3.
        # But checking via factorization is robust.
        # Trial division up to 10^6 is ~78k primes. 10^18 / 10^12 = 10^6.
        # If n > 10^12, we can't trial divide fully.
        # However, we can check: 
        # Let n = p1^e1 ... pk^ek. All ei >= 2.
        # If n is powerful, n is a square. So isqrt(n)^2 == n.
        # If n is a square, let n = m^2. Then m = p1^(e1/2)...
        # This is not sufficient. 12 = 2^2*3 is not powerful. 36=2^2*3^2 is.
        # 36 is a square. 12 is not.
        # Wait, powerful numbers are NOT all squares. 8=2^3 is powerful. 18=2*3^2 is powerful.
        # So "is square" is not necessary.
        
        # Better check:
        # Factorize n.
        # Since n <= 10^18, we can trial divide by primes up to 10^6.
        # If remaining factor > 1, it must be prime or product of two primes > 10^6.
        # If product of two primes p*q, then p,q > 10^6 => n >= 10^12.
        # But we only need to check exponents.
        
        # Optimized is_powerful for n <= 10^18:
        if n == 1: return True
        if n % 4 == 0 or n % 9 == 0 or n % 25 == 0 or n % 49 == 0:
            # Heuristic: many powerful numbers are divisible by small squares
            pass
            
        # Standard trial division is too slow if done naively for many candidates.
        # But we only call it on verified candidates.
        
        # Let's use a faster check.
        # n is powerful iff n is a square times a cube-free part which is 1? No.
        # n = u^2 v^3.
        
        # Algorithm:
        # 1. Trial divide by primes p <= 10^6.
        # 2. Count exponent e. If e == 1, return False.
        # 3. If n_remaining > 1:
        #    It must be p^k where p > 10^6.
        #    If k=1, not powerful.
        #    If k=2, powerful (p^2).
        #    If k=3, powerful (p^3).
        #    If k>=4, p^k = (p^2)^(k/2) if even, or p^3 * (p^2)^... if odd.
        #    Since p > 10^6, p^2 > 10^12. p^3 > 10^18.
        #    So if n_remaining > 10^18, impossible.
        #    If n_remaining <= 10^18:
        #       Check if it's a perfect square. If yes, exp is even >= 2. Powerful.
        #       Check if it's a perfect cube. If yes, exp is multiple of 3 >= 3. Powerful.
        #       If it's a prime (not square, not cube), exp=1. Not powerful.
        #       Note: A number > 1 that is not square and not cube and not prime?
        #       e.g. p^2 q^2? No, we divided by small primes.
        #       If remaining is p^k, p>10^6.
        #       If k=1: prime.
        #       If k=2: square.
        #       If k=3: cube.
        #       If k=4: square.
        #       If k=5: not square, not cube? p^5. 10^6^5 = 10^30 > 10^18.
        #       So k can only be 1, 2, 3 for p > 10^6.
        
        rem = n
        # Precomputed primes up to 10^6? 
        # Generating primes up to 10^6 takes ~0.1s.
        # We can do it once.
        
        # But doing this for every candidate might be slow if many candidates.
        # Let's refine the candidate generation to be very sparse.
        
        return True # Placeholder, will implement proper check below

    # Generate primes up to 10^6
    def sieve(limit):
        is_prime = bytearray([1]) * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i:limit+1:i] = bytearray([0]) * len(range(i*i, limit+1, i))
        return [i for i, b in enumerate(is_prime) if b]
    
    primes = sieve(10**6)
    
    def check_powerful(n):
        if n <= 1: return True
        rem = n
        for p in primes:
            if p * p > rem:
                break
            if rem % p == 0:
                cnt = 0
                while rem % p == 0:
                    rem //= p
                    cnt += 1
                if cnt == 1:
                    return False
        if rem > 1:
            # rem is prime, or p^2, or p^3
            # Check if rem is a perfect square
            r = math.isqrt(rem)
            if r * r == rem:
                return True
            # Check if rem is a perfect cube
            c = int(round(rem ** (1/3)))
            # Adjust for floating point errors
            for cc in [c-1, c, c+1]:
                if cc > 0 and cc**3 == rem:
                    return True
            # If not square and not cube, it's a prime (exp 1)
            return False
        return True

    # Strategy:
    # Powerful numbers are of the form u^2 v^3.
    # We want a + b = N.
    # Let a = u^2 v^3 and b = u'^2 v'^3.
    # We search for N by iterating over small v, v' and looking for u, u'.
    # To maximize the number of pairs, N should be such that there are many (u,v) with u^2 v^3 < N.
    # This happens if N is large.
    # However, we must ensure a+b=N.
    
    # Alternative structural change:
    # Instead of FFT, use "Meet in the Middle" on the powerful numbers themselves?
    # No, the list is too large.
    
    # Let's stick to the current FFT approach but optimize the memory and search range.
    # The previous score was 5.93.
    # The prompt asks for a MATERIAL change.
    # 
    # New Idea: **Direct Enumeration of N via "Collision" of Generators**
    # 
    # We know that powerful numbers are sparse.
    # Let P be the set of powerful numbers.
    # We want to maximize |{ (a,b) in P x P : a+b=N, a<=b }|.
    # 
    # Observation: If we pick N such that N = 2 * (some powerful number), we get at least 1 pair.
    # If N is the sum of two "small" powerful numbers, we might get more?
    # No, the count depends on the global structure.
    # 
    # Let's try a **Randomized Local Search** on N?
    # No, N is discrete and huge.
    #
    # Let's try **Chunked FFT with Smaller M**.
    # The current code uses M=80M.
    # What if we use M=20M but run multiple FFTs for different "offsets" or "scales"?
    # 
    # Actually, the most promising structural change is to **avoid the full FFT** by using the fact that we only need the TOP K counts.
    # 
    # **Sparse Convolution via Hashing?**
    # If the set of powerful numbers is sparse enough, we can iterate over pairs?
    # K ~ 10,000 for M=10^8. Pairs ~ 50 million.
    # 50 million iterations in Python is ~5-10 seconds.
    # This is MUCH faster than FFT (which is ~5-10s too) and uses less memory.
    # And it gives EXACT counts without floating point errors!
    # 
    # Let's test this hypothesis.
    # Generate powerful numbers up to M.
    # Use a set for O(1) lookup.
    # Iterate a in P:
    #   Iterate b in P where b >= a:
    #     N = a + b
    #     count[N] += 1
    #     if b > N - a: break? No, b is just the second element.
    #     We need a <= b.
    #     So for each a, iterate b from a to M.
    #     This is O(K^2 / 2).
    #     If K=10,000, K^2/2 = 50,000,000.
    #     In Python, 50M simple operations might take 20-50 seconds.
    #     This is risky for the 150s limit, but doable if optimized with NumPy?
    #     NumPy doesn't help with the dynamic loop structure easily.
    #     
    #     Can we vectorize?
    #     For a fixed a, b ranges from a to M.
    #     N = a + b.
    #     We want to count how many b in P such that a+b = N.
    #     This is just counting P intersect (N - P).
    #     
    #     Let's try to increase M to get larger K, but keep K^2 manageable.
    #     If M = 10^9, K ~ 100,000?
    #     K^2 = 10^10. Too slow.
    #     
    #     So we are limited to K ~ 10,000-20,000.
    #     This corresponds to M ~ 10^8 - 10^9.
    #     
    #     The FFT approach allows M=10^8 (K~10k) in ~10s.
    #     The O(K^2) approach allows M=10^8 (K~10k) in ~20s.
    #     
    #     Is there a way to search larger N?
    #     The FFT approach finds N up to 2M.
    #     If M=10^8, N up to 2*10^8.
    #     
    #     What if we use **FFT on a subset**?
    #     No.
    #     
    #     **Structural Change: Use `scipy.signal.fftconvolve`?**
    #     It's faster than manual rfft? Maybe.
    #     
    #     **Let's try the O(K^2) approach but with a twist:**
    #     We don't need to count ALL pairs. We just need the MAX count.
    #     We can use a dictionary `counts`.
    #     For a in P:
    #       for b in P:
    #         if b < a: continue
    #         N = a + b
    #         counts[N] = counts.get(N, 0) + 1
    #     
    #     This is O(K^2).
    #     To speed it up, we can stop early if we find a count that is "high"?
    #     No, we don't know the max.
    #     
    #     However, we can use **Numba**? Not allowed (standard lib + numpy/scipy).
    #     
    #     Let's stick to FFT but optimize the **Verification** step.
    #     The current code verifies by iterating P.
    #     
    #     **New Strategy: Multi-Resolution Search**
    #     1. Run FFT on M=10^7 (fast, ~1s). Find top candidates N1.
    #     2. Run FFT on M=10^8 (medium, ~10s). Find top candidates N2.
    #     3. Run FFT on M=5*10^8 (slow, ~60s). Find top candidates N3.
    #     4. Combine and verify.
    #     
    #     This ensures we get a result even if the large FFT fails or times out.
    #     And it might find a better N in the larger range.
    #     
    #     Let's implement this staged approach.

    def run_fft_search(M):
        """
        Generate powerful numbers up to M, perform FFT, return top candidates.
        """
        # Generate powerful numbers
        pw = set()
        max_v = int(M ** (1/3)) + 2
        cubes = [v**3 for v in range(1, max_v+1) if v**3 <= M]
        
        for c in cubes:
            max_u = math.isqrt(M // c)
            for u in range(1, max_u + 1):
                pw.add(u * u * c)
        
        P = sorted(pw)
        K = len(P)
        if K == 0:
            return 0, [], {}
        
        # FFT
        size = 2 * M + 1
        # Use float32 for memory
        arr = np.zeros(size, dtype=np.float32)
        arr[P] = 1.0
        
        fft_arr = np.fft.rfft(arr)
        conv = np.fft.irfft(fft_arr * fft_arr, n=size)
        
        # Find top K candidates
        # We want the top 100 indices
        top_k = 50
        if len(conv) > top_k:
            # argpartition is fast
            top_indices = np.argpartition(conv, -top_k)[-top_k:]
        else:
            top_indices = np.arange(len(conv))
            
        # Sort by value
        top_indices = top_indices[np.argsort(conv[top_indices])[::-1]]
        
        # Verify top candidates
        # We need exact counts and pairs
        # We'll return the best one found in this M range
        best_N = 0
        best_count = 0
        best_pairs = []
        
        P_set = set(P)
        
        for idx in top_indices:
            N_cand = int(idx)
            if N_cand < 2:
                continue
            
            # Verify
            pairs = []
            for a in P:
                if a > N_cand // 2:
                    break
                b = N_cand - a
                if b in P_set:
                    pairs.append([a, b])
            
            if len(pairs) > best_count:
                best_count = len(pairs)
                best_N = N_cand
                best_pairs = pairs
                
            # Early break if conv value is too low
            if conv[idx] < 2 * best_count:
                break
                
        return best_N, best_count, best_pairs

    global_best_N = 2
    global_best_pairs = [[1, 1]]
    global_best_count = 1

    # Stage 1: Small M
    M1 = 5_000_000
    if time.time() < deadline - 10:
        N1, C1, Pairs1 = run_fft_search(M1)
        if C1 > global_best_count:
            global_best_N = N1
            global_best_count = C1
            global_best_pairs = Pairs1

    # Stage 2: Medium M
    M2 = 20_000_000
    if time.time() < deadline - 30:
        N2, C2, Pairs2 = run_fft_search(M2)
        if C2 > global_best_count:
            global_best_N = N2
            global_best_count = C2
            global_best_pairs = Pairs2

    # Stage 3: Large M
    M3 = 80_000_000
    if time.time() < deadline - 60:
        N3, C3, Pairs3 = run_fft_search(M3)
        if C3 > global_best_count:
            global_best_N = N3
            global_best_count = C3
            global_best_pairs = Pairs3

    # Stage 4: Extra Large M (if time permits)
    M4 = 150_000_000
    if time.time() < deadline - 100:
        N4, C4, Pairs4 = run_fft_search(M4)
        if C4 > global_best_count:
            global_best_N = N4
            global_best_count = C4
            global_best_pairs = Pairs4

    return {'N': global_best_N, 'pairs': global_best_pairs}
