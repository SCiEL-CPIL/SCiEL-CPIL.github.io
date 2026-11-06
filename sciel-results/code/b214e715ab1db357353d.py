import time
import numpy as np
from math import isqrt, gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # 1. Generate powerful numbers up to L_fft
    # Count of powerful numbers ~ 3.7 * sqrt(N)
    # For N=2*10^7, count ~ 12k.
    # For N=10^8, count ~ 116k.
    # FFT size 2*N.
    # Let's try N_fft = 5 * 10^6 to be safe on time/memory, 
    # but we will also do a targeted search for larger N.
    
    L_fft = 5 * 10**6
    
    powerful_set = set()
    max_b = int(L_fft ** (1.0/3.0)) + 1
    for b in range(1, max_b + 1):
        b3 = b * b * b
        if b3 > L_fft:
            break
        max_a = isqrt(L_fft // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            powerful_set.add(n)
    
    powerful_list = sorted(powerful_set)
    
    # 2. FFT Convolution for N in [2, 2*L_fft]
    max_N_fft = 2 * L_fft
    arr = np.zeros(max_N_fft + 1, dtype=np.int32)
    for p in powerful_list:
        if p <= max_N_fft:
            arr[p] = 1
            
    # Perform convolution
    # Use fftconvolve or manual FFT? scipy is available.
    from scipy.signal import fftconvolve
    counts = fftconvolve(arr, arr, mode='full')
    
    best_N = 2
    best_count = 0
    # Find top candidates
    # We need to be careful with the "unordered" count logic
    # counts[N] is the number of ordered pairs (a,b) with a+b=N.
    # If a != b, each unordered pair is counted twice.
    # If a == b, it is counted once.
    # So unordered_count = (counts[N] + (1 if N%2==0 and N//2 in powerful_set else 0)) // 2
    
    candidate_Ns = []
    for N in range(2, min(max_N_fft + 1, len(counts))):
        ordered = int(counts[N])
        if ordered == 0:
            continue
        if N % 2 == 0 and (N // 2) in powerful_set:
            unordered = (ordered + 1) // 2
        else:
            unordered = ordered // 2
        if unordered > best_count:
            best_count = unordered
            best_N = N
            candidate_Ns = [(N, unordered)]
        elif unordered == best_count:
            candidate_Ns.append((N, unordered))
            
    # 3. Targeted Search for larger N
    # Hypothesis: N with many representations are multiples of a "core" powerful LCM.
    # Let's define a core set of small powerful numbers.
    core_powerfuls = [p for p in powerful_list if p < 1000]
    # Calculate LCM of core powerfuls
    # LCM of [1, 4, 8, 9, 16, 25, 27, 36, 49, 64, 81, 100, 121, 128, 144, 169, 196, 216, 225, 256, 289, 324, 343, 361, 400, 441, 484, 512, 529, 576, 625, 676, 729, 784, 841, 900, 961, 1000 is too big? 
    # 1000 = 10^3 = 2^3 * 5^3. Powerful? No, 10^3 = 1000. 1000 = 2^3 * 5^3. Exponents are 3, which is >= 2. So 1000 is powerful.
    
    # Let's take LCM of powerfuls up to 100.
    # 1, 4, 8, 9, 16, 25, 27, 36, 49, 64, 81, 100
    # LCM(4, 8, 9, 16, 25, 27, 36, 49, 64, 81, 100)
    # 4=2^2, 8=2^3, 9=3^2, 16=2^4, 25=5^2, 27=3^3, 36=2^2*3^2, 49=7^2, 64=2^6, 81=3^4, 100=2^2*5^2
    # Max powers: 2^6, 3^4, 5^2, 7^2
    # L = 64 * 81 * 25 * 49 = 64 * 81 * 1225 = 5184 * 1225 = 6,350,400
    
    LCM_core = 6350400
    
    # Check multiples of LCM_core up to a larger limit, say 10^8
    # We need to check if N - a is powerful for a in powerful_list (extended?)
    # We only have powerful_list up to L_fft. 
    # For N > 2*L_fft, we might miss pairs where a or b > L_fft.
    # But if N is large, and a is small (from core), b = N-a is large.
    # We need to check if b is powerful.
    # We can check if b is powerful by factorization or by checking if b is in a precomputed set?
    # We don't have a set up to 10^8.
    # However, we can implement a fast is_powerful check for specific numbers.
    
    def is_powerful(n):
        if n == 1:
            return True
        x = n
        # Trial division up to cube root? 
        # If x has a prime factor p with exponent 1, return False.
        # If remaining x > 1, it must be a prime (exp 1) or square of prime (exp 2).
        # Actually, if x > 1 after dividing out all small factors, 
        # x must be p^2 or p^k.
        # Since we check all p up to isqrt(x), if x remains, it's prime.
        # So if x > 1, it's a prime with exponent 1 (from the division loop ending) 
        # UNLESS the original number was p^2 and we didn't divide p?
        # My loop divides fully. So if x > 1, it's a prime factor with exponent 1.
        # So return False.
        
        # Optimization: check small primes first.
        # 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for p in small_primes:
            if p * p > x:
                break
            if x % p == 0:
                cnt = 0
                while x % p == 0:
                    x //= p
                    cnt += 1
                if cnt < 2:
                    return False
        if x > 1:
            # x is prime, exponent 1
            return False
        return True

    # Extend search for N = k * LCM_core
    # We check if there are many pairs (a, N-a) where a is powerful and N-a is powerful.
    # We iterate a in powerful_list (up to L_fft).
    # If a > N, break.
    # If N-a > L_fft, we use is_powerful(N-a).
    # If N-a <= L_fft, we use powerful_set.
    
    best_N_target = best_N
    best_count_target = best_count
    
    # How many multiples to check?
    # Time limit 115s.
    # FFT took ~10s.
    # We have ~100s left.
    # Checking one N takes len(powerful_list) operations ~ 10k.
    # 100s / 10k ops ~ 10,000 N's? 
    # Each op is fast (set lookup or is_powerful).
    # is_powerful is slower.
    # Let's check multiples up to 10^8.
    # 10^8 / 6.3e6 ~ 15 multiples.
    # This is very small.
    # We should check more.
    # Maybe use a smaller LCM?
    # LCM of [4, 8, 9, 16, 25] = 64 * 9 * 25 = 14400.
    # 10^8 / 14400 ~ 6944 multiples.
    # This is better.
    
    LCM_small = 14400
    max_N_search = 10**8
    
    # Pre-calculate powerful_set for fast lookup
    # We already have powerful_set up to L_fft.
    # For b > L_fft, we use is_powerful.
    
    # To speed up is_powerful, we can cache results?
    # Or just rely on the fact that we only check ~7000 * 10000 = 70 million ops?
    # 70 million is_powerful calls is too slow.
    # We need to restrict the search.
    
    # Alternative: Only check a in powerful_list where a is small?
    # If N is large, and a is small, b is large.
    # Most pairs will involve small a and large b?
    # Not necessarily.
    
    # Let's just check the top candidates from FFT and a few multiples of LCM_small.
    # Actually, the FFT likely found the global max for N < 10^7.
    # Is it possible that N > 10^7 has more pairs?
    # The density of powerful numbers decreases as 1/sqrt(N).
    # The number of pairs for a fixed N is roughly sum_{a powerful} 1/sqrt(N-a).
    # This sum converges? No, it behaves like integral 1/sqrt(x) dx ~ sqrt(N).
    # So the number of pairs grows as sqrt(N).
    # So larger N should have more pairs!
    # Why did FFT only go up to 10^7? Because of memory.
    # So we MUST search larger N.
    
    # Strategy:
    # For N in range(L_fft, max_N_search, step):
    #   count = 0
    #   for a in powerful_list:
    #      if a > N: break
    #      b = N - a
    #      if b in powerful_set: count += 1
    #      elif is_powerful(b): count += 1
    #   if count > best_count: update
    
    # This is O(N_range / step * len(powerful_list)).
    # If step is large, we miss peaks.
    # But we suspect peaks are at multiples of LCMs.
    
    # Let's check multiples of LCM_small = 14400 up to 10^8.
    # Number of multiples: 10^8 / 14400 ~ 6944.
    # For each, we iterate powerful_list (12k items).
    # Total ops: 6944 * 12000 = 83 million.
    # In Python, 83 million iterations is ~10-20 seconds.
    # This is feasible!
    
    # We need to optimize the inner loop.
    # Use local variables for speed.
    p_set = powerful_set
    p_list = powerful_list
    is_pow_func = is_powerful
    
    current_time = time.time()
    if current_time < deadline - 30:
        for k in range(1, int(max_N_search / LCM_small) + 1):
            if time.time() > deadline - 10:
                break
            N = k * LCM_small
            if N <= best_N:
                continue
            count = 0
            # Iterate a
            # Optimization: if a > N, break.
            # Since N is large, we iterate all p_list.
            for a in p_list:
                if a > N:
                    break
                b = N - a
                if b < 1:
                    break
                if b in p_set:
                    count += 1
                else:
                    if is_pow_func(b):
                        count += 1
            # Unordered count adjustment?
            # The loop counts ordered pairs (a,b) where a is from p_list.
            # If a != b, we count (a,b) and (b,a) if b is in p_list?
            # No, we only iterate a in p_list.
            # If b is in p_set, we count it.
            # If b is not in p_set but is powerful, we count it.
            # This counts each unordered pair exactly once?
            # If a < b: a is in p_list. b is powerful. Count 1.
            # When we reach a' = b in p_list? 
            # If b is in p_list, then when a=b, we check b=N-b=a. 
            # Wait, the loop is `for a in p_list`.
            # If a and b are both in p_list, and a != b:
            #   Iteration a: b is in p_set -> count++.
            #   Iteration b: N-b = a is in p_set -> count++.
            # So it counts 2 for each unordered pair with a!=b.
            # If a == b:
            #   Iteration a: b=a is in p_set -> count++.
            #   It counts 1.
            # So `count` is the number of ordered pairs.
            # Unordered = (count + (1 if N%2==0 and N//2 in p_set else 0)) // 2
            
            if N % 2 == 0 and (N // 2) in p_set:
                unordered = (count + 1) // 2
            else:
                unordered = count // 2
                
            if unordered > best_count_target:
                best_count_target = unordered
                best_N_target = N

    # Final Decision
    if best_count_target > best_count:
        best_N = best_N_target
        best_count = best_count_target
        
    # Generate pairs for best_N
    pairs = []
    for a in powerful_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b < 1:
            break
        if b in powerful_set:
            pairs.append([a, b])
        else:
            if is_powerful(b):
                pairs.append([a, b])
                
    pairs.sort()
    
    return {'N': best_N, 'pairs': pairs}
