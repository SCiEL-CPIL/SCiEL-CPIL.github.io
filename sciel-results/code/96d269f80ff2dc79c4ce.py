"""FFT-based autocorrelation for finding N with many powerful-number pair representations."""
import time
import random
from math import isqrt
from collections import Counter
import numpy as np

def _is_square_free(n):
    """Check if n is square-free."""
    if n < 2:
        return True
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        if p * p > n:
            break
        if n % (p * p) == 0:
            return False
        while n % p == 0:
            n //= p
    if n == 1:
        return True
    r = isqrt(n)
    if r * r == n:
        return False
    return True

def _generate_powerful(limit):
    """Generate all powerful numbers <= limit as a^2 * b^3 where b is square-free."""
    powerful = set([1])
    max_b = int(limit ** (1/3)) + 2
    
    # Sieve for square-free numbers up to max_b
    sq_free = [True] * (max_b + 1)
    if max_b >= 0:
        sq_free[0] = False
    if max_b >= 1:
        sq_free[1] = True
    limit_sieve = isqrt(max_b) + 1
    for p in range(2, limit_sieve):
        if sq_free[p]:
            p2 = p * p
            for k in range(p2, max_b + 1, p2):
                sq_free[k] = False
    
    for b in range(1, max_b + 1):
        if not sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
            else:
                break
    
    return sorted(powerful)

def _count_reps(N, powerful_list, powerful_set):
    """Count pairs (a, b) with a <= b, a + b = N, a, b in powerful_set."""
    count = 0
    pairs = []
    for a in powerful_list:
        if 2 * a > N:
            break
        b = N - a
        if b in powerful_set:
            count += 1
            pairs.append([a, b])
    return count, pairs

def _verify_powerful(n):
    """Verify n is powerful by checking factorization (for final validation)."""
    if n == 1:
        return True
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            cnt = 0
            while temp % d == 0:
                cnt += 1
                temp //= d
            if cnt < 2:
                return False
        d += 1 if d == 2 else 2
    if temp > 1:
        return False  # remaining factor has exponent 1
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    best_N = None
    best_count = 0
    best_pairs = []
    
    # Generate powerful numbers up to 10^7 for FFT feasibility
    # Sums will be up to 2*10^7, FFT size ~2^25 is manageable
    limit_pw = 10**7
    
    powerful = _generate_powerful(limit_pw)
    powerful_set = set(powerful)
    n_pw = len(powerful)
    
    elapsed = time.time() - start_time
    print(f"Generated {n_pw} powerful numbers up to {limit_pw} in {elapsed:.2f}s")
    
    if n_pw == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # FFT-based autocorrelation to find high-frequency sums
    # Create indicator array
    max_sum = 2 * limit_pw
    fft_size = 1
    while fft_size < max_sum:
        fft_size *= 2
    
    # Indicator function: 1 if powerful, 0 otherwise
    indicator = np.zeros(fft_size, dtype=np.float64)
    for p in powerful:
        if p < fft_size:
            indicator[p] = 1.0
    
    elapsed = time.time() - start_time
    print(f"Indicator array created, FFT size={fft_size}, in {elapsed:.2f}s")
    
    # Compute autocorrelation via FFT
    # autocorr[k] = sum_i indicator[i] * indicator[i+k]
    # This equals the number of pairs (a,b) with a+b=k where a,b are powerful
    # (but includes both (a,b) and (b,a), and (a,a) once)
    
    fft_indicator = np.fft.fft(indicator)
    autocorr = np.fft.ifft(fft_indicator * np.conjugate(fft_indicator)).real
    
    # The autocorrelation at index k gives count of (i,j) with j-i=k
    # We want sums: a+b=N, so we need indicator[a]*indicator[N-a]
    # That's the convolution, not autocorrelation in the standard sense
    # Actually, let me think again...
    
    # For sums: number of pairs (a,b) with a+b=N is sum_a indicator[a]*indicator[N-a]
    # This is the cross-correlation of indicator with itself at lag N
    # Or equivalently, the convolution indicator * indicator at position N
    
    # Let's use convolution instead
    fft_indicator = np.fft.fft(indicator)
    conv = np.fft.ifft(fft_indicator * fft_indicator).real
    
    # conv[N] = sum_i indicator[i] * indicator[N-i] = number of ordered pairs (a,b) with a+b=N
    # For unordered pairs with a<=b, we need to be careful
    # conv[N] counts (a,b) and (b,a) separately for a!=b, and (a,a) once
    # So unordered count = (conv[N] + (1 if N is even and N/2 is powerful)) / 2
    
    # Find top K candidates by convolution value
    # Only consider N in range [2, 10^18]
    valid_range = min(max_sum, fft_size)
    top_indices = np.argsort(conv[:valid_range])[::-1][:200]
    
    elapsed = time.time() - start_time
    print(f"FFT convolution done in {elapsed:.2f}s")
    
    # Verify top candidates
    for idx in top_indices:
        if time.time() > deadline - 20:
            break
        N_cand = int(idx)
        if N_cand < 2 or N_cand > 10**18:
            continue
        
        # Get exact count
        count, pairs = _count_reps(N_cand, powerful, powerful_set)
        if count > best_count:
            best_count = count
            best_N = N_cand
            best_pairs = pairs
            print(f"FFT candidate: N={N_cand}, exact count={count}, FFT estimate={conv[N_cand]:.1f}")
    
    # Also try structured approach with larger powerful set for verification
    # Generate more powerful numbers for verification of candidates
    if time.time() < deadline - 30:
        limit_pw2 = 10**9
        powerful2 = _generate_powerful(limit_pw2)
        powerful_set2 = set(powerful2)
        
        elapsed = time.time() - start_time
        print(f"Generated {len(powerful2)} powerful numbers up to {limit_pw2} in {elapsed:.2f}s")
        
        # Re-verify best candidates with larger set
        if best_N is not None:
            count2, pairs2 = _count_reps(best_N, powerful2, powerful_set2)
            if count2 > best_count:
                best_count = count2
                best_pairs = pairs2
                print(f"Re-verified with larger set: N={best_N}, count={count2}")
        
        # Try structured candidates
        R_candidates = [1, 8, 27, 125, 343, 1331, 2197, 4913, 15625, 28561, 42875, 10648, 27000, 3375, 421875, 9261000]
        primes_4k1 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317]
        
        for R in R_candidates:
            if time.time() > deadline - 10:
                break
            
            M_max = 10**18 // R
            if M_max < 2:
                continue
            
            product = 1
            for k, p in enumerate(primes_4k1):
                new_product = product * p
                M_cand = new_product * new_product
                if M_cand > M_max:
                    break
                
                N_cand = R * M_cand
                if N_cand < 2 or N_cand > 10**18:
                    break
                
                count, pairs = _count_reps(N_cand, powerful2, powerful_set2)
                if count > best_count:
                    best_count = count
                    best_N = N_cand
                    best_pairs = pairs
                    print(f"Structured new best: N={N_cand}, count={count}")
                
                product = new_product
            
            product = 1
            for k, p in enumerate(primes_4k1):
                new_product = product * p
                M_cand = 4 * new_product * new_product
                if M_cand > M_max:
                    break
                
                N_cand = R * M_cand
                if N_cand < 2 or N_cand > 10**18:
                    break
                
                count, pairs = _count_reps(N_cand, powerful2, powerful_set2)
                if count > best_count:
                    best_count = count
                    best_N = N_cand
                    best_pairs = pairs
                    print(f"Structured(4x) new best: N={N_cand}, count={count}")
                
                product = new_product
    
    # Final verification of best result
    if best_N is not None and best_count > 0:
        verified_pairs = []
        for a, b in best_pairs:
            if a + b == best_N and a <= b and a >= 1 and b >= 1:
                if _verify_powerful(a) and _verify_powerful(b):
                    verified_pairs.append([a, b])
        
        if len(verified_pairs) > 0:
            return {'N': best_N, 'pairs': verified_pairs}
    
    # Fallback
    return {'N': 2, 'pairs': [[1, 1]]}
