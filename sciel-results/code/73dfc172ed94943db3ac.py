"""
Strategy: FFT-based convolution to compute the powerful-number pair-count function r(N)
for ALL N up to 2L in O(L log L) time, replacing the O(P^2) pairwise enumeration.

This is a structural change: instead of generating all P^2/2 pairs and bincounting,
we use the indicator array of powerful numbers and convolve it with itself via FFT.
This allows L to scale to ~2e8 (vs ~1e7 for the O(P^2) approach) within 150s,
exploring a much wider range of N where denser sums are expected.

After finding the top candidates from the convolution, we verify exact pair counts
using set membership (O(P) per candidate) and extract the actual pairs.
"""

import time
from math import isqrt
import numpy as np
import bisect

def _generate_powerfuls(limit):
    """
    Generate all powerful numbers <= limit.
    A powerful number has the form a^2 * b^3 where b is square-free.
    Uses a sieve for square-free detection for speed.
    """
    powerfuls = set()
    max_b = int(limit ** (1.0 / 3.0)) + 2
    
    # Sieve for primes up to sqrt(max_b)
    sqrt_max_b = isqrt(max_b)
    is_prime = [True] * (sqrt_max_b + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(sqrt_max_b) + 1):
        if is_prime[i]:
            for j in range(i * i, sqrt_max_b + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, sqrt_max_b + 1) if is_prime[i]]
    
    for b in range(1, max_b + 1):
        # Check if b is square-free
        is_sq_free = True
        for p in primes:
            if p * p > b:
                break
            if b % (p * p) == 0:
                is_sq_free = False
                break
        if not is_sq_free:
            continue
        
        b3 = b ** 3
        if b3 > limit:
            break
        
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerfuls.add(val)
    
    return powerfuls


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # Try progressively larger limits with FFT convolution
    # FFT of size ~2L: for L=2e8, FFT size ~5.4e8, takes ~5-10s
    # Powerful generation for L=2e8 takes ~3-5s
    # Total should fit in 150s
    
    L = 200_000_000
    
    # Check time after generation
    powerfuls = _generate_powerfuls(L)
    powerfuls_list = sorted(powerfuls)
    powerfuls_set = set(powerfuls_list)
    P = len(powerfuls_list)
    
    elapsed_gen = time.time() - start_time
    
    if time.time() > deadline - 40:
        # Not enough time for FFT, fall back to smaller L with direct method
        L_small = 5_000_000
        small_pows = [p for p in powerfuls_list if p <= L_small]
        small_set = set(small_pows)
        count_dict = {}
        pair_dict = {}
        for i in range(len(small_pows)):
            a = small_pows[i]
            for j in range(i, len(small_pows)):
                b = small_pows[j]
                s = a + b
                if s > 2 * L_small:
                    break
                if s not in count_dict:
                    count_dict[s] = 0
                    pair_dict[s] = []
                count_dict[s] += 1
                pair_dict[s].append([a, b])
        if count_dict:
            best_N = max(count_dict, key=count_dict.get)
            return {'N': best_N, 'pairs': pair_dict[best_N]}
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Build indicator array
    # For FFT, we need a power-of-2 size >= 2*L
    max_sum = 2 * L
    # Next power of 2
    fft_size = 1
    while fft_size < max_sum + 1:
        fft_size <<= 1
    
    # Check memory: fft_size * 8 bytes for complex128
    # For L=2e8, fft_size ~ 5.4e8, complex128 = 16 bytes/elem = ~8.6 GB
    # That's too much! Need to use real FFT or smaller L.
    
    # Use real FFT: numpy's rfft on a real array of size N gives N//2+1 complex
    # But we still need the full array for the input.
    # Alternative: use a smaller L that fits in memory.
    
    # For L=1e8: max_sum=2e8, fft_size=2^28=268435456
    # Input array: 268435456 * 8 bytes (float64) = ~2 GB
    # rfft output: 134217729 * 16 bytes = ~2.1 GB
    # Total ~4-5 GB. Might be tight.
    
    # Let's try L=1e8 first
    if L > 100_000_000:
        L = 100_000_000
        powerfuls = _generate_powerfuls(L)
        powerfuls_list = sorted(powerfuls)
        powerfuls_set = set(powerfuls_list)
        P = len(powerfuls_list)
    
    max_sum = 2 * L
    fft_size = 1
    while fft_size < max_sum + 1:
        fft_size <<= 1
    
    elapsed_check = time.time() - start_time
    
    # Check if we have enough time
    if time.time() > deadline - 50:
        # Fallback to smaller
        L = 30_000_000
        powerfuls = _generate_powerfuls(L)
        powerfuls_list = sorted(powerfuls)
        powerfuls_set = set(powerfuls_list)
        P = len(powerfuls_list)
        max_sum = 2 * L
        fft_size = 1
        while fft_size < max_sum + 1:
            fft_size <<= 1
    
    # Build the indicator array
    try:
        indicator = np.zeros(fft_size, dtype=np.float64)
        for p in powerfuls_list:
            indicator[p] = 1.0
        
        # Compute convolution via FFT
        # rfft of indicator, square it, irfft to get convolution
        fft_result = np.fft.rfft(indicator)
        fft_result *= fft_result
        conv = np.fft.irfft(fft_result, n=fft_size)
        
        # conv[n] = sum of indicator[a] * indicator[n-a] for all a
        # This counts ordered pairs (a, b) with a + b = n
        # For a < b: counted twice in conv (once as (a,b), once as (b,a))
        # For a = b: counted once
        # So r(N) = (conv[N] + (1 if N is even and N/2 is powerful)) / 2
        # Actually: conv[N] = 2 * (# pairs with a < b) + (# pairs with a = b)
        # # pairs with a <= b = (# pairs with a < b) + (# pairs with a = b)
        # = (conv[N] - 1_if_self) / 2 + 1_if_self = (conv[N] + 1_if_self) / 2
        # where 1_if_self = 1 if N is even and N/2 is powerful, else 0
        
        # Find top candidates
        # conv values are floats, round to nearest int for ranking
        conv_rounded = np.round(conv[:max_sum + 1])
        
        # Get top 20 candidates by convolution count
        # Use argpartition for efficiency
        top_k = 20
        if max_sum + 1 > top_k:
            top_indices = np.argpartition(conv_rounded, -top_k)[-top_k:]
        else:
            top_indices = np.arange(max_sum + 1)
        
        # Sort by count descending
        top_indices = top_indices[np.argsort(conv_rounded[top_indices])[::-1]]
        
        best_N = None
        best_pairs = None
        best_count = 0
        
        for N_candidate in top_indices:
            N_candidate = int(N_candidate)
            if N_candidate < 2:
                continue
            
            # Verify exact count using set membership
            half_N = N_candidate // 2
            end_idx = bisect.bisect_right(powerfuls_list, half_N)
            
            pairs = []
            for i in range(end_idx):
                a = powerfuls_list[i]
                b = N_candidate - a
                if b in powerfuls_set:
                    if a <= b:
                        pairs.append([a, b])
            
            count = len(pairs)
            if count > best_count:
                best_count = count
                best_N = N_candidate
                best_pairs = pairs
            
            # Time check
            if time.time() > deadline - 20:
                break
        
        if best_pairs and best_count > 0:
            return {'N': best_N, 'pairs': best_pairs}
        
    except MemoryError:
        # Fall back to smaller L with direct O(P^2) method
        L = 10_000_000
        powerfuls = _generate_powerfuls(L)
        powerfuls_list = sorted(powerfuls)
        powerfuls_set = set(powerfuls)
        P = len(powerfuls_list)
        
        count_dict = {}
        pair_dict = {}
        for i in range(P):
            a = powerfuls_list[i]
            for j in range(i, P):
                b = powerfuls_list[j]
                s = a + b
                if s > 2 * L:
                    break
                if s not in count_dict:
                    count_dict[s] = 0
                    pair_dict[s] = []
                count_dict[s] += 1
                pair_dict[s].append([a, b])
        
        if count_dict:
            best_N = max(count_dict, key=count_dict.get)
            return {'N': best_N, 'pairs': pair_dict[best_N]}
    
    except Exception:
        pass
    
    # Final fallback
    return {'N': 2, 'pairs': [[1, 1]]}
