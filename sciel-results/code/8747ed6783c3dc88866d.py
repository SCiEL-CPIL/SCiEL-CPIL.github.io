"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt

def _generate_powerfuls(L):
    """Generate all powerful numbers up to L."""
    powerfuls = set()
    max_b = int(L ** (1/3)) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= L:
                powerfuls.add(val)
    return sorted(powerfuls)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s buffer for 150s limit
    
    # Strategy:
    # 1. Generate powerful numbers up to a larger limit L.
    # 2. Use FFT-based convolution to count pair sums efficiently.
    # 3. Find N with maximum representation count.
    # 4. Extract pairs for that N.
    
    # Try progressively larger L values, stopping when time runs out
    # or when we find a good result.
    
    # Start with a moderate L, then try to go higher
    L_candidates = [10**8, 2*10**8, 5*10**8, 10**9]
    
    best_result = {'N': 2, 'pairs': [[1, 1]]}
    best_count = 1
    
    for L in L_candidates:
        if time.time() > deadline - 40:
            break
        
        # Generate powerful numbers
        powerfuls_list = _generate_powerfuls(L)
        M = len(powerfuls_list)
        
        if M < 2:
            continue
        
        P = np.array(powerfuls_list, dtype=np.int64)
        max_val = int(P[-1])
        
        # Check if max_val is too large for FFT
        if max_val > 50_000_000:  # 50 million, FFT of size 100M is ~800MB
            continue
        
        # Use FFT to compute convolution
        # We want to count pairs (a, b) with a <= b and a + b = N
        # This is equivalent to: for each N, count = number of a in powerfuls 
        # such that N - a is also in powerfuls and a <= N - a (i.e., a <= N/2)
        
        # Create a boolean array indicating which values are powerful
        size = 2 * max_val + 1
        if size > 100_000_000:
            continue
            
        f = np.zeros(size, dtype=np.float64)
        f[P] = 1.0
        
        # FFT convolution
        fft_size = 1
        while fft_size < 2 * size:
            fft_size *= 2
        
        F = np.fft.rfft(f, n=fft_size)
        F2 = F * F
        conv = np.fft.irfft(F2, n=fft_size)
        
        # conv[k] = number of pairs (a, b) with a + b = k (counting all ordered pairs)
        # For a <= b: if a < b, it's counted twice in conv (as (a,b) and (b,a))
        # If a == b, it's counted once.
        # So for each N, the number of unordered pairs with a <= b is:
        #   (conv[N] + (1 if N is even and N/2 is powerful else 0)) // 2
        
        # But we want distinct pairs a <= b, so:
        # count_unordered = (conv[N] + diag) / 2 where diag = 1 if N/2 is powerful
        
        powerfuls_set = set(powerfuls_list)
        
        # Find N with maximum count
        # Only consider N up to 2*L (since max sum is 2*L)
        max_N = 2 * L
        
        # Get conv values for N in range [2, max_N]
        # conv array has conv[k] for k = 0, 1, ..., fft_size-1
        # We need conv[2 : max_N+1]
        
        conv_slice = conv[2:max_N+1]
        
        # For each N, compute unordered pair count
        # conv_slice[i] corresponds to N = i + 2
        Ns = np.arange(2, max_N + 1)
        
        # diag: 1 if N/2 is an integer and N/2 is in powerfuls_set
        # This is a bit tricky to vectorize, so let's compute it differently
        # For now, approximate: unordered_count ≈ conv_slice / 2
        # This is an upper bound (since we're not subtracting the diagonal correction properly)
        # But it's good enough for finding the max
        
        approx_counts = conv_slice / 2.0
        max_idx = np.argmax(approx_counts)
        best_N = int(Ns[max_idx])
        approx_max = int(approx_counts[max_idx])
        
        # Now extract actual pairs for best_N to get exact count
        pairs = []
        limit_a = best_N // 2
        for a in powerfuls_list:
            if a > limit_a:
                break
            b = best_N - a
            if b in powerfuls_set:
                pairs.append([a, b])
        
        actual_count = len(pairs)
        
        if actual_count > best_count:
            best_count = actual_count
            best_result = {'N': best_N, 'pairs': pairs}
        
        # Check time
        if time.time() > deadline - 15:
            break
    
    # Final safety check
    if not best_result['pairs']:
        best_result = {'N': 2, 'pairs': [[1, 1]]}
    
    return best_result

# EVOLVE-BLOCK-END
