import time
import numpy as np
from math import isqrt
from collections import defaultdict

def _powerful_numbers_up_to(M, deadline=None):
    """Generate all powerful numbers up to M."""
    if M < 1:
        return set()
    powerful = set()
    max_v = int(M ** (1/3)) + 2
    for v in range(1, max_v + 1):
        if deadline and time.time() > deadline:
            break
        v_cubed = v ** 3
        if v_cubed > M:
            break
        max_u = isqrt(M // v_cubed)
        for u in range(1, max_u + 1):
            powerful.add(u * u * v_cubed)
    return powerful

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Strategy: Multi-scale FFT with adaptive N selection
    # Use progressively larger M values, performing FFT at each scale
    # Track the best (N, pairs) found across all scales
    
    best_N = 2
    best_pairs = [[1, 1]]
    best_count = 1
    
    # Try multiple scales
    # M determines the max powerful number we consider
    # N can be up to 2*M
    target_Ms = [
        50_000_000,
        25_000_000,
        15_000_000,
        8_000_000,
        5_000_000
    ]
    
    for M in target_Ms:
        if time.time() > deadline - 15:
            break
        
        try:
            # Generate powerful numbers up to M
            powerful_set = _powerful_numbers_up_to(M, deadline - 5)
            powerful_list = sorted(powerful_set)
            K = len(powerful_list)
            
            if K < 2:
                continue
            
            # Prepare FFT
            # We need counts for N up to 2*M
            # Use a slightly larger power of 2 for FFT
            size = 1
            while size < 2 * M + 1:
                size *= 2
            
            # Memory check: float64 array of size `size` needs 8*size bytes
            # Also rfft produces size//2 + 1 complex128 values
            # Total ~ 8*size + 16*(size//2+1) bytes
            # For size=2^28 (~268M), that's ~2.1GB - too much
            # For size=2^27 (~134M), that's ~1GB - borderline
            # For size=2^26 (~67M), that's ~536MB - safer
            if size > 2**27:
                continue
            
            # Create indicator array
            A = np.zeros(size, dtype=np.float64)
            for p in powerful_list:
                A[p] = 1.0
            
            # FFT convolution: count of pairs (a,b) with a+b=N
            F = np.fft.rfft(A)
            F_sq = F * F
            counts_float = np.fft.irfft(F_sq, n=size)
            
            # Focus on N in [M, 2*M] to avoid small N with fewer representations
            # Actually, let's look at a range to find hotspots
            search_start = M // 2
            search_end = min(2 * M, size - 1)
            counts_segment = counts_float[search_start:search_end]
            
            if counts_segment.size == 0:
                continue
            
            max_val = np.max(counts_segment)
            if max_val < 1:
                continue
            
            # Find top candidates
            top_k = min(200, counts_segment.size)
            top_local_indices = np.argpartition(counts_segment, -top_k)[-top_k:]
            top_indices = top_local_indices + search_start
            
            # Verify top candidates with exact counting
            powerful_set_local = powerful_set
            powerful_list_local = powerful_list
            half_N_max = M  # a <= N/2, and a <= M, so a <= min(M, N/2)
            
            for idx in top_indices:
                if time.time() > deadline - 10:
                    break
                N = int(idx)
                if N < 2:
                    continue
                half_N = N // 2
                count = 0
                pairs = []
                
                for a in powerful_list_local:
                    if a > half_N:
                        break
                    b = N - a
                    if b in powerful_set_local:
                        count += 1
                        pairs.append([a, b])
                
                if count > best_count:
                    best_count = count
                    best_N = N
                    best_pairs = pairs
                    
        except MemoryError:
            continue
        except Exception:
            continue
    
    # If we found nothing better, use a small known case
    # N=2: 1+1=2, both powerful (1 is powerful)
    # N=3: 1+2? 2 is not powerful. 
    # N=4: 1+3? 3 not powerful. 2+2? 2 not powerful.
    # N=5: 1+4? 4=2^2 powerful. So 1+4=5.
    # N=6: 2+4? 2 not powerful. 1+5? 5 not powerful.
    # N=7: 3+4? 3 not powerful. 1+6? 6 not powerful.
    # N=8: 4+4=8. Both powerful. 1+7? No.
    # N=9: 1+8? 8=2^3 powerful. So 1+8=9.
    # N=10: 2+8? 2 not powerful. 4+6? No.
    # N=12: 4+8=12. Both powerful.
    # N=16: 4+12? No. 8+8=16. Both powerful.
    # N=18: 1+17? No. 9+9=18. Both powerful.
    
    if best_count <= 1:
        # Try some small N manually to ensure we have a valid answer
        small_candidates = [5, 9, 12, 16, 18, 20, 25, 27, 32, 36, 45, 48, 50, 64, 81, 98, 100]
        for N in small_candidates:
            if time.time() > deadline - 2:
                break
            M_small = N
            p_set = _powerful_numbers_up_to(M_small, deadline - 1)
            half = N // 2
            pairs = []
            for a in sorted(p_set):
                if a > half:
                    break
                b = N - a
                if b in p_set:
                    pairs.append([a, b])
            if len(pairs) > best_count:
                best_count = len(pairs)
                best_N = N
                best_pairs = pairs
    
    return {'N': best_N, 'pairs': best_pairs}
