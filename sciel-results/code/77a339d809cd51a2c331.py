import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Try a larger L first
    L_candidates = [5 * 10**7, 3 * 10**7, 2 * 10**7]
    
    for L in L_candidates:
        try:
            if time.time() > deadline - 20:
                continue # Not enough time for this L
            
            # 1. Generate powerful numbers <= L
            powerful_set = set()
            max_y = int(L ** (1/3)) + 1
            
            sq_free = []
            # Simple sieve for square-free numbers up to max_y
            # max_y is small (~370 for 5e7)
            for y in range(1, max_y + 1):
                is_sq_free = True
                temp_y = y
                d = 2
                while d * d <= temp_y:
                    if temp_y % (d * d) == 0:
                        is_sq_free = False
                        break
                    d += 1
                if is_sq_free:
                    sq_free.append(y)
            
            for y in sq_free:
                y_cubed = y ** 3
                if y_cubed > L:
                    continue
                max_x = int(math.isqrt(L // y_cubed))
                for x in range(1, max_x + 1):
                    val = x * x * y_cubed
                    powerful_set.add(val)
            
            if time.time() > deadline - 15:
                continue # Not enough time for FFT
            
            P = sorted(powerful_set)
            
            # 2. FFT Convolution using rfft for efficiency
            n_fft = 1
            while n_fft < 2 * L + 1:
                n_fft *= 2
                
            # Check if n_fft is too large for memory
            # 2^27 ~ 1.34e8. float64 array ~ 1GB.
            # If n_fft > 2^27, it's risky.
            if n_fft > 2**27:
                continue
            
            A = np.zeros(n_fft, dtype=np.float64)
            for p in P:
                A[p] = 1.0
                
            F = np.fft.rfft(A)
            C = np.fft.irfft(F * F, n=n_fft)
            C = np.round(C).astype(np.int64)
            
            # 3. Find max count
            # Valid N range: 2 to 2L
            max_idx = np.argmax(C)
            best_N = int(max_idx)
            
            if best_N < 2 or best_N > 2 * L:
                best_N = 2
            
            # 4. Retrieve pairs
            pairs = []
            powerful_set_lookup = powerful_set
            half_best = best_N // 2
            
            for a in P:
                if a > half_best:
                    break
                b = best_N - a
                if b in powerful_set_lookup:
                    pairs.append([a, b])
            
            # Return the result from the largest L that succeeded
            return {'N': best_N, 'pairs': pairs}

        except MemoryError:
            continue
        except Exception:
            continue
            
    # Fallback to smallest L if all else fails
    L = 2 * 10**7
    powerful_set = set()
    max_y = int(L ** (1/3)) + 1
    sq_free = []
    for y in range(1, max_y + 1):
        is_sq_free = True
        temp_y = y
        d = 2
        while d * d <= temp_y:
            if temp_y % (d * d) == 0:
                is_sq_free = False
                break
            d += 1
        if is_sq_free:
            sq_free.append(y)
            
    for y in sq_free:
        y_cubed = y ** 3
        if y_cubed > L:
            continue
        max_x = int(math.isqrt(L // y_cubed))
        for x in range(1, max_x + 1):
            val = x * x * y_cubed
            powerful_set.add(val)
            
    P = sorted(powerful_set)
    n_fft = 1
    while n_fft < 2 * L + 1:
        n_fft *= 2
        
    A = np.zeros(n_fft, dtype=np.float64)
    for p in P:
        A[p] = 1.0
        
    F = np.fft.rfft(A)
    C = np.fft.irfft(F * F, n=n_fft)
    C = np.round(C).astype(np.int64)
    
    max_idx = np.argmax(C)
    best_N = int(max_idx)
    
    pairs = []
    powerful_set_lookup = powerful_set
    half_best = best_N // 2
    
    for a in P:
        if a > half_best:
            break
        b = best_N - a
        if b in powerful_set_lookup:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
