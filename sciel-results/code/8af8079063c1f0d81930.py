import math
import time
import random
import array

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # Try to use a large sieve
    # N = 10^8
    # If it fails or takes too long, fall back to smaller or random
    
    # We'll just do N=10^8 directly.
    # If MemoryError, we catch it.
    
    try:
        N = 10**8
        phi = array.array('I', range(N + 1))
        
        # Sieve
        # Optimization: only loop i up to N
        for i in range(2, N + 1):
            if phi[i] == i:
                for j in range(i, N + 1, i):
                    phi[j] -= phi[j] // i
            
            # Time check every 10^6 iterations?
            # Or just rely on total time.
            # The inner loop is fast.
            
            if i % 10**6 == 0 and time.time() - start_time > DEADLINE - 20:
                raise TimeoutError("Sieve taking too long")

        # Scan
        last_occ = array.array('I', [0]) * (N + 1)
        
        best_k = 1
        best_s = 1
        current_start = 1
        
        for i in range(1, N + 1):
            val = phi[i]
            if last_occ[val] >= current_start:
                current_start = last_occ[val] + 1
            last_occ[val] = i
            
            k = i - current_start + 1
            if k > best_k:
                best_k = k
                best_s = current_start
                
            if i % 10**6 == 0 and time.time() - start_time > DEADLINE - 5:
                break
                
        return {'start': best_s, 'length': best_k}
        
    except Exception as e:
        # Fallback to original logic or smaller N
        # Let's just return a small valid result to be safe
        # Or try N=10^7
        try:
            N = 10**7
            phi = array.array('I', range(N + 1))
            for i in range(2, N + 1):
                if phi[i] == i:
                    for j in range(i, N + 1, i):
                        phi[j] -= phi[j] // i
                if time.time() - start_time > DEADLINE - 10:
                    break
                    
            last_occ = array.array('I', [0]) * (N + 1)
            best_k = 1
            best_s = 1
            current_start = 1
            for i in range(1, N + 1):
                val = phi[i]
                if last_occ[val] >= current_start:
                    current_start = last_occ[val] + 1
                last_occ[val] = i
                k = i - current_start + 1
                if k > best_k:
                    best_k = k
                    best_s = current_start
                if time.time() - start_time > DEADLINE - 2:
                    break
            return {'start': best_s, 'length': best_k}
        except:
            return {'start': 1, 'length': 1}

