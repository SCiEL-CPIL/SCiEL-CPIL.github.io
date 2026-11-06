import time
import numpy as np
from math import isqrt

def construct():
        start_time = time.time()
        deadline = start_time + 110
        
        L = 10**9
        powerfuls = set()
        max_b = int(L ** (1/3)) + 1
        for b in range(1, max_b + 1):
            b3 = b ** 3
            if b3 > L:
                break
            max_a = isqrt(L // b3)
            # Vectorize a loop?
            # a ranges 1..max_a
            # n = a^2 * b3
            # We can generate a^2 for all a and multiply by b3
            a_arr = np.arange(1, max_a + 1)
            n_arr = (a_arr * a_arr) * b3
            # Filter n <= L (already ensured by max_a)
            powerfuls.update(n_arr.tolist())
            
        powerfuls = np.array(sorted(powerfuls))
        n_pows = len(powerfuls)
        
        best_N = 0
        best_count = 0
        # We need to count occurrences of each sum.
        # Since sums can be large, we use a dictionary to accumulate counts.
        # But we want to find the max count efficiently.
        
        # Process in chunks
        chunk_size = 500
        counts = {}
        
        for start in range(0, n_pows, chunk_size):
            if time.time() > deadline - 10:
                break
                
            end = min(start + chunk_size, n_pows)
            chunk = powerfuls[start:end]
            
            # Compute all sums chunk + powerfuls
            # This creates a matrix of size (end-start) x n_pows
            sums = np.add.outer(chunk, powerfuls)
            
            # Get unique sums and their counts within this chunk
            unique_sums, counts_chunk = np.unique(sums, return_counts=True)
            
            # Update global counts
            for i in range(len(unique_sums)):
                s = int(unique_sums[i])
                c = int(counts_chunk[i])
                if s in counts:
                    counts[s] += c
                    new_c = counts[s]
                else:
                    counts[s] = c
                    new_c = c
                    
                if new_c > best_count:
                    best_count = new_c
                    best_N = s
        
        if best_count == 0:
            return {'N': 2, 'pairs': [[1, 1]]}
            
        # Collect pairs for best_N
        # We need to find all pairs (a, b) such that a+b=best_N and a,b in powerfuls
        pows_set = set(powerfuls.tolist())
        pairs = []
        # Iterate through powerfuls
        for a in powerfuls:
            if 2 * a > best_N:
                break
            b_val = best_N - a
            if b_val in pows_set and b_val >= a:
                pairs.append([int(a), int(b_val)])
                
        return {'N': best_N, 'pairs': pairs}
