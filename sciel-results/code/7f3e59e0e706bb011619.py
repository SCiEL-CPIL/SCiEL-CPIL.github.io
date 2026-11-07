import math
import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Fallback
    best_M = 333312
    best_k = [336, 372, 434]
    
    # Try to find better collisions
    # Increase N significantly using NumPy
    N = 40_000_000
    
    try:
        # Check memory? We can't easily. Just try.
        sigma = np.zeros(N + 1, dtype=np.int64)
        
        # Sieve
        # To speed up the Python loop overhead, we can step by primes?
        # No, sigma needs all divisors.
        # But we can skip even numbers in the outer loop? No, divisors can be anything.
        
        # Optimization: 
        # The inner operation `sigma[i::i] += i` is O(N/i).
        # Total ops N log N.
        
        # We can parallelize? No, single process.
        
        # Let's just run it.
        for i in range(1, N + 1):
            if time.time() > deadline - 20.0:
                # If we are running out of time, abort and use fallback
                # because partial sigma is invalid
                raise TimeoutError()
            sigma[i::i] += i
        
        # Compute M
        ks = np.arange(1, N + 1, dtype=np.int64)
        Ms = ks * sigma
        
        # Find max frequency
        # np.unique is the standard way
        # To save memory, we can delete sigma and ks after computing Ms?
        del sigma
        del ks
        
        if time.time() > deadline - 10.0:
            raise TimeoutError()
            
        # Get unique values and counts
        # return_inverse allows us to map back to indices if needed, 
        # but we just need the value and then filter.
        vals, counts, _ = np.unique(Ms, return_counts=True, return_index=True)
        
        del Ms
        
        max_count = np.max(counts)
        
        if max_count > len(best_k):
            max_val_idx = np.argmax(counts)
            new_M = int(vals[max_val_idx])
            
            # We deleted Ms. We need to recompute or keep it?
            # Recomputing Ms takes time and memory again.
            # Better to keep Ms or compute mask before deleting.
            
            # Let's restructure:
            # 1. Compute Ms
            # 2. Find max_val
            # 3. Filter Ms == max_val to get indices
            # 4. Delete Ms
            
            # So we shouldn't delete Ms before filtering.
            
            # Let's restart the logic in the try block properly.
            
    except Exception:
        pass

    # The previous block was a draft. Let's write the clean version.
    
    # We will use a smaller N if memory fails? 
    # Let's try N=20,000,000 to be safe on memory (20M*8*3 = 480MB).
    # And if time allows, we can't easily extend.
    
    # Let's stick to N=20,000,000.
    
    # Actually, let's look at the current score 1.6667.
    # This corresponds to 5 solutions.
    # With N=10^7, we found 5.
    # With N=2*10^7, we might find 6 or 7.
    
    # Let's implement N=20,000,000.
    
    # Re-implementing cleanly:
    
    try:
        N = 20_000_000
        sigma = np.zeros(N + 1, dtype=np.int64)
        
        for i in range(1, N + 1):
            if time.time() > deadline - 30.0:
                raise TimeoutError()
            sigma[i::i] += i
            
        ks = np.arange(1, N + 1, dtype=np.int64)
        Ms = ks * sigma
        
        # Free memory
        del sigma
        del ks
        
        if time.time() > deadline - 20.0:
            raise TimeoutError()
            
        vals, counts, _ = np.unique(Ms, return_counts=True, return_index=True)
        
        max_count = np.max(counts)
        
        if max_count > len(best_k):
            max_val_idx = np.argmax(counts)
            new_M = int(vals[max_val_idx])
            
            # Find all k
            # We need Ms to filter. 
            # We deleted it. 
            # We should NOT delete Ms before filtering.
            
            # So:
            # 1. Compute Ms
            # 2. Unique
            # 3. Get max M
            # 4. Filter Ms
            # 5. Delete Ms
            
            # Let's do that.
            
            # Wait, I deleted Ms in the draft. 
            # In the final code, I will not delete until after filtering.
            
            # But I already deleted it in the draft logic above.
            # I will rewrite the try block.
            
    except Exception:
        pass

    # Corrected Logic:
    
    try:
        N = 20_000_000
        sigma = np.zeros(N + 1, dtype=np.int64)
        
        for i in range(1, N + 1):
            if time.time() > deadline - 30.0:
                raise TimeoutError()
            sigma[i::i] += i
            
        ks = np.arange(1, N + 1, dtype=np.int64)
        Ms = ks * sigma
        
        del sigma
        del ks
        
        if time.time() > deadline - 20.0:
            raise TimeoutError()
            
        vals, counts, _ = np.unique(Ms, return_counts=True, return_index=True)
        
        max_count = np.max(counts)
        
        if max_count > len(best_k):
            max_val_idx = np.argmax(counts)
            new_M = int(vals[max_val_idx])
            
            # Filter
            mask = Ms == new_M
            new_ks = np.where(mask)[0] + 1
            
            del Ms
            del vals
            del counts
            del mask
            
            new_k_list = [int(x) for x in new_ks]
            
            # Update best
            best_M = new_M
            best_k = new_k_list
            
    except Exception:
        pass

    # Final check
    if best_M > 10**18:
        best_M = 333312
        best_k = [336, 372, 434]
        
    sqrt_M = math.isqrt(best_M)
    valid_k = [k for k in best_k if k <= sqrt_M]
    
    if len(valid_k) < len(best_k):
        best_k = valid_k
        
    if len(best_k) == 0:
        best_M = 333312
        best_k = [336, 372, 434]

    return {'M': int(best_M), 'k': list(best_k)}
