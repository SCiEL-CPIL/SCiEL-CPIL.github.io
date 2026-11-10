# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict
import numpy as np

def _isqrt(x):
    return math.isqrt(x)

def _sigma_sieve_np(N):
    """Compute sigma(k) for k=1..N using a NumPy sieve."""
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
    return sigma

def construct():
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Start with a known solution
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = len(best_k)
    
    # Try to find better solutions by scanning larger ranges using NumPy
    # The NumPy sieve is much faster, allowing larger N
    for N in [10**6, 2*10**6, 5*10**6, 10**7, 2*10**7]:
        if time.time() > deadline - 20:
            break
        
        try:
            sigma = _sigma_sieve_np(N)
            
            # Group by M value
            # We need to compute M = k * sigma[k] and group
            # To avoid huge memory usage for groups, we can just count and store k for the best ones
            # But we need to find the M with the most k's.
            # We can use a dictionary mapping M to list of k's.
            # However, M can be very large (up to 10^18), so dictionary keys are fine.
            
            # Optimization: Instead of storing all k's for all M's, we can first count frequencies,
            # then find the max frequency, then collect k's for that M.
            # But collecting k's requires another pass or storing them.
            # Given N is up to 2*10^7, storing lists might be memory heavy if many collisions.
            # But collisions are rare. Most M have 1 solution.
            
            # Let's try a single pass with defaultdict(list) and see if memory holds.
            # If N=2*10^7, 20 million entries in a dict is large.
            # Alternative: 
            # 1. Compute M for all k.
            # 2. Sort (M, k) pairs? No, sorting 20M pairs is slow.
            # 3. Use a dictionary to count frequencies first.
            
            freq = defaultdict(int)
            # We also need to recover the k's for the best M.
            # We can store k's in a dict of lists, but only for M's that appear more than once?
            # Or just store all and hope memory is enough.
            # 20M entries in a dict of lists... might be ~1-2GB. Risky.
            
            # Better approach:
            # Pass 1: Count frequencies of M.
            # Pass 2: For the M with max frequency, collect k's.
            
            # Pass 1
            for k in range(1, N + 1):
                M = k * sigma[k]
                if M <= 10**18:
                    freq[M] += 1
            
            # Find best M
            if not freq:
                continue
                
            current_best_M, current_best_count = max(freq.items(), key=lambda x: x[1])
            
            if current_best_count > best_count:
                # Pass 2: Collect k's for current_best_M
                best_k_new = []
                for k in range(1, N + 1):
                    M = k * sigma[k]
                    if M == current_best_M:
                        best_k_new.append(k)
                
                best_M = current_best_M
                best_k = best_k_new
                best_count = current_best_count
                
                # Early exit if we find a very good solution
                if best_count >= 20:
                    break
                    
        except MemoryError:
            break
        except Exception as e:
            print(f"Error in N={N}: {e}")
            break

    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in best_k]
    
    # Verify the solution
    for k in best_k:
        if k > _isqrt(best_M):
            return {'M': 333312, 'k': [336, 372, 434]}
            
    return {'M': best_M, 'k': best_k}
# EVOLVE-BLOCK-END
