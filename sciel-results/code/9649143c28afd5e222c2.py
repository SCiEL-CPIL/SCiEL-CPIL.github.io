import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s buffer for cleanup/return
    
    # Fallback result
    # M = 333312 has solutions 336, 372, 434 (3 solutions)
    # But we know there are better ones.
    # The current best found by the seed was 5 solutions.
    # Let's keep a known good fallback just in case.
    # A known M with 5 solutions:
    # From OEIS or similar, M= 333312 is small.
    # Let's use the result from the previous run if we can't find better?
    # The prompt says "Current program (score 1.6667)" which implies it found 5 solutions.
    # I will try to find more.
    
    fallback_M = 333312
    fallback_k = [336, 372, 434]
    
    # We will try to scan up to N
    # N=10^8 is 800MB for sigma array. 
    # M_vals will be another 800MB.
    # Total 1.6GB. This is risky.
    # Let's try N=5*10^7 (400MB each, 800MB total).
    
    best_result = {'M': fallback_M, 'k': fallback_k}
    best_count = len(fallback_k)
    
    # List of N to try, starting from large to small if memory fails
    Ns = [50_000_000, 20_000_000, 10_000_000]
    
    for N in Ns:
        if time.time() > deadline - 20:
            break
            
        try:
            # 1. Compute sigma(1..N)
            sigma = np.zeros(N + 1, dtype=np.int64)
            
            # Sieve of Eratosthenes style for sum of divisors
            # sigma[i] += i for all multiples of i
            for i in range(1, N + 1):
                sigma[i::i] += i
                
                # Periodic time check
                if i % 5_000_000 == 0:
                    if time.time() > deadline - 10:
                        raise TimeoutError("Sieve too slow")
            
            # 2. Compute M values
            k_vals = np.arange(1, N + 1, dtype=np.int64)
            M_vals = k_vals * sigma[1:]
            
            # 3. Find most frequent M
            # np.unique is O(N log N)
            unique_M, inverse, counts = np.unique(M_vals, return_inverse=True, return_counts=True)
            
            max_count = np.max(counts)
            
            if max_count > best_count:
                best_indices = np.where(counts == max_count)[0]
                # Pick the first one with max count
                # Ideally pick the one with smallest M? Or any.
                # The score is just the count.
                best_idx = best_indices[0]
                best_M = int(unique_M[best_idx])
                
                # Get k values
                k_indices = np.where(inverse == best_idx)[0]
                k_list = [int(k) for k in k_vals[k_indices]]
                
                # Update best
                best_result = {'M': best_M, 'k': k_list}
                best_count = max_count
                
                # If we found a lot, we can stop?
                # Maybe there are more with larger N?
                # But larger N is slower and memory heavy.
                # If we found > 10, it's a very good result.
                # Let's continue to next N if time permits and N is smaller? No, we go large to small.
                # If we found a good result with N=50M, we might want to try N=100M if we had it.
                # But we don't.
                # So we break?
                # Actually, a larger N might reveal a M with even more solutions.
                # But we started with the largest N we planned.
                # So if we succeeded with N=50M, we can break.
                break
                
        except MemoryError:
            print(f"MemoryError for N={N}, trying smaller N")
            continue
        except TimeoutError:
            print(f"Timeout for N={N}, trying smaller N")
            continue
        except Exception as e:
            print(f"Error for N={N}: {e}")
            continue
            
    return best_result
