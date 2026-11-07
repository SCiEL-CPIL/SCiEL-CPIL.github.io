import time
import numpy as np
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin out of 120s

    # Seed solution: M = 333312, k in {336, 372, 434}
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3

    # We want to find M <= 10^18 with many distinct k_i (k_i <= sqrt(M))
    # such that k_i * sigma(k_i) = M.
    
    # Strategy: Compute sigma(k) for k up to K using a sieve.
    # Then compute M_vals = k * sigma(k) and find collisions.
    
    # Try increasing K values, starting high to maximize chance of finding more solutions.
    # If memory or time fails, fall back to smaller K.
    
    K_candidates = [150_000_000, 100_000_000, 50_000_000, 20_000_000]
    
    for K in K_candidates:
        if time.time() > deadline - 20:
            break
            
        try:
            # Compute sigma(1..K) using sieve
            # sigma(k) for k=1.5e8 is approx 1.5e8 * log(log(1.5e8)) ~ 1.5e8 * 3.4 ~ 5e8
            # This fits in int32 (max 2.1e9)
            sigma = np.zeros(K + 1, dtype=np.int32)
            
            # Sieve loop
            # This is the most time-consuming part.
            for j in range(1, K + 1):
                if time.time() > deadline - 15:
                    raise TimeoutError()
                sigma[j::j] += j
                
            # Compute M = k * sigma(k) for k = 1..K
            k_vals = np.arange(1, K + 1, dtype=np.int64)
            # sigma[1:] is int32, k_vals is int64. Result is int64.
            M_vals = k_vals * sigma[1:].astype(np.int64)
            
            # Filter M <= 10^18
            MAX_M = 10**18
            valid_mask = M_vals <= MAX_M
            if not np.any(valid_mask):
                continue
                
            M_valid = M_vals[valid_mask]
            k_valid = k_vals[valid_mask]
            
            n_valid = len(M_valid)
            if n_valid <= best_count:
                continue
                
            # Sort by M to find collisions
            # We sort by M, then by k to ensure distinct k's are grouped
            sort_indices = np.lexsort((k_valid, M_valid))
            M_sorted = M_valid[sort_indices]
            k_sorted = k_valid[sort_indices]
            
            if n_valid > 1:
                diff_M = np.diff(M_sorted)
                change_indices = np.where(diff_M != 0)[0]
                
                # Build run lengths
                starts = np.concatenate(([0], change_indices + 1))
                ends = np.concatenate((change_indices, [n_valid - 1]))
                lengths = ends - starts + 1
                
                max_len = int(np.max(lengths))
                
                if max_len > best_count:
                    max_idx = int(np.argmax(lengths))
                    start_idx = int(starts[max_idx])
                    end_idx = int(ends[max_idx])
                    
                    best_M = int(M_sorted[start_idx])
                    best_k = [int(x) for x in k_sorted[start_idx:end_idx + 1]]
                    best_count = max_len
                    
                    # If we found a significantly better solution, we might stop early
                    # but let's check if we can find even more with a larger K?
                    # No, we are iterating K downwards or upwards?
                    # The list is descending. So if we succeed with 150M, we have the best chance.
                    # If 150M fails, we try 100M.
                    # So if we succeed, we can break?
                    # Actually, a larger K covers all smaller k's.
                    # So the best solution found with K=150M is the best possible from this set.
                    # We don't need to try smaller K's if we succeeded with a larger one.
                    break
                    
        except MemoryError:
            continue
        except TimeoutError:
            continue
        except Exception:
            break

    # Final validation: ensure all k_i satisfy k_i * sigma(k_i) = M
    # and k_i <= sqrt(M), and they are distinct.
    if best_count > 0:
        # Verify distinctness
        if len(set(best_k)) != len(best_k):
            # Should not happen, but clean up
            best_k = list(set(best_k))
            best_count = len(best_k)

        # Verify each k satisfies k*sigma(k) = M
        # We trust the numpy computation, but let's do a quick check on a few or all if small
        # If best_count is large, checking all is O(N * sqrt(N)) which might be slow.
        # But N is likely small (4-10).
        for k_val in best_k:
            # Compute sigma(k_val) exactly
            s = 0
            n = k_val
            for d in range(1, int(math.isqrt(n)) + 1):
                if n % d == 0:
                    s += d
                    other = n // d
                    if other != d:
                        s += other
            if k_val * s != best_M:
                # Remove invalid k
                best_k.remove(k_val)
                best_count -= 1
            if k_val > math.isqrt(best_M):
                best_k.remove(k_val)
                best_count -= 1

    # Ensure types are correct
    return {'M': int(best_M), 'k': list(best_k)}
