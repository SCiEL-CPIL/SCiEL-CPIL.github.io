import time
import numpy as np

def construct():
    start_time = time.time()
    
    # Heuristic: Start with a large K_max
    K_max = 50_000_000
    
    try:
        # 1. Sieve for sigma
        sigma = np.zeros(K_max + 1, dtype=np.int64)
        
        # This loop is the main CPU cost
        for i in range(1, K_max + 1):
            sigma[i::i] += i
            
        # Time check
        elapsed = time.time() - start_time
        if elapsed > 100:
            # Fallback
            return {'M': 333312, 'k': [336, 372, 434]}

        # 2. Compute M = k * sigma(k)
        # We only care about k where M <= 10^18
        # But with K_max=5e7, M is ~1e15, so all pass.
        # To be safe, we can mask, but it adds overhead.
        # Let's just compute all.
        k_vals = np.arange(1, K_max + 1, dtype=np.int64)
        M_vals = k_vals * sigma[1:K_max + 1]
        
        # Filter M <= 10^18
        mask = M_vals <= 10**18
        if not np.all(mask):
            M_vals = M_vals[mask]
            k_vals = k_vals[mask]
            
        # 3. Find most frequent M
        # np.unique sorts, which is O(N log N)
        unique_M, inverse = np.unique(M_vals, return_inverse=True)
        
        # Count occurrences
        counts = np.bincount(inverse)
        
        # Find max count
        max_count = np.max(counts)
        
        # If max count is 1, we have a valid solution (any k)
        # But we want to maximize score.
        
        # Find the M with max count
        # There might be multiple M with the same max count.
        # We just pick one.
        best_idx = np.argmax(counts)
        best_M = unique_M[best_idx]
        
        # Get the list of k for this M
        # inverse == best_idx gives the indices in M_vals
        k_indices = np.where(inverse == best_idx)[0]
        best_k_list = k_vals[k_indices]
        
        # Convert to Python types
        return {'M': int(best_M), 'k': [int(k) for k in best_k_list]}
        
    except Exception:
        # Fallback in case of memory error or timeout
        return {'M': 333312, 'k': [336, 372, 434]}
