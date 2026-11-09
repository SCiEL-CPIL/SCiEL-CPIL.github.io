import math
from collections import defaultdict

def construct():
    import time
    start_time = time.time()
    DEADLINE = 115.0  # seconds, leaving 5s safety margin

    best_result = {'M': 333312, 'k': [336, 372, 434]}
    best_count = 3

    try:
        import numpy as np
        use_numpy = True
    except ImportError:
        use_numpy = False

    if use_numpy:
        # Start with a larger max_k; we'll verify timing and adjust
        max_k = 20000000  # 20 million
        sigma = np.zeros(max_k + 1, dtype=np.int64)
        for i in range(1, max_k + 1):
            sigma[i::i] += i
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            return best_result

        k_arr = np.arange(1, max_k + 1, dtype=np.int64)
        M_arr = k_arr * sigma[1:max_k + 1]
        valid = (M_arr <= 10**18) & (k_arr <= sigma[1:max_k + 1])
        
        # Filter to valid entries
        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            return best_result
        
        M_valid = M_arr[valid_idx]
        k_valid = k_arr[valid_idx]
        
        # Use np.unique to group by M value and find the M with most k's
        unique_M, inverse, counts = np.unique(M_valid, return_inverse=True, return_counts=True)
        
        # Find the M with the maximum count
        max_count = int(counts.max())
        if max_count > best_count:
            best_idx = int(counts.argmax())
            best_M = int(unique_M[best_idx])
            # Get all k values for this M
            mask = inverse == best_idx
            ks = k_valid[mask].tolist()
            best_result = {'M': best_M, 'k': [int(x) for x in ks[:1000]]}
            best_count = max_count

        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            return best_result

        # Also check top few M values in case there are ties or near-misses
        top_n = min(50, len(unique_M))
        top_indices = np.argsort(counts)[::-1][:top_n]
        for idx in top_indices:
            cnt = int(counts[idx])
            if cnt > best_count:
                best_M = int(unique_M[idx])
                mask = inverse == idx
                ks = k_valid[mask].tolist()
                best_result = {'M': best_M, 'k': [int(x) for x in ks[:1000]]}
                best_count = cnt

    else:
        max_k = 1000000
        sigma = [0] * (max_k + 1)
        for i in range(1, max_k + 1):
            for j in range(i, max_k + 1, i):
                sigma[j] += i
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            return best_result

        M_to_ks = defaultdict(list)
        for k in range(1, max_k + 1):
            M = k * sigma[k]
            if M <= 10**18 and k <= sigma[k]:
                M_to_ks[M].append(k)
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            return best_result

        for M, ks in M_to_ks.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_result = {'M': int(M), 'k': [int(x) for x in ks[:1000]]}

    return best_result
