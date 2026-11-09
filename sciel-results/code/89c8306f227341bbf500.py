import numpy as np
import time

def generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    powerful_set = set()
    max_b = int(limit ** (1/3)) + 2
    for b in range(1, max_b + 1):
        b_cubed = b ** 3
        if b_cubed > limit:
            break
        max_a = int((limit // b_cubed) ** 0.5) + 1
        for a in range(1, max_a + 1):
            val = a * a * b_cubed
            if val <= limit:
                powerful_set.add(val)
            else:
                break
    return sorted(list(powerful_set))

def construct():
    start_time = time.time()
    
    # Determine limit based on time
    # We will try 10^8
    limit = 10**8
    P = generate_powerful(limit)
    P_np = np.array(P, dtype=np.int64)
    
    elapsed = time.time() - start_time
    if elapsed > 100:
        # Should not happen for generation
        return {'N': 2, 'pairs': [[1, 1]]}

    # Compute sums
    # If P is small, use full matrix
    if len(P_np) < 5000:
        sums = np.add.outer(P_np, P_np)
        max_sum_val = 2 * limit
        counts = np.bincount(sums.ravel(), minlength=max_sum_val + 1)
        best_sum = np.argmax(counts)
    else:
        # Fallback to Python loop if too large (unlikely for 10^8)
        counts_dict = {}
        P_list = list(P_np)
        for i, a in enumerate(P_list):
            for b in P_list[i:]:
                s = a + b
                counts_dict[s] = counts_dict.get(s, 0) + 1
        best_sum = max(counts_dict, key=counts_dict.get)

    # Find exact pairs for best_sum and neighbors
    candidates = [best_sum]
    for d in range(1, 10):
        if best_sum - d >= 0:
            candidates.append(best_sum - d)
        if best_sum + d <= 2 * limit:
            candidates.append(best_sum + d)
            
    P_set = set(P_np)
    best_score = 0
    best_N = best_sum
    best_pairs = []
    
    for N in candidates:
        pairs = []
        for a in P_np:
            if a > N:
                break
            b = N - a
            if b < a:
                continue
            if b > limit:
                continue
            if b in P_set:
                pairs.append([int(a), int(b)])
        
        if len(pairs) > best_score:
            best_score = len(pairs)
            best_N = N
            best_pairs = pairs
            
    return {'N': int(best_N), 'pairs': best_pairs}
