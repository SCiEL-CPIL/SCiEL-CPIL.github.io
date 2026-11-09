import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130

    def is_square_free(n):
        if n < 2:
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if n % (p * p) == 0:
                return False
        return True

    def generate_powerful(limit):
        powerful = set()
        max_v = int(limit ** (1.0/3.0)) + 2
        sq_free_v = []
        for v in range(1, max_v + 1):
            if is_square_free(v):
                sq_free_v.append(v)
        for v in sq_free_v:
            v3 = v ** 3
            if v3 > limit:
                break
            max_u = int(math.isqrt(limit // v3))
            for u in range(1, max_u + 1):
                x = u * u * v3
                if x <= limit:
                    powerful.add(x)
        return sorted(powerful)

    # Generate powerful numbers
    M = 500_000_000
    P = generate_powerful(M)
    K = len(P)
    P_set = set(P)
    P_np = np.array(P, dtype=np.int64)

    if time.time() > deadline - 30:
        M = 100_000_000
        P = generate_powerful(M)
        K = len(P)
        P_set = set(P)
        P_np = np.array(P, dtype=np.int64)

    # Structural approach: 
    # For each a in P, the number of b in P with b >= a and a+b <= 2M is K - index_of_a
    # But we want N = a+b to have many representations.
    # 
    # Key: N has many reps if many pairs (a,b) sum to it.
    # Use numpy to efficiently count sums.
    
    # Use a histogram approach with numpy
    # The max possible sum is 2*M
    # Use int32 for counts to save memory
    max_sum = 2 * M
    if max_sum > 2_000_000_000:  # too big for int32 array
        max_sum = 2_000_000_000
    
    counts = np.zeros(max_sum + 1, dtype=np.int32)
    
    # Process in chunks to manage memory and allow early termination
    chunk_size = 500
    for i in range(0, K, chunk_size):
        i_end = min(i + chunk_size, K)
        a_chunk = P_np[i:i_end]  # (chunk_size,)
        
        # For each a in chunk, we want to add 1 to counts[a + b] for all b >= a
        # To enforce a <= b, we only consider b from index i onwards
        # But this is complex. Instead, count all pairs and divide by 2 later.
        
        # Compute all sums a_chunk[:, None] + P_np[None, :]
        # This is (chunk_size, K) array
        sums = a_chunk[:, None] + P_np[None, :]  # (chunk_size, K)
        
        # Flatten and use bincount
        flat_sums = sums.ravel()
        # Filter valid sums
        valid = (flat_sums >= 0) & (flat_sums <= max_sum)
        flat_sums = flat_sums[valid]
        
        if len(flat_sums) > 0:
            chunk_counts = np.bincount(flat_sums, minlength=max_sum + 1)
            counts += chunk_counts.astype(np.int32)
        
        if time.time() > deadline - 20:
            break
    
    # Find the sum with maximum count
    # But we counted ordered pairs (a,b) for all a,b in P
    # We want unordered pairs with a <= b
    # For a != b, each pair is counted twice (once as (a,b) and once as (b,a))
    # For a == b, it's counted once
    # So num_unordered = (count + [N even and N/2 in P]) / 2
    
    # Find top candidates
    # Get the index of max count
    best_idx = np.argmax(counts)
    best_N = int(best_idx)
    best_ordered_count = int(counts[best_N])
    
    # Check if N/2 is in P (for the a==b case)
    if best_N % 2 == 0:
        half = best_N // 2
        if half in P_set:
            num_pairs = (best_ordered_count + 1) // 2
        else:
            num_pairs = best_ordered_count // 2
    else:
        num_pairs = best_ordered_count // 2
    
    # Extract pairs
    pairs = []
    for a in P:
        if a > best_N // 2:
            break
        b = best_N - a
        if b >= a and b in P_set:
            pairs.append([a, b])
    
    if not pairs:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N, 'pairs': pairs}
