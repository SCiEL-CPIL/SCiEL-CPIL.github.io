import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 120.0 # Safety margin
    
    # 1. Generate powerful numbers up to L
    # L should be large enough to find many pairs.
    # Based on knowledge, 500M was good. Let's try 1e9 if time allows.
    # 1e9 -> ~60k numbers.
    L = 10**9
    
    powerful_set = set()
    # b must be square-free
    # Max b: b^3 <= L => b <= L^(1/3) ~ 1000
    max_b = int(L ** (1/3)) + 2
    
    # Precompute square-free numbers? Or just check.
    # For each b, we need b to be square-free.
    # Simple check: no prime squared divides b.
    # Or just generate and check.
    
    powerful_nums = []
    for b in range(1, max_b + 1):
        # Check if b is square-free
        is_sf = True
        temp_b = b
        # Small primes check
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            if (p * p) > temp_b:
                break
            if (temp_b % (p * p)) == 0:
                is_sf = False
                break
            while (temp_b % p) == 0:
                temp_b //= p
        if temp_b > 1:
            # Check if temp_b is a square
            r = int(temp_b ** 0.5)
            if r * r == temp_b:
                is_sf = False
        
        if not is_sf:
            continue
            
        b3 = b ** 3
        # a^2 <= L / b3
        max_a = int((L // b3) ** 0.5)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerful_set.add(n)
            else:
                break
                
    P = np.array(sorted(powerful_set), dtype=np.int64)
    K = len(P)
    
    # 2. Search for best N
    # We will sample N in [L, 2*L]
    # But 2*L = 2e9.
    # The number of pairs for N is the count of a in P such that N-a is in P.
    
    best_N = 0
    best_count = 0
    best_pairs = []
    
    # Sample candidates
    # Use random sampling to cover the range
    np.random.seed(42)
    num_samples = 5000
    
    # We can also try specific structured candidates
    # But random is safer.
    
    candidates = np.random.randint(L, 2 * L + 1, num_samples)
    
    for i, N in enumerate(candidates):
        if time.time() > deadline - 10:
            break
            
        # Count pairs
        # a <= N
        mask = P <= N
        P_sub = P[mask]
        if len(P_sub) == 0:
            continue
            
        # Check if N - P_sub is in powerful_set
        # Using np.isin is fast
        # Note: N - P_sub might have negative values if P_sub > N, but mask ensures P <= N
        # So N - P_sub >= 0
        diffs = N - P_sub
        # Check membership
        # np.isin with a set is not directly supported efficiently for large arrays?
        # np.isin uses a hash table internally, which is good.
        in_set = np.isin(diffs, P)
        count = np.sum(in_set)
        
        if count > best_count:
            best_count = count
            best_N = N
            # Store indices or values for pair extraction later
            # We need to extract pairs now or store mask
            # Storing mask is memory heavy if we do it for all.
            # We can just re-extract for the best one at the end.
            # But we need to ensure we don't re-scan if we find a better one.
            # We'll just store N and count.
            
    # Refine around best_N
    if best_N > 0:
        # Check neighbors
        for delta in range(-10, 11):
            N_test = best_N + delta
            if N_test < 2 or N_test > 2 * L:
                continue
            if time.time() > deadline - 5:
                break
            mask = P <= N_test
            P_sub = P[mask]
            if len(P_sub) == 0:
                continue
            diffs = N_test - P_sub
            in_set = np.isin(diffs, P)
            count = np.sum(in_set)
            if count > best_count:
                best_count = count
                best_N = N_test

    # Extract pairs for best_N
    if best_N > 0:
        mask = P <= best_N
        P_sub = P[mask]
        diffs = best_N - P_sub
        in_set = np.isin(diffs, P)
        # Get the actual pairs
        # We need a <= b
        # a is in P_sub[in_set], b is in diffs[in_set]
        a_vals = P_sub[in_set]
        b_vals = diffs[in_set]
        
        # Filter a <= b
        # Since a+b=N, a<=b implies a <= N/2
        # So we can just filter a_vals <= N//2
        valid_mask = a_vals <= (best_N // 2)
        a_final = a_vals[valid_mask]
        b_final = b_vals[valid_mask]
        
        # Sort pairs by a
        sort_idx = np.argsort(a_final)
        a_final = a_final[sort_idx]
        b_final = b_final[sort_idx]
        
        pairs = [[int(a), int(b)] for a, b in zip(a_final, b_final)]
    else:
        # Fallback: small N
        N = 2
        pairs = [[1, 1]]
        best_N = 2

    return {'N': best_N, 'pairs': pairs}
