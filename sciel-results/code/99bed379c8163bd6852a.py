import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    LIMIT = 10**11
    # Generate powerful numbers up to LIMIT
    powerful = set()
    
    # Sieve for square-free up to 10^6
    SF_LIMIT = 10**6
    is_sq_free = np.ones(SF_LIMIT + 1, dtype=bool)
    is_sq_free[0] = False
    for i in range(2, int(np.sqrt(SF_LIMIT)) + 1):
        if is_sq_free[i]:
            is_sq_free[i*i::i*i] = False
            
    sf_list = [i for i in range(1, SF_LIMIT + 1) if is_sq_free[i]]
    
    # Primes for large v check
    def gen_primes(n):
        sieve = np.ones(n + 1, dtype=bool)
        sieve[0:2] = False
        for i in range(2, int(np.sqrt(n)) + 1):
            if sieve[i]:
                sieve[i*i::i] = False
        return [i for i, prime in enumerate(sieve) if prime]
        
    primes = gen_primes(10**5)
    
    def is_square_free_large(x):
        if x <= SF_LIMIT:
            return bool(is_sq_free[x])
        for p in primes:
            if p * p > x:
                break
            if x % (p * p) == 0:
                return False
        return True

    # Generate powerful numbers
    max_u = int(np.sqrt(LIMIT))
    for u in range(1, max_u + 1):
        u2 = u * u
        max_v = LIMIT // u2
        if max_v <= SF_LIMIT:
            # Use precomputed list
            # Find index
            idx = np.searchsorted(sf_list, max_v)
            for v in sf_list[:idx]:
                powerful.add(u2 * v)
        else:
            # Iterate v up to max_v
            # This is slow for small u.
            # But u is small, so max_v is large.
            # We can just iterate v?
            # For u=1, max_v=10^11. Too slow.
            # We must limit this.
            # Let's only generate v up to 10^7 for small u?
            # Or just skip large v for u=1?
            # If we skip, we miss powerful numbers like 10^11.
            # But we need sums.
            # Let's just iterate v up to 10^6 for all u, and for u < 100, iterate v up to 10^7.
            # This is a heuristic to keep time low.
            pass
            
    # Refine generation:
    powerful.clear()
    for u in range(1, max_u + 1):
        u2 = u * u
        max_v = LIMIT // u2
        if u < 100:
            # Check v up to 10^7
            v_limit = min(max_v, 10**7)
            for v in range(1, v_limit + 1):
                if is_square_free_large(v):
                    powerful.add(u2 * v)
        else:
            # Check v up to SF_LIMIT
            idx = np.searchsorted(sf_list, max_v)
            for v in sf_list[:idx]:
                powerful.add(u2 * v)
                
        if time.time() > deadline - 20:
            break

    P = np.array(sorted(powerful), dtype=np.int64)
    
    # Candidate N
    # Use multiples of primorial
    # Primorial of first 10 primes
    M = 2*3*5*7*11*13*17*19*23*29
    candidates = []
    for m in range(1, 1000):
        N = m * M
        if N > 2 * LIMIT:
            break
        candidates.append(N)
        
    # Also add some random multiples
    import random
    random.seed(42)
    for _ in range(100):
        N = random.randint(1, 2 * LIMIT)
        candidates.append(N)
        
    best_N = 0
    best_count = 0
    best_pairs = []
    
    for N in candidates:
        if N > 2 * LIMIT:
            continue
        # Count pairs
        # a <= b => a <= N/2
        mask = P <= N // 2
        A = P[mask]
        if len(A) == 0:
            continue
        B = N - A
        # Check if B is in P
        # Use searchsorted
        idx = np.searchsorted(P, B)
        # Ensure idx is within bounds
        valid_idx = idx < len(P)
        # Check equality
        # Handle bounds
        B_check = B[valid_idx]
        P_check = P[idx[valid_idx]]
        matches = B_check == P_check
        count = np.sum(matches)
        
        if count > best_count:
            best_count = count
            best_N = N
            # Retrieve pairs
            # A[matches] and B[matches]
            # Note: matches is boolean array of length len(A)
            # But we filtered idx. Need to be careful.
            # Let's re-derive pairs for the best one only to save time?
            # Or just store the indices.
            
    # Re-run for best_N to get pairs
    if best_count > 0:
        N = best_N
        mask = P <= N // 2
        A = P[mask]
        B = N - A
        idx = np.searchsorted(P, B)
        valid_idx = idx < len(P)
        B_check = B[valid_idx]
        P_check = P[idx[valid_idx]]
        matches = B_check == P_check
        
        pairs = []
        # A is array, matches is boolean array of same length
        # But matches corresponds to A[valid_idx]? No.
        # A has length L. idx has length L.
        # valid_idx is boolean of length L.
        # So A[valid_idx] are the candidates.
        # B[valid_idx] are the complements.
        # matches is boolean of length sum(valid_idx).
        # We need to map back to A.
        
        # Simpler:
        # Iterate A
        for i in range(len(A)):
            if not valid_idx[i]:
                continue
            if matches[valid_idx[:i+1].sum() - 1]: # This indexing is tricky
                pass
                
        # Let's just do a linear scan for the best N to get pairs, it's fast enough for one N.
        P_set = set(P.tolist())
        pairs = []
        for a in A:
            b = N - a
            if b in P_set:
                pairs.append([a, b])
                
    return {'N': best_N, 'pairs': pairs}
