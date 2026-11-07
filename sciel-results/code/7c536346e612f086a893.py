import time

def construct():
    start_time = time.time()
    limit = 85.0  # Safety margin before 90s limit
    
    N = 12
    period = 27720  # lcm(1..12)
    
    best_m = 0
    best_f = 0
    
    # Precompute divisors or just use modulo
    # We need to find distinct d_1, ..., d_12 in 1..L such that (m+d_i) % i == 0
    
    def can_match(m, L):
        # Generate candidate offsets for each i in 1..12
        # cands[i] is a list of d in 1..L such that (m+d) % (i+1) == 0
        # Note: i in 0..11 corresponds to divisor 1..12
        cands = [[] for _ in range(N)]
        
        for i in range(N):
            div = i + 1
            # We need d such that (m + d) % div == 0
            # d % div == (-m) % div
            rem = (-m) % div
            if rem == 0:
                first_d = div
            else:
                first_d = rem
                
            # Add all d = first_d, first_d + div, ... <= L
            for d in range(first_d, L + 1, div):
                cands[i].append(d)
                
            # If no candidates, impossible
            if not cands[i]:
                return False
        
        # Sort indices by number of candidates (MRV heuristic)
        # This significantly reduces backtracking depth
        indices = list(range(N))
        indices.sort(key=lambda idx: len(cands[idx]))
        
        # used[d] indicates if offset d is already taken
        # Max L is small (<= 144), so we can use a list of bools or a set
        # A set is fast for membership, but list is faster for access if indexed.
        # However, d values are sparse in 1..L.
        # Let's use a set for simplicity and speed with sparse data, 
        # or a boolean array of size L+1.
        used = [False] * (L + 1)
        
        def backtrack(k):
            if k == N:
                return True
            i = indices[k]
            for d in cands[i]:
                if not used[d]:
                    used[d] = True
                    if backtrack(k + 1):
                        return True
                    used[d] = False
            return False
        
        return backtrack(0)
    
    # Search m in one period
    # We check time periodically to avoid timeout
    # 27720 iterations. If each takes ~1ms, total ~27s.
    # If each takes ~10ms, total ~277s (too slow).
    # We need can_match to be very fast.
    
    # Optimization: If we find a new best_f, we can potentially skip some L checks?
    # No, we need exact f(m) to compare.
    
    # To speed up, we can start L from 1.
    # For the "worst" m, f(m) is large.
    
    for m in range(period):
        # Check time
        if m % 500 == 0:
            if time.time() - start_time > limit:
                break
        
        # Find minimal L for this m
        # We know f(m) <= 144
        # Start from L=1
        found = False
        for L in range(1, 145):
            if can_match(m, L):
                if L > best_f:
                    best_f = L
                    best_m = m
                found = True
                break
        # If not found, L must be > 144, but problem states max is 144.
        # If loop finishes without break, something is wrong, but contract says max 144.
        
    return best_m
