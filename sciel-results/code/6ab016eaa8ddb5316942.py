import time

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s margin before 90s limit
    
    N = 12
    LCM = 27720
    
    # Pre-compute indices sorted by "tightness" (fewer candidates first)
    # For a given L, count of candidates for i is approx L/i.
    # So 12 has fewest, 1 has most.
    # This order is static for the DFS pruning logic, as L/i is monotonic with i.
    indices = list(range(N, 0, -1))
    
    def can_match(m, L):
        # Check if there is a matching in (m, m+L]
        # Candidates for i are k in 1..L such that (m+k) % i == 0
        
        # 1. Pre-compute candidates for each i
        cands = [None] * (N + 1)
        for i in range(1, N + 1):
            cands[i] = []
            # First multiple of i > m
            # k_min such that m + k_min >= i * (floor(m/i) + 1)
            # Actually, just iterate k? L is small.
            # Or compute start_k.
            # m % i = r. We need m+k ≡ 0 mod i => k ≡ -r mod i.
            # Smallest k > 0 is i - (m % i) if m % i != 0 else i.
            # Let's just loop k from 1 to L. It's fast enough for L < 200.
            # But we can be smarter.
            rem = m % i
            if rem == 0:
                start_k = i
            else:
                start_k = i - rem
            for k in range(start_k, L + 1, i):
                cands[i].append(k)
            
            if not cands[i]:
                return False

        # 2. Sort indices by number of candidates (fewest first)
        # We already have indices sorted by i descending, which correlates with fewer candidates.
        # But exact count depends on m and L.
        # Let's sort dynamically based on len(cands[i])
        sorted_indices = sorted(range(1, N + 1), key=lambda x: len(cands[x]))
        
        # 3. DFS
        used = [False] * (L + 1) # used[k] is True if k is taken
        
        def dfs(idx_pos):
            if idx_pos == N:
                return True
            
            i = sorted_indices[idx_pos]
            
            for k in cands[i]:
                if not used[k]:
                    used[k] = True
                    if dfs(idx_pos + 1):
                        return True
                    used[k] = False
                    
            return False
            
        return dfs(0)

    best_m = 0
    best_L = 0
    
    # Iterate through one period
    for m in range(LCM):
        if time.time() > deadline:
            break
            
        # Try to see if f(12, m) > best_L
        # Check L from 1 to best_L
        is_better = True
        for L in range(1, best_L + 1):
            if can_match(m, L):
                is_better = False
                break
        
        if is_better:
            # f(12, m) > best_L
            # Find exact f(12, m)
            L = best_L + 1
            while not can_match(m, L):
                L += 1
                if L > 500: # Safety break, should not happen
                    break
            best_L = L
            best_m = m
            
    return int(best_m)
