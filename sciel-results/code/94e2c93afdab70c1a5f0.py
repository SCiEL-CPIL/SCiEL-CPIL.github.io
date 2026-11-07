import time

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    n = 12
    max_L = 144
    period = 27720  # lcm(1, 2, ..., 12)
    
    # Precompute f(12, r) for all r in [0, period)
    # f(12, r) is the minimum L >= 1 such that a matching exists in (r, r+L]
    f_values = [0] * period
    
    # Helper to check if a matching exists for given m and L
    def has_matching(m, L):
        # Generate candidates for each i
        candidates = [None] * (n + 1)
        
        # We'll use a list of sets or lists
        # To speed up DFS, we sort indices by number of candidates
        indices = list(range(1, n + 1))
        
        for i in indices:
            # First multiple of i > m
            # ceil((m+1)/i) * i
            start = ((m + 1) + i - 1) // i * i
            if start <= m:
                start += i
            cands = []
            b = start
            limit = m + L
            while b <= limit:
                cands.append(b)
                b += i
            candidates[i] = cands
            if not cands:
                return False
        
        # Sort indices by number of candidates (most constrained first)
        indices.sort(key=lambda i: len(candidates[i]))
        
        # DFS with backtracking
        used = set()
        
        def dfs(idx):
            if idx == n:
                return True
            i = indices[idx]
            for b in candidates[i]:
                if b not in used:
                    used.add(b)
                    if dfs(idx + 1):
                        return True
                    used.remove(b)
            return False
        
        return dfs(0)
    
    # Precompute f_values
    for r in range(period):
        found = False
        for L in range(1, max_L + 1):
            if time.time() > deadline:
                # If we run out of time, assume worst case or break
                # But this should not happen as the computation is fast
                f_values[r] = max_L
                break
            if has_matching(r, L):
                f_values[r] = L
                found = True
                break
        if not found:
            f_values[r] = max_L
            
    # Find the m in [0, 1000000] that maximizes f(12, m)
    # f(12, m) = f_values[m % period]
    max_f = 0
    best_m = 0
    
    # We can optimize this scan by precomputing the max value for each residue
    # and then just checking which residues appear in the range
    # But 10^6 is small enough to just scan
    
    # To be safe and fast, let's just iterate
    # We want the first m (or any m) with the max value
    # The problem asks to "Return one int m". It doesn't specify which one if multiple exist.
    # "Maximize this independently computed L."
    # So we just need any m that gives the global max.
    
    # Let's find the global max first
    global_max = max(f_values)
    
    # Then find the first m in [0, 1000000] that achieves it
    for m in range(1000001):
        if f_values[m % period] == global_max:
            best_m = m
            break
            
    return int(best_m)

# If the precomputation is too slow, we might need to optimize has_matching further.
# But for N=12 and L<=144, the candidate lists are very short.
# Max candidates for i=1 is 144. For i=12 is 12.
# The branching factor is small.
