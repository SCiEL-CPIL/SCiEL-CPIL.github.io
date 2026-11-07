import sys
sys.setrecursionlimit(10000)

def _precompute():
    n = 12
    period = 27720
    max_L = 144  # upper bound on f(12, m)
    
    results = [0] * period
    
    # For each residue m mod period, find minimal L
    # We use a sliding window approach: for each m, we incrementally extend L
    # and check if a perfect matching exists using Hall's theorem conditions
    # or direct backtracking.
    
    # Key insight: for a given m and L, we need to match each i in 1..n
    # to a distinct d in 1..L such that i | (m+d).
    # d ≡ -m (mod i)
    
    # For efficiency, we precompute for each i the list of d values
    # in 1..max_L that satisfy d ≡ -m (mod i) for a given m.
    
    # Actually, let's use a different approach:
    # For each m, we can determine the "next" valid d for each i as we extend L.
    # We'll use a priority-queue-like approach or just iterate L from 1 to max_L.
    
    # Since n=12 is small, let's use backtracking with pruning.
    # For each m, we'll check L=1,2,... until we find a valid matching.
    
    # To speed up, we can use the following observation:
    # The set of valid d values for i in window (m, m+L] is:
    # d in {1,...,L} such that (m+d) % i == 0
    # This is equivalent to d ≡ (-m) % i (mod i)
    
    # Let's precompute for each i and each possible residue r = (-m) % i,
    # the sorted list of d values in 1..max_L that are ≡ r (mod i).
    
    # valid_ds[i][r] = sorted list of d in 1..max_L with d ≡ r (mod i)
    valid_ds = [None] * (n + 1)
    for i in range(1, n + 1):
        valid_ds[i] = [[] for _ in range(i)]
        for d in range(1, max_L + 1):
            r = d % i
            valid_ds[i][r].append(d)
    
    for m in range(period):
        found = False
        for L in range(1, max_L + 1):
            # Build adjacency: for each i, the valid d values in 1..L
            adj = [[] for _ in range(n)]
            for i in range(1, n + 1):
                r = (-m) % i
                # valid_ds[i][r] contains all d in 1..max_L with d ≡ r (mod i)
                # We need those <= L
                vd = valid_ds[i][r]
                # Binary search for the last element <= L
                # Since vd is sorted, we can use bisect
                # But for small lists, linear scan is fine
                # Actually, let's just filter
                adj[i-1] = [d for d in vd if d <= L]
            
            # Quick check: if any adj[i] is empty, no matching
            if any(len(a) == 0 for a in adj):
                continue
            
            # Use backtracking with most-constrained-first heuristic
            # Order by number of options
            order = sorted(range(n), key=lambda idx: len(adj[idx]))
            
            # Hall's condition quick check for the most constrained subset
            # For the k most constrained nodes, they need at least k distinct d values
            # Let's check the union of their adjacencies
            for k in range(1, n + 1):
                subset = order[:k]
                union = set()
                for idx in subset:
                    union.update(adj[idx])
                if len(union) < k:
                    break  # Hall's condition violated, no matching
            else:
                # Hall's condition satisfied for all subsets (necessary but not sufficient)
                # Do backtracking
                assigned = [False] * (L + 1)
                
                def backtrack(pos):
                    if pos == n:
                        return True
                    i_idx = order[pos]
                    for d in adj[i_idx]:
                        if not assigned[d]:
                            assigned[d] = True
                            if backtrack(pos + 1):
                                return True
                            assigned[d] = False
                    return False
                
                if backtrack(0):
                    results[m] = L
                    found = True
                    break
        
        if not found:
            results[m] = max_L
    
    return results

# Precompute at module load time
_results = _precompute()

def construct():
    period = 27720
    best_r = 0
    best_val = -1
    for r in range(period):
        if _results[r] > best_val:
            best_val = _results[r]
            best_r = r
    
    return best_r
