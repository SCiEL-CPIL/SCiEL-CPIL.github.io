import time

def _has_perfect_matching(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1,...,b_n with i | b_i."""
    # Build adjacency: for each i in 1..n, which positions j (1..L) have (m+j) % i == 0
    adj = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        # Find all j in 1..L where (m+j) % i == 0
        # (m+j) % i == 0 means j % i == (-m) % i
        target = (-m) % i
        if target == 0:
            first = i  # smallest j >= 1 with j % i == 0
        else:
            first = target
        if first > L:
            return False
        # Set bits for first, first+i, first+2i, ... <= L
        j = first
        while j <= L:
            mask |= (1 << (j - 1))
            j += i
        adj[i] = mask
    
    # Hopcroft-Karp style or just DFS matching with bit operations
    match_r = [0] * L  # match_r[j_idx] = which i is matched to position j_idx (0-indexed)
    
    # Order i by increasing number of candidates (most constrained first)
    order = list(range(1, n + 1))
    order.sort(key=lambda i: bin(adj[i]).count('1'))
    
    def dfs(i, visited):
        candidates = adj[i] & ~visited
        while candidates:
            low_bit = candidates & -candidates
            j_idx = low_bit.bit_length() - 1
            candidates ^= low_bit
            if match_r[j_idx] == 0 or dfs(match_r[j_idx], visited | low_bit):
                match_r[j_idx] = i
                return True
        return False
    
    for i in order:
        if not dfs(i, 0):
            return False
    return True

def _compute_f(m, n=12, max_L=144):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    lo, hi = 1, max_L
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_perfect_matching(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    n = 12
    max_m = 1000000
    P = 27720  # LCM(1..12)
    
    best_m = 0
    best_f = 0
    
    # f(12, m) depends only on m mod P
    # Precompute f for all residues, find the best one
    # To save time, we can skip residues that clearly can't beat current best
    
    # First pass: quick upper bound check
    # For a given residue r, the minimum possible L is at least max over i of (smallest j with (r+j)%i==0)
    # But that's a lower bound on L, not directly helpful for upper bounding f
    
    # Let's just do the full scan with optimized matching
    # Each _compute_f does ~8 matching calls (binary search over 1..144)
    # Each matching call is very fast with bit operations
    
    for r in range(P):
        if time.time() > deadline:
            break
        
        f_val = _compute_f(r, n)
        if f_val > best_f:
            best_f = f_val
            best_m = r
    
    # best_m is in [0, P-1] which is within [0, max_m]
    best_m = int(best_m)
    if best_m < 0 or best_m > max_m:
        best_m = 0
    
    return best_m
