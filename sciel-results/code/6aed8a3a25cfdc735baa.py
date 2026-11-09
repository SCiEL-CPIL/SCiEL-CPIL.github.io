import time

# EVOLVE-BLOCK-START
P = 27720  # LCM(1..12)
MAX_L = 144
MIN_L = 12

def _has_matching(m, L):
    """
    Check if interval (m, m+L] contains distinct b_1..b_12 with i | b_i.
    Uses direct candidate generation and DFS augmenting path matching.
    """
    if L < MIN_L:
        return False
    
    # For each divisor i (1..12), generate list of valid positions j in [0, L)
    # such that (m + 1 + j) is divisible by i.
    # (m + 1 + j) % i == 0  =>  j % i == (-(m + 1)) % i
    adj = [[] for _ in range(12)]
    base = m + 1
    
    for i in range(1, 13):
        # Calculate the starting offset j0
        # j must satisfy j ≡ -base (mod i)
        r0 = (-base) % i
        # Add all j = r0, r0 + i, r0 + 2i, ... < L
        for j in range(r0, L, i):
            adj[i - 1].append(j)
            
    # Bipartite matching: Left side = 12 divisors (indices 0..11), Right side = L positions (indices 0..L-1)
    # match_j[j] = index of divisor matched to position j, or -1 if unmatched
    match_j = [-1] * L
    
    def dfs(i, visited):
        """
        Try to find an augmenting path for divisor i.
        i: current divisor index (0..11)
        visited: boolean array of size L to track visited positions in current DFS
        """
        for j in adj[i]:
            if visited[j]:
                continue
            visited[j] = True
            if match_j[j] == -1 or dfs(match_j[j], visited):
                match_j[j] = i
                return True
        return False
    
    # Try to match all 12 divisors
    for i in range(12):
        visited = [False] * L
        if not dfs(i, visited):
            return False
            
    return True

def _f(m):
    """
    Compute f(12, m): least positive L such that (m, m+L] has valid matching.
    """
    lo, hi = MIN_L, MAX_L
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    """
    Find m in [0, 1000000] maximizing f(12, m).
    Since f(12, m) is periodic with period P=27720, we only need to check
    one representative for each residue class mod P.
    """
    deadline = time.time() + 85.0  # Safety margin
    best_m = 0
    best_f = 0
    
    # Iterate through all residues mod P
    for r in range(P):
        if time.time() > deadline:
            break
            
        # f(12, r) is the value for any m ≡ r (mod P)
        # Since r < P <= 1000000, r is a valid candidate m
        f_val = _f(r)
        
        if f_val > best_f:
            best_f = f_val
            best_m = r
            
        # If we reach the theoretical maximum, we can stop early
        if best_f >= MAX_L:
            break
            
    return int(best_m)
# EVOLVE-BLOCK-END
