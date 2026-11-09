import time

P = 27720  # LCM(1..12)

def has_matching(m, L):
    """Check if interval (m, m+L] has distinct b_1..b_12 with i|b_i."""
    base = m + 1
    # Build adjacency: for divisor d (1..12), which positions j in [0,L) have (base+j)%d==0
    adj = []
    for d in range(1, 13):
        target = (-base) % d
        positions = list(range(target, L, d))
        adj.append(positions)
    
    # Order left nodes by ascending degree (most constrained first)
    order = sorted(range(12), key=lambda i: len(adj[i]))
    
    # Backtracking with MRV heuristic
    match_to = [-1] * L  # match_to[j] = which left node is matched to j
    match_from = [-1] * 12
    
    def backtrack(idx):
        if idx == 12:
            return True
        i = order[idx]
        for j in adj[i]:
            if match_to[j] != -1:
                continue
            match_to[j] = i
            match_from[i] = j
            if backtrack(idx + 1):
                return True
            match_to[j] = -1
            match_from[i] = -1
        return False
    
    return backtrack(0)

def compute_f(m):
    """Compute f(12, m): least positive L such that matching exists."""
    for L in range(12, 145):
        if has_matching(m, L):
            return L
    return 145  # should not happen

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    best_m = 0
    best_L = 0
    
    for m in range(P):
        if time.time() > deadline:
            break
        
        L = compute_f(m)
        if L > best_L:
            best_L = L
            best_m = m
            if best_L == 144:
                break
    
    return best_m
