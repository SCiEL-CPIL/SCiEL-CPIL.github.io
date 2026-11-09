import time

# EVOLVE-BLOCK-START
P = 27720  # LCM(1..12)

def _precompute_masks():
    """For each residue r in [0, P), compute bitmask of which i (1..12) divide r."""
    masks = [0] * P
    for r in range(P):
        m = 0
        for i in range(1, 13):
            if r % i == 0:
                m |= (1 << (i - 1))
        masks[r] = m
    return masks

MASKS = _precompute_masks()

def _has_matching(m, L):
    """Check if interval (m, m+L] contains distinct b_1..b_12 with i | b_i."""
    if L < 12:
        return False
    base = m + 1
    base_mod = base % P
    
    # For each i (1..12), find positions j in [0, L) where (m+1+j) is divisible by i
    # (m+1+j) % i == 0  <=>  (base + j) % i == 0  <=>  j % i == (-base) % i
    # But since base_mod = base % P, and P is multiple of i, base_mod % i = base % i
    # So we need (base_mod + j) % i == 0, i.e., j % i == (-base_mod) % i
    
    # For each i, the valid positions j are those where j ≡ (-base_mod) mod i
    # Within [0, L), these are: r0, r0+i, r0+2i, ... where r0 = (-base_mod) % i
    # If r0 >= L, no positions; else positions are r0, r0+i, ... < L
    
    # Build adjacency: adj[i] = list of position indices j (0-indexed) that are divisible by i+1
    adj = [[] for _ in range(12)]
    for i in range(12):
        divisor = i + 1
        r0 = (-base_mod) % divisor
        if r0 < L:
            for j in range(r0, L, divisor):
                adj[i].append(j)
    
    # Now do bipartite matching: left side = 12 divisors, right side = L positions
    # Use DFS augmenting path (Kuhn's algorithm)
    match = [-1] * L  # match[j] = which divisor i is matched to position j
    
    def dfs(i, visited):
        for j in adj[i]:
            if not visited[j]:
                visited[j] = True
                if match[j] == -1 or dfs(match[j], visited):
                    match[j] = i
                    return True
        return False
    
    for i in range(12):
        visited = [False] * L
        if not dfs(i, visited):
            return False
    return True

def _compute_f(m):
    """Compute f(12, m): least positive L such that (m, m+L] has valid matching."""
    # L must be at least 12 (need 12 distinct numbers)
    # Upper bound: since patterns repeat every P=27720, worst case L is bounded
    # In practice, L is small. Let's try up to P to be safe, but likely much less.
    # Actually, since we just need 12 distinct multiples, and in any window of length
    # that's a multiple of LCM(1..12), we're guaranteed to have enough. But the
    # window is (m, m+L], so length L.
    # Let's try L from 12 upward.
    for L in range(12, P + 1):
        if _has_matching(m, L):
            return L
    return P  # Should not reach here in practice

def construct():
    deadline = time.time() + 85.0  # Leave 5s margin for evaluator overhead
    best_m = 0
    best_f = 0
    
    for r in range(P):
        if time.time() > deadline:
            break
        f_val = _compute_f(r)
        if f_val > best_f:
            best_f = f_val
            best_m = r
    
    return int(best_m)
# EVOLVE-BLOCK-END
