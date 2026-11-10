"""Worst interval starts for distinct-multiple matching."""
import time
import random

# EVOLVE-BLOCK-START

def _compute_f(m, n=12):
    """Compute f(n, m): least L>0 s.t. (m, m+L] has distinct b_i with i|b_i."""
    
    def can_match(L):
        # Build adjacency: for each i in 1..n, which b-indices (0..L-1) work
        # b-index j corresponds to b = m+1+j
        # b is valid for i if (m+1+j) % i == 0, i.e., j % i == (i-1-m) % i
        # i.e., j ≡ (-m-1) mod i, i.e., j ≡ (i - 1 - m % i) % i ... let me just compute
        
        adj = [[] for _ in range(n)]
        for i in range(1, n+1):
            # j in [0, L-1] such that (m+1+j) % i == 0
            # (m+1) % i = r, need j % i == (i - r) % i = (-r) % i
            r = (m + 1) % i
            target = (-r) % i  # j ≡ target (mod i)
            # j = target, target+i, target+2i, ... < L
            j = target
            while j < L:
                adj[i-1].append(j)
                j += i
        
        # Bipartite matching (Kuhn's algorithm)
        match = [-1] * L  # match[j] = i matched to b-index j
        
        def dfs(i, visited):
            for j in adj[i]:
                if visited[j]:
                    continue
                visited[j] = True
                if match[j] == -1 or dfs(match[j], visited):
                    match[j] = i
                    return True
            return False
        
        found = 0
        for i in range(n):
            visited = [False] * L
            if dfs(i, visited):
                found += 1
        
        return found == n
    
    # Lower bound: max over i of (first multiple of i > m) - m
    lo = 1
    for i in range(1, n+1):
        # first multiple of i strictly greater than m
        q = m // i
        if q * i == m:
            first = (q + 1) * i
        else:
            first = (q + 1) * i
        # Actually: first multiple > m is: ((m // i) + 1) * i
        # But if m % i == 0, m itself is a multiple but not > m, so next is m+i
        # ((m//i)+1)*i handles this: if m%2==0, m//i*2=m, +1 gives (m//i+1)*2 = m+2
        lb = ((m // i) + 1) * i - m
        if lb > lo:
            lo = lb
    
    hi = 144
    
    if lo > hi:
        return hi
    
    while lo < hi:
        mid = (lo + hi) // 2
        if can_match(mid):
            hi = mid
        else:
            lo = mid + 1
    
    return lo


def construct():
    """Find m in [0, 1000000] that maximizes f(12, m)."""
    deadline = time.time() + 80  # 80s safety margin
    
    best_m = 0
    best_f = 0
    
    # f(12, m) depends only on m mod 27720 (lcm of 1..12)
    # So we can evaluate all 27720 residue classes
    # Then pick the best residue r, and find any m in [0,1000000] with m ≡ r (mod 27720)
    
    LCM = 27720
    
    # Evaluate all residue classes
    # For each r in 0..27719, compute f(12, r) (since m mod 27720 = r)
    # But wait: f(12, m) depends on m mod i for each i, which is determined by m mod 27720
    # So f(12, m) = f(12, m % 27720) ... is that true?
    # Let's verify: the condition is (m, m+L] contains b_i with i|b_i
    # b_i = m+1+j, and we need (m+1+j) % i == 0
    # This depends on m mod i, which is determined by m mod 27720
    # And L is bounded by 144 << 27720, so the structure is periodic
    # YES, f(12, m) = f(12, m % 27720) for the purpose of the matching structure
    
    # But actually, we need to be careful: m % 27720 = r means m = q*27720 + r
    # The multiples of i in (m, m+L] are the same pattern as in (r, r+L] shifted
    # So yes, f(12, m) depends only on r = m % 27720
    
    # Evaluate all 27720 residues
    results = []
    for r in range(LCM):
        if time.time() > deadline:
            break
        f_val = _compute_f(r)
        results.append((f_val, r))
        if f_val > best_f:
            best_f = f_val
            best_m = r  # r is in [0, 27719], valid m
    
    # Find the best residue and use it directly (it's in [0, 1000000])
    # best_m is already a valid m in [0, 27719] ⊂ [0, 1000000]
    
    # If time remains, verify and try local search
    if time.time() < deadline - 5:
        # Local search around best_m
        for delta in range(-100, 101):
            if time.time() > deadline:
                break
            m2 = best_m + delta
            if 0 <= m2 <= 1000000:
                f2 = _compute_f(m2)
                if f2 > best_f:
                    best_f = f2
                    best_m = m2
    
    return int(best_m)

# EVOLVE-BLOCK-END
