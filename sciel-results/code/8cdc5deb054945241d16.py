import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 12
    K = 1
    for i in range(1, n + 1):
        K = K * i // math.gcd(K, i)
    # K = 27720
    
    max_L = 144
    
    # Precompute candidates for each i and each residue r?
    # No, candidates depend on r.
    # But (r+j) % i == 0  <=>  j % i == (-r) % i.
    # So for a fixed i and r, the valid j's are those where j % i == (-r) % i.
    # We can precompute for each i and each rem in 0..i-1, the list of j in 1..max_L with j % i == rem.
    
    # valid_js[i][rem] = list of j in 1..max_L such that j % i == rem
    valid_js = [[[] for _ in range(i)] for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, max_L + 1):
            rem = j % i
            valid_js[i][rem].append(j)
            
    # Now for a given r and L, candidates for i are valid_js[i][(-r) % i] filtered by j <= L.
    
    def has_matching(r, L):
        # candidates[i] is list of j in 1..L such that (r+j) % i == 0
        # We need to assign distinct j to each i in 1..n
        # Use DFS / Backtracking
        # Sort i by number of candidates to prune early
        items = []
        for i in range(1, n + 1):
            rem = (-r) % i
            cands = [j for j in valid_js[i][rem] if j <= L]
            if not cands:
                return False
            items.append((i, cands))
            
        # Sort by len(cands) ascending
        items.sort(key=lambda x: len(x[1]))
        
        used = [False] * (L + 1)
        
        def dfs(idx):
            if idx == n:
                return True
            i, cands = items[idx]
            for j in cands:
                if not used[j]:
                    used[j] = True
                    if dfs(idx + 1):
                        return True
                    used[j] = False
            return False
            
        return dfs(0)

    # Compute L(r) for all r
    L_vals = [0] * K
    for r in range(K):
        for L in range(1, max_L + 1):
            if has_matching(r, L):
                L_vals[r] = L
                break
        if time.time() > deadline:
            # If we run out of time, we might not have all L_vals.
            # But K is only 27720, should be fast.
            pass

    # Find max L
    max_L_found = max(L_vals)
    
    # Find residues with this max L
    best_residues = [r for r in range(K) if L_vals[r] == max_L_found]
    
    # Find smallest m in [0, 1000000] with m % K in best_residues
    best_m = None
    for r in best_residues:
        # m = r + k*K
        # We want smallest m >= 0
        # r is in [0, K), so r itself is a candidate if r <= 1000000
        # Since K=27720 and 1000000 > K, r is always valid.
        # But we want the global smallest m.
        # The candidates are r, r+K, r+2K, ...
        # The smallest for a given r is r.
        # So we just need the smallest r in best_residues?
        # Yes, because m = r + k*K. For k=0, m=r.
        # Since r < K, and K < 1000000, m=r is the smallest for that residue.
        # So we just need min(best_residues).
        
        if best_m is None or r < best_m:
            best_m = r
            
    return best_m
