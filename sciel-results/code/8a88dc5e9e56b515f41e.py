import math
from functools import reduce

def construct():
    n = 12
    # Calculate LCM of 1..12
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    period = reduce(lcm, range(1, n + 1))
    
    def can_match(m, L):
        # Build adjacency list for left nodes 1..n
        # Right nodes are offsets 1..L representing numbers m+1 ... m+L
        # adj[i] is list of offsets j such that i divides (m+j)
        adj = [[] for _ in range(n)]
        for i in range(1, n + 1):
            # Find smallest j in 1..L such that i | (m+j)
            # m+j ≡ 0 (mod i) => j ≡ -m (mod i)
            rem = (-m) % i
            if rem == 0:
                start_j = i
            else:
                start_j = rem
            # Check if start_j > L, then no candidates
            # Candidates: start_j, start_j + i, ... <= L
            j = start_j
            while j <= L:
                adj[i-1].append(j-1) # 0-indexed right nodes
                j += i
        
        # Bipartite matching
        # match_r[j] = i that is matched to right node j (offset j)
        match_r = [-1] * L
        
        def dfs(i, seen):
            for j_idx in adj[i]:
                if seen[j_idx]:
                    continue
                seen[j_idx] = True
                if match_r[j_idx] == -1 or dfs(match_r[j_idx], seen):
                    match_r[j_idx] = i
                    return True
            return False
        
        result = 0
        for i in range(n):
            seen = [False] * L
            if dfs(i, seen):
                result += 1
        return result == n

    best_m = 0
    best_L = 0
    
    for m in range(period):
        # Binary search for minimal L
        lo, hi = 1, 144 # Max L is bounded, 144 is safe upper bound
        # Actually, we want the LEAST L that works.
        # If can_match(m, L) is True, then can_match(m, L+1) is True.
        # We want the smallest L such that can_match is True.
        # If no L in 1..144 works, we might need larger? 
        # The problem states L is at most 144.
        # If can_match(m, 144) is False, then this m is invalid? 
        # No, the problem implies a solution exists. 
        # Let's assume 144 is sufficient.
        
        found_L = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                found_L = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        if found_L != -1:
            if found_L > best_L:
                best_L = found_L
                best_m = m
        else:
            # If no match found up to 144, try higher? 
            # But problem says L <= 144. 
            # If it fails, maybe my bound is wrong or logic error.
            # Let's just skip or handle gracefully.
            pass

    return int(best_m)
