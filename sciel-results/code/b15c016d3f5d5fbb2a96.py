import time

P = 27720  # LCM(1..12)

def precompute_best():
    """Precompute f(12, m) for all m in [0, P) and return the best m and its L."""
    
    # Precompute masks: for each residue r, which i (1..12) divide r
    masks = [0] * P
    for r in range(P):
        m = 0
        for i in range(1, 13):
            if r % i == 0:
                m |= (1 << (i - 1))
        masks[r] = m
    
    best_m = 0
    best_L = 0
    
    for start in range(P):
        # For this start residue, find min L such that (start, start+L] has perfect matching
        # Interval is start+1 to start+L
        # We need to assign each i (1..12) to a distinct j in [0, L-1] such that i | (start+1+j)
        
        # Binary search for min L
        lo, hi = 1, 144
        min_L = hi + 1  # If no L works, min_L stays > 144
        
        while lo <= hi:
            mid = (lo + hi) // 2
            # Check if matching exists for L = mid
            # Build adjacency: for each i (0..11), list of j (0..mid-1) where (i+1) | (start+1+j)
            
            adj = [[] for _ in range(12)]
            base = start + 1
            for j in range(mid):
                x = base + j
                r = x % P
                mask = masks[r]
                for i in range(12):
                    if mask & (1 << i):
                        adj[i].append(j)
            
            # Kuhn's algorithm with ordering by degree (process constrained vertices first)
            match_to = [-1] * mid      # match_to[j] = i
            match_from = [-1] * 12     # match_from[i] = j
            
            # Order vertices by their degree (ascending) for better performance
            order = sorted(range(12), key=lambda i: len(adj[i]))
            
            def dfs(i, visited):
                for j in adj[i]:
                    if visited[j]:
                        continue
                    visited[j] = True
                    if match_to[j] == -1 or dfs(match_to[j], visited):
                        match_to[j] = i
                        match_from[i] = j
                        return True
                return False
            
            count = 0
            for i in order:
                visited = [False] * mid
                if dfs(i, visited):
                    count += 1
                else:
                    break  # Can't achieve perfect matching
            
            if count == 12:
                min_L = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        if min_L <= 144 and min_L > best_L:
            best_L = min_L
            best_m = start
            if best_L == 144:
                break
    
    return best_m, best_L

# Precompute at module load
_BEST_M, _BEST_L = precompute_best()

def construct():
    """Return the precomputed best m that maximizes f(12, m)."""
    return int(_BEST_M)
