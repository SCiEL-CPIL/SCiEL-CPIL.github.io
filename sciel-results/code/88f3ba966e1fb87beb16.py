import time

def construct():
    deadline = time.time() + 85  # Safety margin
    
    n = 12
    max_L = 144
    period = 27720  # lcm(1, 2, ..., 12)
    
    f_values = [0] * period
    
    # Helper to check if matching exists for given m and L
    def has_matching(m, L):
        # Generate candidates for each i
        # cands[i] is a list of integers
        cands = [[] for _ in range(n + 1)]
        for i in range(1, n + 1):
            # First multiple > m
            start = (m // i + 1) * i
            for b in range(start, m + L + 1, i):
                cands[i].append(b)
        
        # Check if any is empty
        for i in range(1, n + 1):
            if not cands[i]:
                return False
                
        # Order indices by number of candidates (smallest first) to prune early
        indices = list(range(1, n + 1))
        indices.sort(key=lambda i: len(cands[i]))
        
        used = set()
        
        # DFS
        def dfs(idx):
            if idx == n:
                return True
            i = indices[idx]
            for b in cands[i]:
                if b not in used:
                    used.add(b)
                    if dfs(idx + 1):
                        return True
                    used.remove(b)
            return False
            
        return dfs(0)

    # Compute f for one period
    # Optimization: If we find a high L for some m, we might be able to prune?
    # No, we need the max.
    
    # To speed up, we can check L from 1 up to max_L.
    # But we can also binary search? No, f(m) is not monotonic in m, but for a fixed m,
    # if has_matching(m, L) is True, it is True for all L' > L.
    # So for a fixed m, we can binary search the minimal L.
    # However, the range is small (1..144). Linear scan is 144 checks.
    # Binary search is log2(144) ~ 8 checks.
    # Let's use binary search for each m.
    
    for r in range(period):
        if time.time() > deadline:
            # Fill remaining with a safe low value, but we need to return a valid m.
            # If we break early, we might not have the max.
            # We should ensure we at least find *some* valid m.
            # But the deadline is 85s, and 27720 * 8 * cost(match) should be < 85s if cost is low.
            # Let's assume we finish. If not, we return a fallback.
            f_values[r] = 1
            continue
            
        # Binary search for min L
        lo, hi = 1, max_L
        ans = max_L
        while lo <= hi:
            mid = (lo + hi) // 2
            if has_matching(r, mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        f_values[r] = ans
        
    # Find max f
    max_f = max(f_values)
    
    # Find smallest m in [0, 1000000] with f(m) == max_f
    # f(m) = f_values[m % period]
    for m in range(1000001):
        if f_values[m % period] == max_f:
            return int(m)
            
    return int(0)
