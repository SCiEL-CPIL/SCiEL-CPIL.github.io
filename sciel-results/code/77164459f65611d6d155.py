import time
import sys

def _can_match(m, L, n=12):
    """
    Check if (m, m+L] contains distinct b_1...b_n with i | b_i.
    Uses DFS backtracking.
    """
    # Get the numbers in the interval
    nums = list(range(m + 1, m + L + 1))
    
    # For each i in 1..n, find indices in nums that are divisible by i
    # adj[i] will be a list of indices in nums
    adj = [[] for _ in range(n)]
    
    # Precompute options for each i
    # To speed up DFS, we sort i by the number of options (most constrained first)
    options = []
    for i in range(1, n + 1):
        opts = []
        for idx, v in enumerate(nums):
            if v % i == 0:
                opts.append(idx)
        if not opts:
            return False
        options.append((len(opts), i, opts))
        
    # Sort by number of options
    options.sort(key=lambda x: x[0])
    
    # used[i] tracks if number at index i in nums is used
    used = [False] * L
    
    def dfs(k):
        if k == n:
            return True
        _, i, opts = options[k]
        for idx in opts:
            if not used[idx]:
                used[idx] = True
                if dfs(k + 1):
                    return True
                used[idx] = False
        return False
    
    return dfs(0)

def _f_val(m, n=12):
    """
    Find the least L such that (m, m+L] allows a matching.
    """
    # Max L is known to be small, 200 is a safe upper bound
    for L in range(1, 201):
        if _can_match(m, L, n):
            return L
    return 200 # Fallback, shouldn't happen

def construct():
    start_time = time.time()
    best_m = 0
    best_L = 0
    max_m = 1000000
    
    # We can potentially skip some m if we find a very high L, 
    # but we need to be sure. 
    # A full scan is likely fast enough.
    for m in range(max_m + 1):
        # Time guard
        if time.time() - start_time > 50:
            break
        
        L = _f_val(m)
        if L > best_L:
            best_L = L
            best_m = m
            
    return best_m
