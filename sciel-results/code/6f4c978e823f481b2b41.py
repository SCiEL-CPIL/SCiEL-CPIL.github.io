import sys
sys.setrecursionlimit(10000)

def construct():
    P = 27720
    best_m = 0
    best_L = 0
    
    # Precompute divisors or just use modulo
    # We need to check for m in [0, P-1]
    
    def check(m, L):
        # Returns True if there is a matching for window (m, m+L]
        # Indices 1..12
        # Candidates for i are numbers in (m, m+L] divisible by i
        
        # To speed up, we can just check Hall's condition or use DFS
        # DFS is safer and fast for N=12
        
        # Build list of candidates for each i
        # candidates[i] = list of numbers
        # We need to assign distinct numbers.
        
        # Optimization: If any i has 0 candidates, return False.
        # Sort i by number of candidates.
        
        cand = {}
        for i in range(1, 13):
            # Numbers x in (m, m+L] such that x % i == 0
            # First multiple > m:
            # start = m + 1
            # first_mult = ((m + 1 + i - 1) // i) * i
            # Actually:
            # rem = (m + 1) % i
            # if rem == 0: first = m + 1
            # else: first = m + 1 + (i - rem)
            
            start = m + 1
            end = m + L
            # Find first multiple of i >= start
            if start % i == 0:
                first = start
            else:
                first = start + (i - start % i)
            
            if first > end:
                return False # No candidate for i
            
            # List of candidates
            # cands = list(range(first, end + 1, i))
            # We don't need the list, we can just iterate in DFS
            # But for DFS, we need to know which are available.
            # Since L is small, we can just store the set of used numbers.
            
            cand[i] = list(range(first, end + 1, i))
            
        # Sort indices by len(cand[i])
        indices = sorted(range(1, 13), key=lambda x: len(cand[x]))
        
        used = set()
        
        def dfs(idx):
            if idx == 12:
                return True
            i = indices[idx]
            for num in cand[i]:
                if num not in used:
                    used.add(num)
                    if dfs(idx + 1):
                        return True
                    used.remove(num)
            return False
            
        return dfs(0)

    for m in range(P):
        # Binary search or linear search for L?
        # Linear is fine since L is small.
        for L in range(1, 145):
            if check(m, L):
                if L > best_L:
                    best_L = L
                    best_m = m
                break
                
    return best_m
