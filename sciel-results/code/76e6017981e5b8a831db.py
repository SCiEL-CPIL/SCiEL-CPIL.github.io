"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # Safety margin before 90s kill
    
    n = 12
    # Period of the problem
    period = 27720  # lcm(1,2,...,12)
    
    # For each residue r in [0, period-1], find the worst L
    # f(n, m) = least L>0 such that (m, m+L] has distinct b_i with i|b_i
    
    # Key: f(12, m) depends only on m mod 27720
    # For a given m, we binary search on L: check if matching exists in (m, m+L]
    
    # Precompute: for each i in 1..12, which positions in [1, period] are divisible by i
    # Actually, let's think differently.
    
    # For m mod period = r, the window (m, m+L] = (r, r+L] mod period (with wrap)
    # But since L <= 144 < period, no wrap-around issue if we handle it
    
    # Better approach: for each r in [0, period-1], compute f(12, r)
    # Then pick the r that maximizes f, and return any m ≡ r (mod period) in [0, 1000000]
    
    # f(12, m): least L such that there exist distinct b_1..b_12 in (m, m+L] with i|b_i
    
    # For a fixed m and L, build bipartite graph:
    # Left: indices 1..12
    # Right: numbers in (m, m+L] 
    # Edge i -> x if i divides x
    # Need perfect matching
    
    # Since L <= 144 and period = 27720, and we need to check all 27720 residues,
    # we need this to be fast.
    
    # For each residue r, the set of candidate numbers in (r, r+L] mod period
    # is determined by r and L.
    
    # Let's precompute for each i, the list of positions in [1, period] divisible by i
    # pos_div[i] = set of x in [1, period] where x % i == 0
    
    # Actually, let's use a different approach:
    # For each r, simulate increasing L from 1 to 144, and check if matching exists.
    # The first L where matching exists is f(12, r).
    
    # To check matching for given r and L:
    # Numbers in (r, r+L] mod period: these are {r+1, r+2, ..., r+L} (mod period, in [1,period])
    # For each i in 1..12, candidates = {x in that range : i | x}
    # Check if perfect matching exists via DFS/Hopcroft-Karp
    
    # Since n=12 is small, DFS-based matching is fine.
    
    def has_matching(r, L):
        """Check if there's a perfect matching for residue r and window length L."""
        # Numbers in (r, r+L] modulo period, represented in [1, period]
        # These are r+1, r+2, ..., r+L (each mod period, mapped to [1,period])
        # Since L <= 144 < period, no wrap issues if r+L <= period
        # If r + L > period, we have wrap: numbers are r+1..period and 1..(r+L-period)
        
        # For each i in 1..12, find which numbers in the window are divisible by i
        # Use bitmask for the 12 indices
        
        # Build adjacency: for each number x in window, which i divide x?
        # Or: for each i, which numbers in window are divisible by i?
        
        # Let's use DFS matching
        # match[i] = which number is assigned to index i (or -1)
        # We'll use standard bipartite matching
        
        # Numbers in window
        nums = []
        for k in range(1, L + 1):
            x = (r + k - 1) % period + 1  # This gives (r+k) mod period in [1,period]
            # Wait: (r, r+L] means r+1 to r+L
            # x = r + k for k=1..L, mod period, in [1, period]
            x = (r + k - 1) % period + 1
            nums.append(x)
        
        # For each i in 1..12, candidates are indices j where nums[j] % i == 0
        # We need to assign distinct j to each i
        
        # DFS-based bipartite matching
        # Left side: indices 1..12 (size 12)
        # Right side: positions 0..L-1 (size L, up to 144)
        
        # adj[i] = list of right-node indices j where nums[j] % (i+1) == 0
        adj = [[] for _ in range(n)]
        for j in range(L):
            x = nums[j]
            for i in range(n):
                if x % (i + 1) == 0:
                    adj[i].append(j)
        
        # Check if any i has no candidates
        for i in range(n):
            if not adj[i]:
                return False
        
        # Standard DFS bipartite matching
        # matchR[j] = which left node is matched to right node j
        matchR = [-1] * L
        
        def dfs(i, visited):
            for j in adj[i]:
                if visited[j]:
                    continue
                visited[j] = True
                if matchR[j] == -1 or dfs(matchR[j], visited):
                    matchR[j] = i
                    return True
            return False
        
        result = 0
        for i in range(n):
            visited = [False] * L
            if dfs(i, visited):
                result += 1
            else:
                return False
        return result == n
    
    # For each residue r, find f(12, r) = smallest L >= 1 with has_matching(r, L)
    # We want to maximize this L
    
    best_L = 0
    best_r = 0
    
    for r in range(period):
        if time.time() > deadline - 5:
            break
        # Binary search or linear search for smallest L
        # L is at most 144
        found = False
        for L in range(1, 145):
            if has_matching(r, L):
                if L > best_L:
                    best_L = L
                    best_r = r
                found = True
                break
    
    # Now find m in [0, 1000000] with m % period == best_r
    # m = best_r + k * period for some k >= 0, m <= 1000000
    if best_r <= 1000000:
        m = best_r
    else:
        # best_r < period = 27720 < 1000000, so this shouldn't happen
        m = best_r % (1000000 // period + 1)  # fallback
    
    # Ensure m is a Python int
    return int(m)
# EVOLVE-BLOCK-END
