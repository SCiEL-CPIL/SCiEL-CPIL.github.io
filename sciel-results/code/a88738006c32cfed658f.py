import sys
sys.setrecursionlimit(10000)

def construct():
    n = 12
    period = 27720 # lcm(1..12)
    
    def max_matching(m, L):
        # Left side: 0..n-1 representing indices 1..n
        # Right side: numbers in (m, m+L]
        # We build adjacency list for left nodes
        # adj[i] = list of right node indices (or just the numbers, but we need distinctness)
        # Let's map numbers to indices 0..K-1
        # Actually, simpler: just use the numbers themselves as identifiers in the matching array?
        # No, standard bipartite matching uses indices.
        
        # Identify candidate numbers
        candidates = []
        for x in range(m + 1, m + L + 1):
            if x > 0:
                candidates.append(x)
        
        K = len(candidates)
        if K < n:
            return False
            
        # adj[i] will be a list of indices into 'candidates'
        adj = [[] for _ in range(n)]
        for i in range(n):
            idx_i = i + 1
            for j, x in enumerate(candidates):
                if x % idx_i == 0:
                    adj[i].append(j)
        
        # Check if any node has no candidates
        for i in range(n):
            if not adj[i]:
                return False
                
        # Bipartite matching (DFS)
        match_to_right = [-1] * K # match_to_right[j] = i if j is matched to i
        match_to_left = [-1] * n  # match_to_left[i] = j if i is matched to j
        
        def dfs(u, visited):
            for v in adj[u]:
                if visited[v]:
                    continue
                visited[v] = True
                if match_to_right[v] == -1 or dfs(match_to_right[v], visited):
                    match_to_right[v] = u
                    match_to_left[u] = v
                    return True
            return False
            
        count = 0
        for u in range(n):
            visited = [False] * K
            if dfs(u, visited):
                count += 1
            if count == n:
                return True
        return count == n

    best_m = 0
    best_L = 0
    
    for m in range(period):
        # Find the smallest L such that matching exists
        # Max possible L is small, but let's cap it at 100 to be safe
        found = False
        for L in range(1, 101):
            if max_matching(m, L):
                if L > best_L:
                    best_L = L
                    best_m = m
                found = True
                break
        # If not found by 100, it's an error, but theoretically it should be found
        # Given n=12, L will be very small.
        
    return best_m
