import time
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin (90s limit)
    
    P = 27720  # LCM(1, 2, ..., 12)
    N = 12     # number of divisors to match
    
    # Precompute for each i in 1..12, the residues mod P where a number is divisible by i
    # For a number x, x % i == 0. Since x = r + 1 + j, we need (r+1+j) % i == 0
    # i.e., j % i == (i - 1 - r) % i, i.e., j ≡ (i-1-r) mod i
    
    # For matching: left nodes are i=1..12, right nodes are j=0..L-1
    # Edge exists if (r+1+j) % i == 0
    
    # We'll compute f(r) for each residue r in 0..P-1
    # f(r) = smallest L >= 1 such that there's a perfect matching
    
    # Strategy: for each r, try L = 1, 2, 3, ... until matching found
    # But we can be smarter: use the structure
    
    # For a given r and L, build adjacency: for each i, which j in 0..L-1 work
    # j works for i iff (r + 1 + j) % i == 0, i.e., j ≡ (i - 1 - r) % i (mod i)
    
    # Precompute for each r and each i, the list of j values in 0..143 that satisfy the condition
    # Max L we need to check is at most 144 per the problem statement
    
    # For speed, precompute masks: for each r, for each i, a bitmask of j in 0..143 where edge exists
    # Then use bitmask matching
    
    # Actually, let's just do direct matching with pruning
    
    def has_perfect_matching(r, L):
        """Check if there's a perfect matching for residue r with window size L."""
        # adj[i] = set of j in 0..L-1 such that (r+1+j) % i == 0
        adj = [set() for _ in range(N)]
        for j in range(L):
            val = r + 1 + j
            for i in range(1, N + 1):
                if val % i == 0:
                    adj[i-1].add(j)
        
        # Use DFS with backtracking, trying most constrained first
        # Order: sort by degree (ascending)
        order = sorted(range(N), key=lambda i: len(adj[i]))
        
        # Check Hall's condition quickly: if any i has degree 0, fail
        for i in range(N):
            if len(adj[i]) == 0:
                return False
        
        # DFS matching
        match_to_j = [-1] * N  # match_to_j[i] = which j is matched to i, or -1
        
        def dfs(i, visited):
            for j in adj[i]:
                if visited[j]:
                    continue
                visited[j] = True
                if match_to_j[j] < 0 or dfs(match_to_j[j], visited):
                    match_to_j[j] = i
                    return True
            return False
        
        # Standard bipartite matching: try to match each left node
        # But we need perfect matching, so we need all 12 matched
        # Use standard Kuhn's algorithm
        match_j_to_i = [-1] * L  # match_j_to_i[j] = i or -1
        
        def try_match(i, visited):
            for j in adj[i]:
                if visited[j]:
                    continue
                visited[j] = True
                if match_j_to_i[j] < 0 or try_match(match_j_to_i[j], visited):
                    match_j_to_i[j] = i
                    return True
            return False
        
        count = 0
        for i in order:
            visited = [False] * L
            if try_match(i, visited):
                count += 1
            else:
                return False
        
        return count == N
    
    # Precompute f(r) for all residues
    best_L = 0
    best_r = 0
    
    # For each residue r, find f(r)
    # f(r) is the smallest L >= 1 with a perfect matching
    # We can binary search or just iterate L from 1 upward
    
    # Since max L is 144, just iterate
    for r in range(P):
        if time.time() > deadline:
            break
        for L in range(1, 145):
            if has_perfect_matching(r, L):
                if L > best_L:
                    best_L = L
                    best_r = r
                break
    
    # Find the actual m <= 1000000 with residue best_r mod P
    # m ≡ best_r (mod P), m in [0, 1000000]
    # Smallest such m: best_r itself if best_r <= 1000000
    # Otherwise, best_r + k*P for some k... but best_r < P = 27720, so best_r <= 1000000 always
    m = best_r
    if m > 1000000:
        m = m % P  # but this won't happen since P < 1000000
    
    return int(m)

# EVOLVE-BLOCK-END
