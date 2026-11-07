import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0 # 20s margin
    
    n = 12
    P = 27720 # lcm(1..12)
    
    # Precompute divisibility masks for k=1..144?
    # For a fixed r, x = r+k.
    # x % i == 0 iff (r+k) % i == 0.
    
    def can_match(r, L):
        # options[i] is list of k (1-indexed in 1..L) such that (r+k)%i==0
        # We want to assign distinct k to each i.
        # i ranges 1..12.
        
        # Sort i by number of options
        # options_count[i]
        # 
        # Let's just build adjacency list for left nodes 1..12 to right nodes 1..L.
        # adj[i] = [k1, k2, ...]
        
        adj = [[] for _ in range(n+1)]
        for k in range(1, L+1):
            val = r + k
            for i in range(1, n+1):
                if val % i == 0:
                    adj[i].append(k)
        
        # Sort left nodes by degree
        order = list(range(1, n+1))
        order.sort(key=lambda i: len(adj[i]))
        
        # If any node has 0 options, return False
        if any(len(adj[i]) == 0 for i in order):
            return False
            
        # Kuhn's algorithm / DFS
        # match_to[i] = k assigned to i
        # match_from[k] = i assigned to k
        match_from = [-1] * (L+1)
        
        def dfs(i, visited):
            for k in adj[i]:
                if visited[k]: continue
                visited[k] = True
                if match_from[k] == -1 or dfs(match_from[k], visited):
                    match_from[k] = i
                    return True
            return False
        
        # We need to match all n nodes.
        # Standard Kuhn's usually iterates left nodes and tries to find augmenting paths.
        # We process in order of constraint.
        
        # Reset match_from for each new L? Yes, inside can_match.
        
        # Actually, standard Kuhn's:
        # for u in left_nodes:
        #   visited = set()
        #   if not dfs(u, visited): return False
        
        # But we sorted order. Does Kuhn's work with any order?
        # Yes, it finds a max matching. If max matching size < n, return False.
        
        # Let's implement it carefully.
        
        # We need to track which left node is matched to which right node.
        # match_left[i] = k
        # match_right[k] = i
        
        # Let's use the standard implementation.
        
        # Re-initialize for each L check?
        # Yes.
        
        # Optimization: Precompute adj for all L?
        # adj depends on L (the range of k).
        # But adj[i] for L is just the prefix of adj[i] for larger L.
        # We can compute adj for max L=144 once per r.
        
        # Let's restructure.
        pass
        
    # We will compute best_r
    best_r = 0
    best_L = 0
    
    for r in range(P):
        # Binary search L
        lo, hi = 1, 144
        # Check if L=144 works? It should.
        # Actually, we just need the min L.
        
        # To save time, we can check L in increasing order?
        # Or binary search.
        
        # Let's use binary search.
        # First, verify 144 works.
        # If not, something is wrong.
        
        # Helper to check specific L
        def check_L(r, L):
             # Build adj for 1..L
             # This is O(L * n)
             # Then matching O(n * E)
             pass
             
        # Since P is 27720, and we do binary search (7 steps),
        # and each step is fast, this should be fine.
        
        # Let's implement check_L efficiently.
        
        ans = 0
        for r in range(P):
             # Find min L
             lo, hi = 1, 144
             while lo < hi:
                 mid = (lo + hi) // 2
                 if check_L(r, mid):
                     hi = mid
                 else:
                     lo = mid + 1
             if lo > best_L:
                 best_L = lo
                 best_r = r
                 
        return best_r
