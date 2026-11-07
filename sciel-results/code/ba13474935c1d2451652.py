import time

def construct():
    N = 12
    LCM = 27720
    MAX_M = 1000000
    
    # We want to find m that maximizes f(N, m).
    # f(N, m) is the minimal L such that (m, m+L] contains distinct b_i divisible by i.
    # Due to periodicity, f(N, m) depends only on m % LCM.
    
    # Precompute min_L for each residue r in [0, LCM)
    # min_L[r] = minimal L such that matching exists for start residue r.
    
    def can_match(r, L):
        # Check if there is a perfect matching for indices 1..N to distinct numbers in (r, r+L]
        # such that i | b_i.
        
        # Build adjacency list for indices 1..N
        # adj[i] = list of numbers b in (r, r+L] such that b % i == 0
        # We need to assign distinct b to each i.
        
        # To speed up, we can represent the available numbers as a set or list.
        # The numbers are r+1, ..., r+L.
        # We can check if a number b is available and divisible by i.
        
        # Let's use a matching algorithm.
        # match[b_index] = i assigned to b, or -1.
        # But b ranges from r+1 to r+L. We can index them 0..L-1.
        
        # Actually, simpler:
        # For each index i (1..N), the candidates are multiples of i in (r, r+L].
        # Let's collect candidates for each i.
        
        candidates = [[] for _ in range(N+1)]
        for i in range(1, N+1):
            # Find first multiple of i > r
            first_mult = ((r // i) + 1) * i
            for b in range(first_mult, r + L + 1, i):
                candidates[i].append(b)
                
        # Now we have candidates[i] for each i.
        # We need to pick one b from each candidates[i] such that all picked b are distinct.
        # This is a bipartite matching problem.
        # Left: i (1..N). Right: b (values in union of candidates).
        
        # Since N is small, we can use a recursive backtracking with memoization or just simple DFS.
        # Let's use a standard bipartite matching (Kuhn's algorithm).
        
        # Map b values to indices 0..M-1 for the right side?
        # The union of candidates can be up to L * N size, but distinct values are <= L.
        # So right side size <= L.
        
        # Let's identify all unique b's involved.
        all_bs = set()
        for i in range(1, N+1):
            for b in candidates[i]:
                all_bs.add(b)
        
        if len(all_bs) < N:
            return False
            
        # Map b to index 0..len(all_bs)-1
        b_list = list(all_bs)
        b_to_idx = {b: idx for idx, b in enumerate(b_list)}
        
        # adj[i] now contains indices of b's
        adj = [[] for _ in range(N+1)]
        for i in range(1, N+1):
            for b in candidates[i]:
                adj[i].append(b_to_idx[b])
                
        # Kuhn's algorithm
        match_to_b = [-1] * len(b_list) # match_to_b[b_idx] = i
        
        def dfs(i, visited):
            for b_idx in adj[i]:
                if visited[b_idx]:
                    continue
                visited[b_idx] = True
                if match_to_b[b_idx] == -1 or dfs(match_to_b[b_idx], visited):
                    match_to_b[b_idx] = i
                    return True
            return False
            
        for i in range(1, N+1):
            visited = [False] * len(b_list)
            if not dfs(i, visited):
                return False
                
        return True

    min_Ls = [0] * LCM
    best_L = 0
    best_residues = []
    
    # Time guard
    start_time = time.time()
    deadline = start_time + 80 # Leave 10s margin
    
    for r in range(LCM):
        if time.time() > deadline:
            # If we run out of time, break and use best found so far
            # But we need to finish the loop or return a valid m.
            # Since LCM is 27720, it should be fast.
            pass
            
        # Binary search or linear search for minimal L?
        # L is small (max 144). Linear search is fine.
        # We know L must be at least 1.
        # We can start from 1 and go up.
        # Optimization: We can start from a lower bound.
        # A lower bound for L is the max over i of (smallest multiple of i > r - r).
        # Actually, just iterate L from 1 to 150.
        
        found_L = None
        for L in range(1, 150):
            if can_match(r, L):
                found_L = L
                break
                
        if found_L is None:
            found_L = 150 # Should not happen
            
        min_Ls[r] = found_L
        if found_L > best_L:
            best_L = found_L
            best_residues = [r]
        elif found_L == best_L:
            best_residues.append(r)
            
    # Find m in [0, MAX_M] with residue in best_residues
    # m = k * LCM + r
    # We want the smallest m >= 0.
    # For each r in best_residues, the smallest m is r (if r <= MAX_M).
    # Since LCM=27720 and MAX_M=1000000, r is always <= MAX_M.
    # So the smallest m is min(best_residues).
    
    m = min(best_residues)
    
    return int(m)
