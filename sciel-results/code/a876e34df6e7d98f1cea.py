def solve_mis(candidates, L):
    if len(candidates) == 0:
        return []
    
    N = len(candidates)
    G = np.dot(candidates, candidates.T)
    threshold = L / 2.0
    A = (G > threshold).astype(np.int8)
    np.fill_diagonal(A, 0)
    
    best_set = []
    num_restarts = 20
    
    for r in range(num_restarts):
        order = list(range(N))
        random.shuffle(order)
        
        S = set()
        available = set(range(N))
        
        for i in order:
            if i not in available:
                continue
            # Check conflict with S
            if S:
                # Vectorized check
                # Convert S to list for indexing
                # This is still slow in Python loop.
                # Better: maintain a boolean array `in_S`
                pass
            
            # Let's use a boolean array for S
            # in_S = np.zeros(N, dtype=bool)
            
            # Actually, the greedy loop with `available` set is okay if N is small.
            # For N=1000, it's fine.
            
            conflict = False
            if S:
                # Check if any neighbor of i is in S
                # A[i] is a vector.
                # We can check intersection with S.
                # np.any(A[i, list(S)])
                # Creating list(S) every time is slow.
                
                # Alternative:
                # Keep `in_S` array.
                # conflict = np.any(A[i] & in_S)
                
            if not conflict:
                S.add(i)
                # in_S[i] = True
                # Remove neighbors from available
                neighbors = np.where(A[i] == 1)[0]
                for n in neighbors:
                    if n in available:
                        available.discard(n)
                        
        # Local search with in_S array
        in_S = np.zeros(N, dtype=bool)
        for i in S:
            in_S[i] = True
            
        changed = True
        while changed:
            changed = False
            # Find all i not in S
            # candidates_to_add = np.where(~in_S)[0]
            # For each, check if A[i] & in_S is all 0
            
            # Vectorized:
            # Conflicts for all non-S nodes:
            # C = A[~in_S, :] @ in_S
            # Nodes with 0 conflicts can be added.
            
            non_S = np.where(~in_S)[0]
            if len(non_S) == 0:
                break
                
            # Compute conflict counts for non-S nodes
            # A is N x N
            # A[non_S, :] is len(non_S) x N
            # Dot with in_S (N,) gives len(non_S) vector of conflict counts
            
            conflict_counts = np.dot(A[non_S, :], in_S)
            
            # Indices in non_S where count is 0
            addable = non_S[conflict_counts == 0]
            
            if len(addable) > 0:
                for idx in addable:
                    in_S[idx] = True
                    S.add(idx)
                changed = True
                
        if len(S) > len(best_set):
            best_set = list(S)
            
    return candidates[best_set]
