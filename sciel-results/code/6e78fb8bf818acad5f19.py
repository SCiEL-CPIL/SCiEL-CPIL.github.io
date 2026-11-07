import sys

def construct():
    N = 12
    PERIOD = 27720  # LCM(1..12)
    MAX_L = 200     # Safe upper bound for L
    
    def can_match(m, L):
        # Left nodes: 0..11 representing i=1..12
        # Right nodes: values k in (m, m+L]
        # We need a matching where left node i matches a right node k such that i divides k.
        
        # Build adjacency list for left nodes
        # adj[i] = list of k values
        adj = [[] for _ in range(N)]
        
        # To make matching faster, we can map k to an index 0..L-1
        # But k values are not contiguous indices in a simple way for the matching algorithm
        # unless we just use the value k as the identifier.
        # Since L is small, we can just use k as the ID.
        
        for i in range(1, N + 1):
            idx = i - 1
            # Find multiples of i in (m, m+L]
            # First multiple > m
            # k = m + 1 + ((i - (m + 1) % i) % i)
            # Simpler:
            start_k = m + 1
            # Round up to next multiple of i
            rem = start_k % i
            if rem != 0:
                start_k += i - rem
            else:
                # start_k is a multiple, but we need > m, so if start_k == m+1 and m+1 is multiple, it's valid.
                # The formula above handles this. If rem==0, start_k is already a multiple.
                pass
            
            k = start_k
            while k <= m + L:
                adj[idx].append(k)
                k += i
        
        # Bipartite matching using DFS (Kuhn's algorithm)
        # match_to[k] = i (index 0..11) that k is matched to
        # We need to check if all i can be matched.
        
        # To speed up, we can use a dictionary or a large array for match_to
        # Max k is m + L <= 27720 + 200.
        # We can use a list of size PERIOD + MAX_L + 10, initialized to -1.
        # But re-initializing is O(PERIOD).
        # Better: use a dict or just a list and clear used keys.
        # Or, since L is small, the number of right nodes is small.
        # Let's collect all unique k's first.
        
        all_ks = set()
        for i in range(N):
            all_ks.update(adj[i])
        
        if len(all_ks) < N:
            return False
            
        # Map k to index 0..M-1 for faster array access in matching?
        # Or just use k as key in a dict.
        # Dict lookup is O(1) but slower than list.
        # Given small size, list is better if we can map.
        
        # Let's stick to a simple recursive DFS with a "seen" set for the current DFS step.
        # Standard Kuhn's:
        # match_right: dict k -> i
        match_right = {}
        
        def dfs(i, seen):
            for k in adj[i]:
                if k in seen:
                    continue
                seen.add(k)
                # If k is not matched, or if the i that currently matches k can find another match
                if k not in match_right or dfs(match_right[k], seen):
                    match_right[k] = i
                    return True
            return False
        
        # We want to match all N left nodes.
        # To be efficient, we can iterate left nodes in some order.
        # Usually, iterating 0..N-1 is fine.
        
        # Optimization: Sort left nodes by number of options (ascending) to prune earlier?
        # Or just standard.
        
        # We need to reset match_right for each check? Yes.
        # And we need to ensure we find a matching of size N.
        
        # Note: Kuhn's algorithm finds a maximal matching, not necessarily maximum?
        # No, Kuhn's (DFS based) finds maximum matching if run for all vertices.
        
        count = 0
        for i in range(N):
            seen = set()
            if dfs(i, seen):
                count += 1
            else:
                # If one fails, can we continue?
                # In bipartite matching, if a vertex cannot be matched, it might be because
                # of previous choices. But Kuhn's with backtracking (the `dfs(match_right[k], seen)` part)
                # handles reassignment.
                # So if dfs returns False, this specific i cannot be matched in the current state?
                # Actually, if we process i=0,1,...,N-1, and one fails, the total matching size is less than N.
                # We can break early.
                return False
                
        return True

    best_m = 0
    best_L = 0
    
    # Search m in one period
    for m in range(PERIOD):
        # Binary search for minimum L
        # L must be at least 1.
        # Max L is bounded by PERIOD, but practically much smaller.
        # Let's use 200 as high bound.
        low = 1
        high = MAX_L
        min_L = -1
        
        # Check if MAX_L is sufficient?
        # It should be, as period is 27720.
        # But for safety, if not, we might miss it.
        # However, 200 is very likely enough.
        
        while low <= high:
            mid = (low + high) // 2
            if can_match(m, mid):
                min_L = mid
                high = mid - 1
            else:
                low = mid + 1
                
        if min_L != -1 and min_L > best_L:
            best_L = min_L
            best_m = m
            
    return best_m
