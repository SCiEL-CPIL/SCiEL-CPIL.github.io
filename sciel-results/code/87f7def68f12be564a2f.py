import itertools

def construct(n):
    if n == 1:
        return "1"
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Map from permutation tuple to index for O(1) lookup
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Track visited permutations
    visited = [False] * N
    
    # Initialize with the first permutation (1..n)
    current_word = list(range(1, n + 1))
    visited[perm_to_idx[tuple(current_word)]] = True
    
    # To efficiently find "least recently used", we can track the last seen index of each symbol
    # But since we are appending, we can just look at the current_word.
    # For n=7, current_word length is ~5000-6000, so scanning is fast enough.
    
    for _ in range(N - 1):
        # Get the last n-1 symbols
        if len(current_word) < n - 1:
            # Should not happen if n >= 2 and we start with n symbols
            prefix = tuple(current_word)
        else:
            prefix = tuple(current_word[-(n - 1):])
            
        best_next_symbol = -1
        best_recency = float('inf')
        best_perm_idx = -1
        
        # Try each possible next symbol
        for s in range(1, n + 1):
            # Form the new permutation window
            # The new window is prefix + (s,)
            # But wait, the window of length n is current_word[-(n-1):] + [s]
            new_window = prefix + (s,)
            
            # Check if this window is a valid permutation of 1..n
            # Since prefix is a suffix of a valid permutation, and s is 1..n,
            # it might contain duplicates if s is already in prefix.
            # A superpermutation window must be a permutation, so all symbols distinct.
            if len(set(new_window)) != n:
                continue
                
            # Check if this permutation has been visited
            if new_window not in perm_to_idx:
                continue
                
            idx = perm_to_idx[new_window]
            if visited[idx]:
                continue
            
            # Calculate recency of symbol s
            # Find the last index in current_word where s appears
            # We search from the end
            recency = -1 # -1 means not found, which is "infinitely old"
            for j in range(len(current_word) - 1, -1, -1):
                if current_word[j] == s:
                    recency = j
                    break
            
            # We want to MINIMIZE recency (earliest occurrence is best)
            # If recency is -1, it's the best.
            # If we have a tie in recency, we prefer smaller s (lexicographic order)
            
            # Let's define a score: 
            # Primary: -recency (higher is better, so -1 becomes 1, 0 becomes 0, etc? No.)
            # Let's just compare tuples: (recency, s)
            # We want the smallest recency. If recency is -1, it's smallest.
            # So we minimize (recency, s).
            
            current_score = (recency, s)
            
            if best_next_symbol == -1 or current_score < (best_recency, best_next_symbol):
                best_next_symbol = s
                best_recency = recency
                best_perm_idx = idx
        
        if best_next_symbol == -1:
            # Fallback: if no valid move found (should not happen in a complete graph of perms with this heuristic? 
            # Actually, it might get stuck if it visits all perms reachable via this greedy path? 
            # No, the graph of permutations is connected. But greedy might get stuck in a local minimum of unvisited?
            # If stuck, we must backtrack or just fail. But for n=7, greedy LRU usually works.
            # If we really get stuck, we can just append any unvisited perm fully.
            # Find any unvisited perm
            for i in range(N):
                if not visited[i]:
                    visited[i] = True
                    p = perms[i]
                    current_word.extend(p)
                    break
        else:
            visited[best_perm_idx] = True
            current_word.append(best_next_symbol)
            
    return "".join(map(str, current_word))
