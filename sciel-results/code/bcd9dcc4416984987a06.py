import itertools
from collections import deque

def construct(n):
    if n == 1:
        return "1"
    
    # Generate all permutations and sort them lexicographically
    perms = sorted(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Map from prefix to list of indices of permutations starting with that prefix
    prefix_to_indices = {}
    for i, p in enumerate(perms):
        for k in range(1, n):
            prefix = p[:k]
            if prefix not in prefix_to_indices:
                prefix_to_indices[prefix] = []
            prefix_to_indices[prefix].append(i)
    
    # Track which permutations have been visited
    visited = [False] * N
    visited[0] = True
    
    # Start with the first permutation
    current_word = list(perms[0])
    
    # Maintain a deque of the last n-1 symbols for circular buffer
    # Also track recent usage order for LRU heuristic
    recent_order = deque(perms[0][1:])  # last n-1 symbols
    
    for _ in range(N - 1):
        found = False
        
        # Try max overlap first
        for k in range(n - 1, 0, -1):
            suffix = tuple(current_word[-k:])
            candidates = prefix_to_indices.get(suffix, [])
            
            if not candidates:
                continue
            
            # Among unvisited candidates, prefer those that use symbols not recently used
            # (LRU heuristic: least recently used symbol first)
            best_idx = -1
            best_score = -1
            
            # Get the last symbol used (most recently) for LRU comparison
            # We want to minimize the "recency" of the new symbol
            # If the new permutation's first new symbol (at position k) hasn't been used recently, prefer it
            
            for idx in candidates:
                if not visited[idx]:
                    next_perm = perms[idx]
                    # The new symbols being added are next_perm[k:]
                    # The first new symbol is next_perm[k]
                    new_symbol = next_perm[k]
                    
                    # Check how recently this symbol was used
                    # If it's not in recent_order, it's very good (score = n)
                    # Otherwise, score = position from end (0 = most recent, n-1 = least recent)
                    try:
                        pos = list(recent_order).index(new_symbol)
                        score = n - 1 - pos  # higher is better (less recent)
                    except ValueError:
                        score = n  # not in recent window, best case
                    
                    if score > best_score:
                        best_score = score
                        best_idx = idx
            
            if best_idx != -1:
                visited[best_idx] = True
                next_perm = perms[best_idx]
                # Append the new symbols
                current_word.extend(next_perm[k:])
                
                # Update recent_order: remove oldest, add new symbols
                for sym in next_perm[k:]:
                    recent_order.append(sym)
                    if len(recent_order) > n - 1:
                        recent_order.popleft()
                
                found = True
                break
        
        if not found:
            # No overlap > 0, pick the lexicographically smallest unvisited permutation
            for i in range(N):
                if not visited[i]:
                    visited[i] = True
                    next_perm = perms[i]
                    current_word.extend(next_perm)
                    
                    # Update recent_order
                    for sym in next_perm:
                        recent_order.append(sym)
                        if len(recent_order) > n - 1:
                            recent_order.popleft()
                    
                    found = True
                    break
                    
    return "".join(map(str, current_word))
