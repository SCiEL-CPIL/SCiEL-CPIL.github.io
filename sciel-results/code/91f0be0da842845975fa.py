import itertools

def construct(n):
    if n == 1:
        return "1"
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Map from tuple to index for quick lookup
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # For greedy fallback
    prefix_to_indices = {}
    for i, p in enumerate(perms):
        for k in range(1, n):
            prefix = p[:k]
            if prefix not in prefix_to_indices:
                prefix_to_indices[prefix] = []
            prefix_to_indices[prefix].append(i)

    visited = set()
    # Start with first permutation
    current_word = list(perms[0])
    visited.add(tuple(perms[0]))
    
    # Track last occurrence of each symbol for LRU
    last_occ = {i: -1 for i in range(1, n+1)}
    for idx, sym in enumerate(current_word):
        last_occ[sym] = idx
        
    for _ in range(N - 1):
        found = False
        # Get last n-1 symbols
        if len(current_word) >= n - 1:
            suffix = tuple(current_word[-(n-1):])
            
            # Determine LRU order
            symbols = list(range(1, n + 1))
            symbols.sort(key=lambda x: last_occ[x])
            
            for s in symbols:
                candidate_window = suffix + (s,)
                # Check if candidate_window is a valid permutation
                if candidate_window in perm_to_idx:
                    if candidate_window not in visited:
                        # Accept this move
                        current_word.append(s)
                        visited.add(candidate_window)
                        # Update last_occ for s
                        last_occ[s] = len(current_word) - 1
                        found = True
                        break
            
        if not found:
            # Fallback to greedy overlap
            for k in range(n - 1, 0, -1):
                suffix = tuple(current_word[-k:])
                candidates = prefix_to_indices.get(suffix, [])
                
                next_idx = -1
                for idx in candidates:
                    if tuple(perms[idx]) not in visited:
                        next_idx = idx
                        break
                
                if next_idx != -1:
                    next_perm = perms[next_idx]
                    current_word.extend(next_perm[k:])
                    visited.add(tuple(next_perm))
                    # Update last_occ for all new symbols
                    appended = next_perm[k:]
                    start_idx = len(current_word) - len(appended)
                    for offset, sym in enumerate(appended):
                        last_occ[sym] = start_idx + offset
                    found = True
                    break
                    
            if not found:
                # Should not happen, but pick any unvisited
                for i in range(N):
                    if tuple(perms[i]) not in visited:
                        current_word.extend(perms[i])
                        visited.add(tuple(perms[i]))
                        for offset, sym in enumerate(perms[i]):
                            last_occ[sym] = len(current_word) - n + offset
                        break
                        
    return "".join(map(str, current_word))
