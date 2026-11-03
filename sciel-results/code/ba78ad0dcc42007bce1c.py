import itertools

def construct(n):
    if n == 1:
        return "1"
    
    # Generate all permutations and sort them lexicographically
    # Sorting helps with deterministic tie-breaking
    perms = sorted(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Map from prefix to list of indices of permutations starting with that prefix
    # We need to quickly find candidates for a given prefix
    # Since n is small (7), we can just filter, but for efficiency we can use a dict
    prefix_to_indices = {}
    for i, p in enumerate(perms):
        for k in range(1, n):
            prefix = p[:k]
            if prefix not in prefix_to_indices:
                prefix_to_indices[prefix] = []
            prefix_to_indices[prefix].append(i)
    
    visited = [False] * N
    current_idx = 0
    visited[0] = True
    
    # Start with the first permutation
    current_word = list(perms[0])
    
    for _ in range(N - 1):
        found = False
        # Try max overlap first
        for k in range(n - 1, 0, -1):
            suffix = tuple(current_word[-k:])
            candidates = prefix_to_indices.get(suffix, [])
            
            # Find the unvisited candidate that appends the smallest symbol (heuristic improvement)
            # If tie, pick lexicographically smallest permutation
            best_idx = -1
            best_next_symbol = n + 1
            best_perm = None
            for idx in candidates:
                if not visited[idx]:
                    next_perm = perms[idx]
                    next_symbol = next_perm[k]
                    if next_symbol < best_next_symbol or (next_symbol == best_next_symbol and (best_perm is None or next_perm < best_perm)):
                        best_next_symbol = next_symbol
                        best_idx = idx
                        best_perm = next_perm
            
            if best_idx != -1:
                visited[best_idx] = True
                next_perm = best_perm
                # Append the new symbols
                current_word.extend(next_perm[k:])
                current_idx = best_idx
                found = True
                break
        
        if not found:
            # No overlap > 0, pick the lexicographically smallest unvisited permutation
            for i in range(N):
                if not visited[i]:
                    visited[i] = True
                    next_perm = perms[i]
                    current_word.extend(next_perm)
                    current_idx = i
                    found = True
                    break
                    
    return "".join(map(str, current_word))
