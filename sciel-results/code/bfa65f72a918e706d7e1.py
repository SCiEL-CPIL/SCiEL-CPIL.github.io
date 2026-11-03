import math

def construct(n):
    if n == 1:
        return "1"
    
    # Initialize with the first permutation
    word = list(range(1, n + 1))
    seen = {tuple(word)}
    total_perms = math.factorial(n)
    
    # Track the last position each symbol was added
    # Using a dict for clarity, though a list of size n+1 works too
    last_seen = {i: i for i in range(1, n + 1)}
    current_pos = n
    
    # We need to find all n! permutations
    while len(seen) < total_perms:
        # The suffix of length n-1
        suffix = tuple(word[-(n - 1):])
        
        valid_candidates = []
        invalid_candidates = []
        
        for s in range(1, n + 1):
            new_window = suffix + (s,)
            if new_window in seen:
                invalid_candidates.append(s)
            else:
                valid_candidates.append(s)
        
        # Choose the least recently used symbol
        # Prefer valid (new) permutations
        if valid_candidates:
            # Pick the one with the smallest last_seen index
            next_sym = min(valid_candidates, key=lambda x: last_seen[x])
        else:
            # If no new permutations are possible, pick the least recently used among all
            next_sym = min(invalid_candidates, key=lambda x: last_seen[x])
        
        # Append the symbol
        word.append(next_sym)
        current_pos += 1
        last_seen[next_sym] = current_pos
        
        # Update seen set
        new_window = tuple(word[-n:])
        seen.add(new_window)
        
    return "".join(map(str, word))
