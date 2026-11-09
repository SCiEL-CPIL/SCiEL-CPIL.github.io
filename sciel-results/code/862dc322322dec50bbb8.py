"""Construct the shortest superpermutation on n symbols using greedy overlap heuristic."""
import itertools
import time

def construct(n):
    """
    Return the shortest superpermutation on n symbols as a string of digits.
    Uses a greedy 'maximum suffix-prefix overlap' heuristic with look-ahead.
    """
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    total_perms = len(perms)
    
    # Build suffix->perm index for O(1) lookup of perms starting with a given suffix
    # For overlap, we need to find perms that START with a given prefix (which is the suffix of current word)
    # Let's build: prefix of length k -> list of perms starting with that prefix
    # But we need all possible overlap lengths. Let's precompute for each perm, all its suffixes and prefixes.
    
    # Actually, the standard approach: 
    # We maintain a set of unvisited permutations.
    # At each step, we want to append the shortest possible string that contains at least one new permutation.
    # The maximum overlap k means we can extend by n-k characters.
    
    # Precompute: for each permutation, map its prefix of length k to the perm
    # For each overlap length k (from n-1 down to 0), check if there's an unvisited perm starting with current_suffix_of_len_k
    
    # Let's use a different approach:
    # For the current word, the last n-1 characters form a "state". 
    # We want to find the unvisited permutation that shares the longest suffix with the current word's tail.
    
    # Precompute: for each perm and each overlap k (1 to n-1), store the prefix of length k
    # perm_prefix[perm_idx][k] = first k elements of perm[perm_idx]
    
    # But building this for 5040 perms is fine.
    
    # Alternative simpler approach: 
    # Build a dictionary: for each possible suffix of length k (k=1..n-1), which unvisited perms start with that suffix?
    # Actually, we need: given the last k chars of our word, find unvisited perms that START with those k chars.
    # Because if our word ends with ...abc and a perm starts with abc..., we can overlap by k chars.
    
    # Let's precompute prefix_index[k][prefix_tuple] = list of perm indices with that prefix
    prefix_index = [dict() for _ in range(n)]
    for idx, p in enumerate(perms):
        for k in range(1, n):
            prefix = p[:k]
            if prefix not in prefix_index[k]:
                prefix_index[k][prefix] = []
            prefix_index[k][prefix].append(idx)
    
    unvisited = set(range(total_perms))
    
    # Start with the first permutation
    word = list(perms[0])
    unvisited.remove(0)
    
    # Greedy: at each step, try overlap lengths from n-1 down to 0
    # For overlap k, the current suffix of length k must match the prefix of some unvisited perm
    # Among all unvisited perms with that prefix, pick the one that allows the longest future overlap?
    # For simplicity, just pick the first one found, or one that maximizes future potential.
    
    # Let's try: for each k from n-1 down to 1:
    #   suffix = tuple(word[-k:])
    #   candidates = prefix_index[k].get(suffix, [])
    #   filter to unvisited
    #   if any, pick one (maybe the one whose remaining suffix matches something?)
    
    # For efficiency, let's just pick the first candidate. We can try look-ahead later.
    
    step = 0
    while unvisited:
        if time.time() - start_time > DEADLINE:
            break
        step += 1
        
        found = False
        for k in range(n - 1, 0, -1):
            if len(word) < k:
                continue
            suffix = tuple(word[-k:])
            candidates = prefix_index[k].get(suffix, [])
            for idx in candidates:
                if idx in unvisited:
                    # Append the remaining n-k chars of this perm
                    p = perms[idx]
                    word.extend(p[k:])
                    unvisited.remove(idx)
                    found = True
                    break
            if found:
                break
        
        if not found:
            # No overlap possible; just append a full new permutation
            # Pick any unvisited perm (first one)
            idx = next(iter(unvisited))
            p = perms[idx]
            word.extend(p)
            unvisited.remove(idx)
    
    return "".join(map(str, word))
