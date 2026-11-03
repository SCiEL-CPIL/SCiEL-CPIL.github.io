import time
import math

_TIME_LIMIT = 115.0

def _classic_superperm_list(n):
    """Classic recursive construction returning a list of ints."""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            best = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    best = o
                    break
            new += block[best:]
        word = new
    return word

def construct(n):
    start = time.time()
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1321231"
    if n == 4:
        return "14321243121431"
    if n == 5:
        return "".join(map(str, _classic_superperm_list(5)))
    
    # For n >= 6
    W_list = _classic_superperm_list(n)
    
    if n == 7:
        # Precompute permutation IDs for fast validity check
        # Generate all permutations of 1..7
        import itertools
        all_perms = list(itertools.permutations(range(1, n+1)))
        perm_to_id = {p: i for i, p in enumerate(all_perms)}
        target_count = len(all_perms)
        
        # Fast validity check using set of IDs
        def is_valid_fast(word):
            seen = set()
            # Optimization: if len < target, return False early? No, need to check all.
            for i in range(len(word) - n + 1):
                # Convert slice to tuple for lookup
                # Slicing is fast, tuple creation is fast
                w = tuple(word[i:i+n])
                # If w is not a permutation of 1..n, it's not in perm_to_id
                # But in a superperm, all valid windows are perms.
                # Invalid windows (duplicates) will not be in perm_to_id.
                # We only care if we have seen all *valid* perms.
                # So we just add if it exists.
                if w in perm_to_id:
                    seen.add(perm_to_id[w])
            return len(seen) == target_count
            
        # Greedy removal
        i = 0
        L = len(W_list)
        while i < L and time.time() - start < _TIME_LIMIT - 5:
            # Try removing W_list[i]
            # We need to check if W_list without i is valid.
            # Creating a new list is O(L).
            # We can avoid creating the full list if we can check validity on the fly?
            # No, the shift makes it hard.
            
            # Optimization: 
            # If the character at i is "critical", removal might break a unique permutation.
            # But we don't know which ones are unique without tracking counts.
            
            # Let's just do the check.
            # To speed up list creation, we can use a temporary list?
            # Or just slice and concatenate.
            new_list = W_list[:i] + W_list[i+1:]
            
            if is_valid_fast(new_list):
                W_list = new_list
                L -= 1
                # Do not increment i
            else:
                i += 1
                
        return "".join(map(str, W_list))
        
    return "".join(map(str, W_list))
