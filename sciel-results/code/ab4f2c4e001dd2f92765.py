import time
import itertools

def construct(n):
    if n == 1:
        return "1"
    
    # Generate standard superpermutation
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
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    if n <= 6:
        return "".join(map(str, word))
        
    # Pruning heuristic for n >= 7
    # We try to remove symbols greedily.
    # Time limit guard
    start_time = time.time()
    max_time = 100.0 # Leave 20s buffer
    
    # Precompute initial permutation counts
    # For n=7, 5040 perms.
    # Word length ~ 6000.
    perm_counts = {}
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        perm_counts[p] = perm_counts.get(p, 0) + 1
        
    # List of all permutations to check against? 
    # Actually, we just need to know if a count drops to 0.
    # But we also need to ensure we don't lose a permutation that is NOT in the word?
    # The standard word contains ALL permutations.
    # So if we remove a symbol, we only risk losing a permutation if its count becomes 0.
    
    # Greedy removal
    # We iterate through indices. If a removal is safe, we do it and restart from the beginning 
    # (or continue, but restart ensures we don't miss opportunities created by previous removals).
    # To be efficient, we can just scan left to right, and if a removal is safe, do it and reset index to max(0, i-1) 
    # because the character at i-1 might now be removable.
    
    i = 0
    while i < len(word):
        if time.time() - start_time > max_time:
            break
            
        # Check if word[i] can be removed
        # Affected old windows: start j in [i-n+1, i]
        min_j = max(0, i - n + 1)
        max_j = min(i, len(word) - n)
        
        safe = True
        affected_perms = []
        
        for j in range(min_j, max_j + 1):
            p = tuple(word[j:j+n])
            if perm_counts[p] <= 1:
                safe = False
                break
            affected_perms.append(p)
            
        if safe:
            # Perform removal
            # Decrement counts for affected_perms
            for p in affected_perms:
                perm_counts[p] -= 1
                
            # Remove word[i]
            word.pop(i)
            
            # Update counts for new windows
            # New windows are those that span the junction i-1 and i (in new word)
            # In new word, the junction is at index i (old i+1 is now at i).
            # The characters involved are new_word[i-1] (old i-1) and new_word[i] (old i+1).
            # Windows starting at j in new word that include both or are newly formed.
            # Actually, any window in new_word that was not in old_word must cross the gap.
            # The gap is between index i-1 and i in new_word.
            # So windows starting at j such that j <= i-1 and j+n-1 >= i.
            # i.e., i-n+1 <= j <= i-1.
            
            new_min_j = max(0, i - n + 1)
            new_max_j = min(i - 1, len(word) - n)
            
            for j in range(new_min_j, new_max_j + 1):
                p_new = tuple(word[j:j+n])
                perm_counts[p_new] = perm_counts.get(p_new, 0) + 1
                
            # Move back one step to re-check i-1
            i = max(0, i - 1)
        else:
            i += 1
            
    return "".join(map(str, word))
