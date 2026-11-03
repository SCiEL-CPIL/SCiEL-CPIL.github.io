import time
from math import factorial
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Generate Classic Superpermutation
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
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new)-o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    target_count = factorial(n)
    
    # Precompute all permutations for lookup? 
    # For n=7, 5040 perms. We can map perm to index 0..5039.
    # This allows using a list for freq instead of dict/set of tuples, which is faster.
    all_perms = list(permutations(range(1, n+1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    
    def get_perm_idx(w, i):
        # w is list, i is start index
        # Return index if valid perm, else -1
        p = tuple(w[i:i+n])
        # Check if it's a permutation (distinct elements)
        # Since w contains 1..n, we can just check if it's in perm_to_idx
        # But tuple creation is expensive.
        # Alternative: encode as integer?
        # For n=7, base-8 or similar.
        # Or just use the tuple as key in a dict.
        # Let's stick to tuple for simplicity first, optimize if needed.
        return perm_to_idx.get(p, -1)

    # Initialize freq
    freq = [0] * target_count
    covered = 0
    
    for i in range(len(word) - n + 1):
        idx = get_perm_idx(word, i)
        if idx != -1:
            if freq[idx] == 0:
                covered += 1
            freq[idx] += 1
            
    # Local Search: Greedy Deletion
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(word)):
            if time.time() > deadline:
                break
            
            # Try deleting word[i]
            # Old crossing windows: start at j in [i-n+1, i]
            # But only those that are valid perms (within bounds)
            # Window j is valid if 0 <= j <= len(word)-n
            
            # Save old state to rollback if invalid
            old_freq_changes = []
            old_covered = covered
            
            # Remove old crossing windows
            for j in range(max(0, i - n + 1), min(i + 1, len(word) - n + 1)):
                idx = get_perm_idx(word, j)
                if idx != -1:
                    freq[idx] -= 1
                    if freq[idx] == 0:
                        covered -= 1
                    old_freq_changes.append((idx, -1))
            
            # Add new crossing windows
            # In new word (length L-1), window starts at j' in [i-n+1, i-1]
            # j' corresponds to old word indices:
            # w'[j':j'+n] = w[j':i] + w[i+1:j'+n+1]
            # Note: j' < i.
            # The start index in old word for the first part is j'.
            # The end index in old word for the second part is j'+n+1.
            # We need to form the tuple from word.
            
            valid_new_crossings = []
            for j_prime in range(max(0, i - n + 1), min(i, len(word) - n)): # j' < i and j'+n < len(word)
                # Construct new window
                # w[j_prime:i] + w[i+1:j_prime+n+1]
                # Note: j_prime+n+1 <= len(word)
                new_win = tuple(word[j_prime:i] + word[i+1:j_prime+n+1])
                idx = perm_to_idx.get(new_win, -1)
                if idx != -1:
                    if freq[idx] == 0:
                        covered += 1
                    freq[idx] += 1
                    valid_new_crossings.append((idx, 1))
            
            if covered == target_count:
                # Accept deletion
                word = word[:i] + word[i+1:]
                improved = True
                break
            else:
                # Rollback
                for idx, delta in old_freq_changes:
                    freq[idx] += delta
                for idx, delta in valid_new_crossings:
                    freq[idx] += delta
                covered = old_covered
                
    return "".join(map(str, word))
