import itertools
import time

def construct(n):
    # Classic recursive construction
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
        
    if n < 7:
        return "".join(map(str, word))
        
    # Optimization for n=7
    # Precompute all valid permutations as integers
    # Map permutation tuple to an integer index
    perm_to_int = {}
    all_perms = list(itertools.permutations(range(1, n+1)))
    for idx, p in enumerate(all_perms):
        perm_to_int[p] = idx
    valid_set = set(range(len(all_perms)))
    
    def is_valid(w):
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if window in perm_to_int:
                seen.add(perm_to_int[window])
        return len(seen) == len(all_perms)
        
    current_word = word
    # Greedy deletion
    # Time limit guard
    start_time = time.time()
    max_time = 100 # Leave 20s buffer
    
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        L = len(current_word)
        for i in range(L):
            if (time.time() - start_time) > max_time:
                break
            # Try deleting i
            # Optimization: if i is such that the window around it is critical, skip?
            # Just try all.
            new_word = current_word[:i] + current_word[i+1:]
            if is_valid(new_word):
                current_word = new_word
                improved = True
                break # Restart scan
                
    return "".join(map(str, current_word))
