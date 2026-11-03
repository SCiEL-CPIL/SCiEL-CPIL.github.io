import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # 1. Generate standard superpermutation
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
            # Find max overlap
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n < 3:
        return "".join(map(str, word))

    # Optimization: Multi-start Greedy Deletion
    best_word = word
    best_len = len(word)
    
    # Helper to get non-essential indices
    def get_non_essential_indices(w):
        L = len(w)
        if L < n:
            return []
        
        perm_indices = {}
        for i in range(L - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                if window in perm_indices:
                    perm_indices[window].append(i)
                else:
                    perm_indices[window] = [i]
                    
        essential_indices = [False] * L
        for perm, indices in perm_indices.items():
            if len(indices) == 1:
                idx = indices[0]
                for k in range(n):
                    essential_indices[idx + k] = True
        
        non_essential = [i for i in range(L) if not essential_indices[i]]
        return non_essential

    # Run multiple restarts
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        current_word = word[:]
        while True:
            non_ess = get_non_essential_indices(current_word)
            if not non_ess:
                break
            
            # Strategy: Pick a random non-essential index to escape local minima
            # Or pick one that seems "central" or something? Random is good for diversity.
            # To speed up convergence, we can also try picking the one that is 
            # part of the fewest total windows? 
            # For now, random selection from non-essential.
            idx_to_remove = random.choice(non_ess)
            
            current_word = current_word[:idx_to_remove] + current_word[idx_to_remove+1:]
            
            if len(current_word) < best_len:
                best_len = len(current_word)
                best_word = current_word[:]
                
            # Check time inside inner loop as well? 
            # The inner loop runs at most ~10-20 times. It's fast.
            # But if we get stuck, we break.
        
        # If we found a better word, we might want to try to optimize it further?
        # The current design restarts from the original. 
        # We could also restart from best_word found so far?
        # Let's stick to restarting from original for simplicity and diversity, 
        # but if we have time, we can also try to polish best_word.
        
        # Actually, restarting from original is safer against getting stuck in a bad basin.
        # But if we found a good one, maybe we can try to improve it further?
        # The "essential" check is local. If we are stuck, we are stuck.
        # So restarting from original with different random seeds is the way.

    return "".join(map(str, best_word))
