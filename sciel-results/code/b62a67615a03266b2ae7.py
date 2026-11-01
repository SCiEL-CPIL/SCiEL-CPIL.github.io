import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # 1. Standard recursive construction
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # 2. Local search: Greedy removal
    if n < 5: # For small n, standard is optimal
        return "".join(map(str, word))
        
    target_perms = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if len(set(w)) == n:
            target_perms.add(w)
            
    current_word = word
    current_len = len(current_word)
    
    # Helper to check if a word covers all perms
    def is_valid(w):
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        return seen == target_perms

    # Try to remove symbols
    # We will iterate multiple times, trying to remove symbols at random positions
    # If we remove a symbol, we must re-check validity.
    
    # Optimization: Instead of full re-check, we can update the set of seen perms.
    # But removing a symbol shifts indices, so many windows change.
    # Full re-check is O(L). L ~ 6000. 6000 * 7 = 42000 ops.
    # 1000 iterations = 42M ops. Python might be slow.
    # Let's limit iterations.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Randomize order to avoid bias
        indices = list(range(len(current_word)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            # Try removing current_word[i]
            new_word = current_word[:i] + current_word[i+1:]
            
            # Check validity
            if is_valid(new_word):
                current_word = new_word
                improved = True
                # Update target_perms? No, target_perms is fixed (all perms of 1..n)
                # But we only need to check if new_word covers ALL of them.
                # So target_perms is constant.
                
    return "".join(map(str, current_word))
