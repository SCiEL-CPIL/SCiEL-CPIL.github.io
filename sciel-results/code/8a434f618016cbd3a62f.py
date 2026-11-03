import time
import random
from itertools import permutations

def construct(n):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    
    # For n <= 6, standard is optimal
    if n <= 6:
        return "".join(map(str, word))
    
    # For n > 6, try to shorten via local search
    # Convert to list of ints for manipulation
    current_word = word
    current_len = len(current_word)
    
    # Pre-compute required permutations
    required_perms = set(permutations(range(1, n+1)))
    
    def is_valid(w):
        # Check if all required perms are in w
        # w is a list of ints
        L = len(w)
        if L < n:
            return False
        seen = set()
        for i in range(L - n + 1):
            seen.add(tuple(w[i:i+n]))
        # Optimization: if we see all required, we can stop early?
        # But we need to ensure NO extra perms? No, extra perms are fine.
        # We just need required_perms subset of seen.
        # And len(seen) must be at least n!
        # Actually, if len(seen) < n!, it's invalid.
        if len(seen) < len(required_perms):
            return False
        return required_perms.issubset(seen)

    # Initial check
    if not is_valid(current_word):
        return "".join(map(str, word)) # Should not happen

    best_word = current_word
    best_len = current_len
    
    # Local Search: Random removal
    # To speed up is_valid, we can maintain a set of windows
    # But for simplicity, we re-calculate.
    # 5913 length, 5040 perms.
    # is_valid takes ~0.005s?
    # We can do ~20000 checks in 100s.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        if len(current_word) <= 1:
            break
            
        # Pick a random index to remove
        # Avoid removing first or last? 
        # Removing first/last reduces length but might break perms.
        # Let's allow any index.
        i = random.randint(0, len(current_word) - 1)
        
        # Simulate removal
        # Create new list
        # To avoid copying the whole list every time (expensive), 
        # we can check validity on the fly or copy.
        # Copying 6000 ints is fast (~0.001s).
        candidate = current_word[:i] + current_word[i+1:]
        
        if is_valid(candidate):
            current_word = candidate
            best_word = candidate
            best_len = len(candidate)
            # If we found a shorter one, we can continue searching from here
        else:
            # Optional: try swapping? 
            # Let's stick to removal for now.
            pass

    return "".join(map(str, best_word))
