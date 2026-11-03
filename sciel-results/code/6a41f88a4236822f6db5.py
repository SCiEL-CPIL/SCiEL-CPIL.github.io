"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import permutations

# EVOLVE-BLOCK-START
def _build_recursive_superperm(n):
    """Classic recursive construction of length 1! + 2! + ... + n!."""
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
    return word

def construct(n):
    start_time = time.time()
    time_limit = 110.0  # seconds, leaving 10s safety margin
    
    # 1. Build standard superpermutation
    word = _build_recursive_superperm(n)
    
    if n > 8:
        # For large n, optimization is too slow, return standard
        return "".join(map(str, word))
    
    # 2. Setup permutation counting
    # Generate all permutations and map to index
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts
    # counts[i] = number of times permutation i appears in word
    counts = [0] * num_perms
    
    # Initial counts
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        idx = perm_to_idx.get(window)
        if idx is not None:
            counts[idx] += 1
    
    # 3. Greedy Pruning Loop
    # We repeatedly try to remove characters.
    # To be efficient, we scan the word and try to remove each char.
    # If a removal is valid, we do it and restart the scan from the beginning
    # (or continue, but restart is simpler and ensures we don't miss things
    # due to index shifts, though we can be smarter).
    # Actually, since removal shifts indices, restarting is safer.
    # But we can also just iterate and if we remove, we restart the outer loop.
    
    current_word = word
    while True:
        if time.time() - start_time > time_limit:
            break
            
        progress = False
        # Try to remove each position
        # We iterate from left to right.
        # If we remove one, we restart.
        for pos in range(len(current_word)):
            if time.time() - start_time > time_limit:
                break
            
            # Check if removal is valid
            # We need to know if any count will become 0.
            # We simulate the removal by checking the windows that would be lost.
            # If any of those windows is the *only* occurrence of that permutation,
            # removal is invalid.
            
            start_min = max(0, pos - n + 1)
            start_max = min(pos, len(current_word) - n)
            
            valid = True
            lost_indices = []
            
            for i in range(start_min, start_max + 1):
                window = tuple(current_word[i:i + n])
                idx = perm_to_idx.get(window)
                if idx is not None:
                    if counts[idx] == 1:
                        valid = False
                        break
                    else:
                        lost_indices.append(idx)
            
            if valid:
                # Perform removal
                for idx in lost_indices:
                    counts[idx] -= 1
                
                # Update word
                current_word = current_word[:pos] + current_word[pos+1:]
                progress = True
                break # Restart scan
        
        if not progress:
            break
            
    return "".join(map(str, current_word))
# EVOLVE-BLOCK-END
