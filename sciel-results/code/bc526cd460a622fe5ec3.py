"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations

def _factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def _classic_superperm(n):
    """Classic recursive superpermutation construction."""
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _build_perm_index(word, n):
    """Build index of all permutation occurrences in word."""
    n_fact = _factorial(n)
    # Map each permutation to its list of starting positions
    perm_positions = {}
    for j in range(len(word) - n + 1):
        window = tuple(word[j:j+n])
        if len(set(window)) == n:
            if window not in perm_positions:
                perm_positions[window] = []
            perm_positions[window].append(j)
    return perm_positions

def _try_shorten(word, n, deadline):
    """Shorten superpermutation via greedy deletion with critical tracking."""
    word = list(word)
    target_perms = _factorial(n)
    
    # Build initial occurrence map
    perm_positions = _build_perm_index(word, n)
    
    if len(perm_positions) < target_perms:
        return word
    
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        L = len(word)
        
        # Find positions that are "safe" to remove: positions where no 
        # permutation has its only occurrence starting at or overlapping that position
        # Actually, we need: removing position i is safe iff for every permutation,
        # it still has at least one occurrence in the new word.
        
        # Efficient approach: try removing from end to start, check validity incrementally
        # For each candidate removal, we can check if any permutation loses its last occurrence
        
        # Simpler: just try removals and do a fast validity check
        for i in range(L - 1, -1, -1):
            if time.time() > deadline - 15:
                break
            if len(word) <= n:
                break
            
            # Try removing position i
            new_word = word[:i] + word[i+1:]
            
            # Fast check: count distinct permutations in new_word
            seen = set()
            for j in range(len(new_word) - n + 1):
                window = tuple(new_word[j:j+n])
                if len(set(window)) == n:
                    seen.add(window)
            
            if len(seen) == target_perms:
                word = new_word
                improved = True
                # Rebuild perm_positions for next iteration
                perm_positions = _build_perm_index(word, n)
                break
    
    return word

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Generate classic superpermutation
    word = _classic_superperm(n)
    
    if n <= 6:
        return "".join(map(str, word))
    
    # For n=7, try to shorten
    if time.time() < deadline - 20:
        word = _try_shorten(word, n, deadline)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
