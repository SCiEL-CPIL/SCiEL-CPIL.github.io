"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import permutations as _perms
from collections import Counter

# EVOLVE-BLOCK-START
def construct(n):
    if n <= 1:
        return "1" if n == 1 else ""
    if n == 2:
        return "121"
    
    # Start with classic construction
    word = _classic_construct(n)
    L = len(word)
    
    # Precompute valid permutations
    valid_perms = set(_perms(range(1, n+1)))
    
    deadline = time.time() + 110.0
    
    removed = [False] * L
    perm_count = Counter()
    
    # Count all windows
    for i in range(L - n + 1):
        perm_count[tuple(word[i:i+n])] += 1
    
    # Multiple passes to remove positions
    for pass_num in range(20):
        if time.time() > deadline:
            break
        
        progress = False
        # Try removing positions from right to left
        for pos in range(L - 1, -1, -1):
            if time.time() > deadline:
                break
            if removed[pos]:
                continue
            
            # Check if all PERMUTATION windows containing pos have count > 1
            valid = True
            min_i = max(0, pos - n + 1)
            max_i = min(pos, L - n)
            for wi in range(min_i, max_i + 1):
                w = tuple(word[wi:wi+n])
                if w in valid_perms:
                    if perm_count[w] <= 1:
                        valid = False
                        break
            
            if valid:
                removed[pos] = True
                progress = True
                # Update counts for windows containing pos
                for wi in range(min_i, max_i + 1):
                    w = tuple(word[wi:wi+n])
                    if w in valid_perms:
                        perm_count[w] -= 1
    
        if not progress:
            break
    
    # Build final word
    final_word = [word[i] for i in range(L) if not removed[i]]
    return "".join(map(str, final_word))


def _classic_construct(n):
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
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word
# EVOLVE-BLOCK-END
