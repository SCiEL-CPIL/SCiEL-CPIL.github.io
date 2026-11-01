"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct(n):
    start_time = time.time()
    deadline = start_time + 145  # 145s deadline, 5s margin
    
    if n <= 1:
        return "1"
    
    # For small n, we can precompute optimal superpermutations
    # The classic construction is optimal for n <= 5
    # For n=6, classic gives 873 which is optimal
    # For n=7, classic gives 5913, optimal is 5905
    
    # Build up from n=1
    word = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
        
        # Extract all (m-1)-permutations from the current word
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build new word by inserting m into each permutation
        # Standard: P m P for each P, with maximal overlap
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap with end of new
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        
        word = new
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
