"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from itertools import permutations

def construct(n):
    # Standard recursive construction
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
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    s = "".join(map(str, word))
    
    # Local optimization: try removing each symbol and check validity
    # Using fast string operations
    if n <= 7:
        # Pre-compute all permutations as strings
        all_perms = set("".join(map(str, p)) for p in permutations(range(1, n + 1)))
        target_count = math.factorial(n)
        
        def is_valid(ss):
            if len(ss) < n:
                return False
            # Check if the set of all windows is exactly all_perms
            # Using a set comprehension is fast
            windows = set(ss[i:i+n] for i in range(len(ss) - n + 1))
            return windows == all_perms
        
        improved = True
        # Limit iterations to prevent timeout if stuck in a loop of small gains
        # but greedy removal usually converges fast
        max_iter = 100 
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            # Try removing each character
            # We iterate from 0 to len-2
            for pos in range(len(s) - 1):
                trial = s[:pos] + s[pos+1:]
                if is_valid(trial):
                    s = trial
                    improved = True
                    break
    return s
# EVOLVE-BLOCK-END
