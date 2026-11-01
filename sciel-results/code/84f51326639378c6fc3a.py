"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import permutations

# EVOLVE-BLOCK-START
def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Classic recursive construction of length 1! + 2! + ... + n!: each permutation P of the
    # previous word, in order of appearance, becomes P m P, overlapped as far as possible.
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
        
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
