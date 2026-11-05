"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations

def construct(n):
    # Classic recursive construction of length 1! + 2! + ... + n!: each permutation P of the
    # previous word, in order of appearance, becomes P m P, overlapped as far as possible.
    # This is the standard, well-understood construction that guarantees validity.
    # For n=7, length = 1! + 2! + 3! + 4! + 5! + 6! + 7! = 1+2+6+24+120+720+5040 = 5913
    # 
    # We use this as the base. The task asks for the shortest superpermutation.
    # The known best is 5905 (Teodorescu 2026). The classic construction gives 5913.
    # 
    # We can try to improve upon the classic construction by finding better orderings
    # of permutations at each step, or by using a different construction method.
    #
    # For now, return the classic construction which is valid and fast.
    
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
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
