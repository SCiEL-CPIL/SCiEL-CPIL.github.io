"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct(n):
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # For n <= 7, use the classic recursive construction which is fast enough
    # and produces a valid superpermutation.
    # For n = 7, the classic construction gives length 1!+2!+...+7! = 5916
    # The shortest known is 5905, but we can't easily find that in time.
    # For now, return the classic construction which is valid.
    
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
