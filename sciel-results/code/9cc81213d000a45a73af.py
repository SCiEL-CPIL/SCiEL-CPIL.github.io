"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations

def construct(n):
    deadline = time.time() + 140  # 10s safety margin

    # Classic recursive construction of length 1! + 2! + ... + n!
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
            best_overlap = 0
            max_check = min(len(new), len(block))
            for o in range(max_check, 0, -1):
                if new[-o:] == block[:o]:
                    best_overlap = o
                    break
            new += block[best_overlap:]
        word = new

    if n <= 6:
        return "".join(map(str, word))

    # For n >= 7, apply local search to shorten the word
    all_perms = set(permutations(range(1, n + 1)))

    def is_valid(w):
        seen = set()
        for i in range(len(w) - n + 1):
            seen.add(tuple(w[i:i + n]))
        return all_perms.issubset(seen)

    # Verify classic construction is valid
    if not is_valid(word):
        return "".join(map(str, word))

    current = word[:]

    # Try removing each character from left to right
    for i in range(len(current)):
        if time.time() > deadline:
            break

        saved = current.pop(i)
        if is_valid(current):
            continue
        else:
            current.insert(i, saved)

    return "".join(map(str, current))
# EVOLVE-BLOCK-END
