import time
import random

def construct():
    n = 128
    start = time.time()
    deadline = start + 80

    def is_valid_with(S, x):
        """Check if adding x to set S keeps the set valid (no forbidden triple)."""
        S_new = S | {x}
        elems = list(S)
        for i in range(len(elems)):
            a = elems[i]
            for j in range(i + 1, len(elems)):
                b = elems[j]
                # Triple {x, a, b} must have distinct elements
                if x == a or x == b:
                    continue
                s1 = x + a
                s2 = x + b
                s3 = a + b
                if s1 in S_new and s2 in S_new and s3 in S_new:
                    return False
        return True

    best = set()

    # Greedy from high to low (guarantees at least the seed solution size)
    S = set()
    for x in range(n, 0, -1):
        if is_valid_with(S, x):
            S.add(x)
    if len(S) > len(best):
        best = S.copy()

    # Greedy from low to high
    S = set()
    for x in range(1, n + 1):
        if is_valid_with(S, x):
            S.add(x)
    if len(S) > len(best):
        best = S.copy()

    # Randomized greedy restarts to find larger valid sets
    while time.time() < deadline:
        S = set()
        order = list(range(1, n + 1))
        random.shuffle(order)
        for x in order:
            if is_valid_with(S, x):
                S.add(x)
        if len(S) > len(best):
            best = S.copy()

    return sorted(best)
