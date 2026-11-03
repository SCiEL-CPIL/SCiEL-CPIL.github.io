import math
import time
import sys

def is_superpermutation(word, n):
    if len(word) < n:
        return False
    perms = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i+n])
        # Check if it's a permutation of 1..n
        # For n=7, checking set(window) == set(range(1, n+1)) is O(n)
        # But we can just add to set and check size at end?
        # No, duplicates are allowed in windows, but must be valid perms.
        # A window is a valid perm iff len(set(window)) == n.
        if len(set(window)) == n:
            perms.add(window)
    return len(perms) == int(math.factorial(n))

def construct(n):
    # Classic construction
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
            # Find max overlap
            max_o = min(len(new), len(block))
            for o in range(max_o, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Local search: try to delete chars
    # To be safe, only do this if n is small enough or time allows
    if n <= 7:
        start_time = time.time()
        deadline = 110.0 # 10s margin
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try deleting from end to start
            for i in range(len(word) - 1, -1, -1):
                if time.time() > deadline:
                    break
                # Check if word without i is valid
                # Optimization: Only check if the char is "redundant"
                # Simple check:
                candidate = word[:i] + word[i+1:]
                if is_superpermutation(candidate, n):
                    word = candidate
                    improved = True
                    break # Restart scan from end as indices changed
    
    return "".join(map(str, word))
