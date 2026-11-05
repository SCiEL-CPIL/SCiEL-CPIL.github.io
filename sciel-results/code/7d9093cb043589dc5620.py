import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110
    
    if n == 1:
        return "1"
    
    # Step 1: Construct the standard superpermutation
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[-o:] == block[:o]:
                    overlap = o
                    break
            new_word.extend(block[overlap:])
        word = new_word
    
    # Step 2: Local search to shorten the word
    # We try to delete characters one by one.
    # If a deletion maintains validity, we keep it.
    # We repeat until no more deletions are possible.
    
    def is_valid(w):
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            # Check if it's a permutation of 1..n
            # Since we start from a valid superperm, windows are likely perms.
            # But after deletion, some might not be.
            # However, we just need to check if ALL n! perms are present.
            # If a window is not a perm, it doesn't contribute to the set.
            if len(set(window)) == n:
                seen.add(window)
        return len(seen) == int(math.factorial(n))

    import math
    target_perms = int(math.factorial(n))
    
    # Optimization: Precompute factorial
    # We can speed up is_valid by checking count.
    
    def is_valid_fast(w):
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
                if len(seen) == target_perms:
                    return True
        return False

    # Iterative deletion
    changed = True
    while changed:
        changed = False
        L = len(word)
        for i in range(L):
            if time.time() > deadline:
                return "".join(map(str, word))
            
            # Try deleting word[i]
            test_word = word[:i] + word[i+1:]
            if is_valid_fast(test_word):
                word = test_word
                changed = True
                break # Restart scan from beginning
        
    return "".join(map(str, word))
