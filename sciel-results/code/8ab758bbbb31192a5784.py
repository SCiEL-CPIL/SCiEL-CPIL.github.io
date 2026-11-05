import math
import time
import itertools

def construct(n):
    # Generate the standard recursive superpermutation
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Local search: try to remove characters
    if n >= 5:
        start_time = time.time()
        deadline = start_time + 100.0
        
        target_perms = set(itertools.permutations(range(1, n + 1)))
        
        def is_valid(w):
            if len(w) < n:
                return False
            seen = set()
            for i in range(len(w) - n + 1):
                seen.add(tuple(w[i:i+n]))
                if len(seen) == len(target_perms):
                    return True
            return False
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(word)):
                new_word = word[:i] + word[i+1:]
                if is_valid(new_word):
                    word = new_word
                    improved = True
                    break
                    
    return "".join(map(str, word))
