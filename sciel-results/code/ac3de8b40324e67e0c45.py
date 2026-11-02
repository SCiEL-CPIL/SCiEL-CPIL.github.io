import time
import itertools
from collections import Counter

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # 1. Generate classic superpermutation
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
            # Overlap logic: find max overlap between end of new and start of block
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new)-o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    if n <= 4:
        return "".join(map(str, word))

    # Precompute all permutations of 1..n
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Initialize counts of each permutation in the current word
    # We only care about counts for permutations that are in all_perms
    # But for efficiency, we can just count all windows and check membership
    # Actually, storing counts for all 5040 perms is fine.
    perm_counts = {p: 0 for p in all_perms}
    
    # Initial count pass
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if w in perm_counts:
            perm_counts[w] += 1
    
    # Greedy deletion with efficient count updates
    # Single pass: scan left to right, delete if safe, continue from i (not restart)
    changed = True
    while changed and (time.time() < deadline):
        changed = False
        i = 0
        L = len(word)
        while i < L and (time.time() < deadline):
            # Determine which windows contain position i
            s_min = max(0, i - n + 1)
            s_max = min(i, L - n)
            
            # Check if any of these windows is a permutation with count 1
            safe = True
            for s in range(s_min, s_max + 1):
                w = tuple(word[s:s+n])
                if w in perm_counts and perm_counts[w] == 1:
                    safe = False
                    break
            
            if safe:
                # Decrement counts of windows containing position i
                for s in range(s_min, s_max + 1):
                    w = tuple(word[s:s+n])
                    if w in perm_counts:
                        perm_counts[w] -= 1
                
                # Remove the element
                word.pop(i)
                L = len(word)
                
                # Increment counts of new windows
                # Windows starting at s in [s_min, L-n] in the new word are affected
                for s in range(s_min, L - n + 1):
                    w = tuple(word[s:s+n])
                    if w in perm_counts:
                        perm_counts[w] += 1
                
                changed = True
                # Stay at position i since next element shifted in
            else:
                i += 1
    
    return "".join(map(str, word))
