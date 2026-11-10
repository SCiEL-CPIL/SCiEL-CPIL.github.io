import time
import math
from collections import Counter

def construct(n):
    start_time = time.time()
    deadline = start_time + 110
    
    # Step 1: Generate classic superpermutation
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
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    if n <= 5:
        return "".join(map(str, word))
        
    # Step 2: Greedy Deletion
    target_perms = math.factorial(n)
    
    counts = Counter()
    seen_set = set()
    for i in range(len(word) - n + 1):
        perm = tuple(word[i:i+n])
        counts[perm] += 1
        seen_set.add(perm)
        
    L = len(word)
    
    while time.time() < deadline:
        improved = False
        # We scan from left to right.
        # If we delete, we break and restart the scan from 0.
        # This is O(L^2) in the worst case if many deletions happen early.
        # But deletions reduce L, so it's manageable.
        
        for i in range(L):
            if time.time() > deadline:
                break
                
            # Calculate remove and add windows
            # Remove: k in [max(0, i-n+1), i]
            # Add: k in [max(0, i-n+1), i-1]
            
            valid = True
            # We need to check if any perm in R has count 1 and is not in A.
            # We can build the set A first.
            add_perms = set()
            for k in range(max(0, i - n + 1), i):
                if k + n <= L:
                    # Construct window
                    # word[k:i] + word[i+1:k+n]
                    # Note: indices in old word
                    w1 = word[k:i]
                    w2 = word[i+1:k+n]
                    if len(w1) + len(w2) == n:
                        p = tuple(w1 + w2)
                        add_perms.add(p)
            
            for k in range(max(0, i - n + 1), i + 1):
                if k + n <= L:
                    p = tuple(word[k:k+n])
                    if counts[p] == 1 and p not in add_perms:
                        valid = False
                        break
            
            if valid:
                # Perform deletion
                # Update counts
                # Remove windows
                for k in range(max(0, i - n + 1), i + 1):
                    if k + n <= L:
                        p = tuple(word[k:k+n])
                        counts[p] -= 1
                        if counts[p] == 0:
                            seen_set.discard(p)
                            
                # Add windows
                for k in range(max(0, i - n + 1), i):
                    if k + n <= L:
                        w1 = word[k:i]
                        w2 = word[i+1:k+n]
                        if len(w1) + len(w2) == n:
                            p = tuple(w1 + w2)
                            counts[p] += 1
                            seen_set.add(p)
                            
                # Delete
                word = word[:i] + word[i+1:]
                L -= 1
                improved = True
                break # Restart scan
                
        if not improved:
            break
            
    return "".join(map(str, word))
