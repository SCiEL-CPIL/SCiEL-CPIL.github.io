import sys
import time

def construct(n):
    # Generate initial superpermutation
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
    
    # Greedy deletion
    # Map permutations to IDs for fast counting
    # We only care about permutations of 1..n
    # Precompute all valid permutations? 5040 is small.
    # But we can just use a dict of tuples.
    
    L = len(word)
    # Initial counts
    counts = {}
    for i in range(L - n + 1):
        w = tuple(word[i:i+n])
        if len(set(w)) == n:
            counts[w] = counts.get(w, 0) + 1
        else:
            # Should not happen in valid superpermutation construction
            pass
            
    # Check if all n! permutations are present
    # We assume the initial word is valid.
    
    changed = True
    while changed:
        changed = False
        # Scan from right to left
        for i in range(L - 1, -1, -1):
            # Determine which windows contain index i
            # Window j contains i if j <= i < j+n => i-n+1 <= j <= i
            # Also 0 <= j <= L-n
            start_j = max(0, i - n + 1)
            end_j = min(i, L - n)
            
            valid = True
            affected_windows = []
            
            for j in range(start_j, end_j + 1):
                w = tuple(word[j:j+n])
                # We only track valid permutations
                if len(set(w)) == n:
                    affected_windows.append(w)
                    # Check if count would become 0
                    if counts.get(w, 0) == 1:
                        valid = False
                        break
            
            if valid:
                # Remove word[i]
                # Update counts for affected windows
                for w in affected_windows:
                    counts[w] -= 1
                    if counts[w] == 0:
                        del counts[w]
                
                # Remove from list
                word.pop(i)
                L -= 1
                changed = True
    
    return "".join(map(str, word))
