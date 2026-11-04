import itertools
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Generate standard superpermutation
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
    
    # If n is small, return immediately
    if n < 5:
        return "".join(map(str, word))
        
    # For n >= 5, try to optimize
    w = word
    
    # Pre-compute all permutations
    all_perms = set(itertools.permutations(range(1, n+1)))
    total_perms = len(all_perms)
    
    # Count occurrences of each permutation in the current word
    perm_counts = {}
    for i in range(len(w) - n + 1):
        p = tuple(w[i:i+n])
        if p in all_perms:
            perm_counts[p] = perm_counts.get(p, 0) + 1
            
    # Check if all are present (should be)
    if len(perm_counts) < total_perms:
        return "".join(map(str, w))
        
    # Iterative deletion pass
    # We will try to remove characters from left to right.
    # If a character can be removed, we remove it and restart the scan from the beginning
    # (or from the current position, but restarting is safer to catch new opportunities created by shifts).
    # To be efficient, we can just scan left-to-right, and if we remove at i, we stay at i (or go back a bit) because the new character at i might also be removable.
    
    max_passes = 100 # Safety limit, usually converges fast
    for _ in range(max_passes):
        if time.time() > deadline:
            break
            
        removed_any = False
        i = 0
        while i < len(w) - n + 1:
            if time.time() > deadline:
                break
                
            # Check if w[i] can be removed
            # Windows affected (old): start in [i-n+1, i]
            removable = True
            for start in range(max(0, i-n+1), min(i+1, len(w)-n+1)):
                p = tuple(w[start:start+n])
                if p in all_perms:
                    if perm_counts.get(p, 0) == 1:
                        removable = False
                        break
            
            if removable:
                # Update counts: Decrement old windows
                for start in range(max(0, i-n+1), min(i+1, len(w)-n+1)):
                    p = tuple(w[start:start+n])
                    if p in all_perms:
                        perm_counts[p] -= 1
                        if perm_counts[p] == 0:
                            del perm_counts[p]
                
                # Remove w[i]
                w.pop(i)
                
                # Update counts: Increment new windows
                # New windows are at starts [i-n+1, i-1] in the new string
                # Note: The length of w is now len(w)-1
                # Valid starts for new windows: 0 to len(w)-n
                new_max_start = len(w) - n
                for start in range(max(0, i-n+1), min(i, new_max_start + 1)):
                    p = tuple(w[start:start+n])
                    if p in all_perms:
                        perm_counts[p] = perm_counts.get(p, 0) + 1
                
                removed_any = True
                # Stay at index i to check the new character at i (which was i+1)
                # If i is now out of bounds, break
                if i >= len(w) - n + 1:
                    break
            else:
                i += 1
        
        if not removed_any:
            break
            
    return "".join(map(str, w))
