import time
import random

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Step 1: Generate standard superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract permutations of length m-1 from word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Overlap
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    if n < 5:
        return "".join(map(str, word))
        
    # Step 2: Greedy Deletion
    # Initialize counts
    perm_counts = {}
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        perm_counts[p] = perm_counts.get(p, 0) + 1
        
    # Helper to check validity and update
    def try_remove(i):
        nonlocal word, perm_counts
        L = len(word)
        if L <= n:
            return False
            
        # Identify removed windows
        removed_perms = []
        min_start = max(0, i - n + 1)
        max_start = min(i, L - n)
        for s in range(min_start, max_start + 1):
            p = tuple(word[s:s+n])
            removed_perms.append(p)
            
        # Identify added windows
        added_perms = []
        min_s_new = max(0, i - n + 1)
        max_s_new = min(i - 1, L - 1 - n)
        
        for s_new in range(min_s_new, max_s_new + 1):
            p_list = []
            for k in range(s_new, s_new+n):
                if k < i:
                    p_list.append(word[k])
                else:
                    p_list.append(word[k+1])
            p = tuple(p_list)
            added_perms.append(p)
            
        # Apply changes
        for p in removed_perms:
            perm_counts[p] -= 1
        for p in added_perms:
            perm_counts[p] = perm_counts.get(p, 0) + 1
            
        # Check validity
        valid = True
        for p in removed_perms:
            if perm_counts[p] < 1:
                valid = False
                break
        # Note: added_perms counts only increase, so they won't become < 1 unless they were 0 before?
        # No, if they were 0, they become 1. If they were 1, they become 2.
        # So we only need to check removed_perms.
        
        if valid:
            # Perform removal
            word = word[:i] + word[i+1:]
            return True
        else:
            # Revert
            for p in removed_perms:
                perm_counts[p] += 1
            for p in added_perms:
                perm_counts[p] -= 1
            return False

    # Run deletion passes
    # Time limit guard
    start_time = time.time()
    max_time = 140 # 10s margin
    
    # We do multiple passes. In each pass, we scan from left to right.
    # If we remove something, we restart the scan from the beginning (or current index).
    # Restarting from beginning is safer but slower.
    # Let's try scanning left to right, and if a removal happens, we continue from the same index (which is now the next element).
    # This is a standard greedy approach.
    
    while time.time() - start_time < max_time:
        progress = False
        i = 0
        while i < len(word) - n + 1:
            if try_remove(i):
                progress = True
                # Do not increment i, because the next element shifted into position i
            else:
                i += 1
                
        if not progress:
            break
            
    return "".join(map(str, word))
