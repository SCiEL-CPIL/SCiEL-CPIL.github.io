import time
import itertools
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if n != 7:
        return _standard_construct(n)
    
    # 1. Generate standard superpermutation for n=7
    # L_7 = 5913
    word = _standard_list(n)
    
    # 2. Precompute permutation counts
    # We only care about valid permutations (distinct symbols)
    perm_count = {}
    L = len(word)
    for i in range(L - n + 1):
        window = tuple(word[i:i+n])
        if len(set(window)) == n:
            perm_count[window] = perm_count.get(window, 0) + 1
            
    # 3. Greedy Deletion
    # We will do multiple passes.
    # In each pass, we scan from left to right.
    # Deleting at i shifts all subsequent indices.
    # So if we delete at i, we should restart the scan from i (which now contains the old i+1) or i-1.
    # Scanning left to right is generally safer for "shifted" logic if we handle indices carefully.
    # However, right to left is often preferred for in-place deletion because indices < i are unaffected.
    # Let's use Right-to-Left scan.
    
    max_passes = 100
    for p in range(max_passes):
        if time.time() > deadline - 5.0:
            break
            
        improved = False
        L = len(word)
        
        # Scan from right to left
        for i in range(L - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
                
            # Determine affected windows in current word
            # Windows start index k such that k <= i <= k+n-1
            # => i-n+1 <= k <= i
            start_k = max(0, i - n + 1)
            end_k = min(i, L - n)
            
            if start_k > end_k:
                continue
                
            # Collect old perms that appear ONLY in these windows (count == 1)
            # And store the old windows themselves to update counts later
            old_windows = []
            critical_perms = set()
            
            for k in range(start_k, end_k + 1):
                window = tuple(word[k:k+n])
                if len(set(window)) == n:
                    old_windows.append((k, window))
                    if perm_count[window] == 1:
                        critical_perms.add(window)
            
            if not critical_perms:
                # No critical perms, safe to delete
                # Update counts: decrement old windows
                for k, window in old_windows:
                    perm_count[window] -= 1
                    if perm_count[window] == 0:
                        del perm_count[window]
                
                # Perform deletion
                word.pop(i)
                L -= 1
                improved = True
                
                # Update counts: increment new windows
                # The new windows are at indices start_k to end_k (in new word)
                # Note: The set of start indices for affected windows remains the same range [start_k, end_k]
                # because the deletion happens at i, and windows start_k..end_k all "cross" i.
                # In the new word, the window starting at k (for k in start_k..end_k) is:
                # new_word[k:k+n]
                for k in range(start_k, end_k + 1):
                    new_win = tuple(word[k:k+n])
                    if len(set(new_win)) == n:
                        perm_count[new_win] = perm_count.get(new_win, 0) + 1
                        
                # Continue scanning leftwards (i-1)
                # The index i in the old word is gone.
                # The element now at i is old i+1.
                # We are iterating downwards, so next is i-1.
                # This is correct.
            else:
                # Check if critical perms are covered in new configuration
                # Simulate deletion
                # New segment: word[start_k:i] + word[i+1:end_k+n+1]
                # We need to check if all critical_perms appear as valid windows in this new segment.
                
                # Optimization: Instead of checking substring in a large string,
                # we can just check if the critical perm appears in the new local windows.
                # The new local windows are exactly those starting at k in [start_k, end_k].
                
                new_windows = []
                for k in range(start_k, end_k + 1):
                    # Construct new window
                    # word[k:i] is part 1
                    # word[i+1:k+n] is part 2 (if k < i)
                    # Actually, simpler:
                    # new_word = word[:i] + word[i+1:]
                    # But we don't want to create the full list.
                    # We can check specific windows.
                    
                    # Window k in new word:
                    # If k+n-1 < i: unchanged (but k is in affected range, so this shouldn't happen? 
                    # Wait. Affected range is defined by i.
                    # If k is in [start_k, end_k], then k <= i <= k+n-1.
                    # So the window ALWAYS includes i or crosses it.
                    # So the window content changes for all k in [start_k, end_k].
                    
                    # Let's construct the new window tuple.
                    # It consists of word[k:i] + word[i+1:k+n]
                    # Note: length of word[k:i] is i-k.
                    # Length of word[i+1:k+n] is (k+n) - (i+1) = k+n-i-1.
                    # Total length: i-k + k+n-i-1 = n-1.
                    # Wait, a window of length n needs n elements.
                    # The deleted element was at i.
                    # The elements are k..i-1 and i+1..k+n-1?
                    # No.
                    # Old window: k, k+1, ..., k+n-1.
                    # New window: k, ..., i-1, i+1, ..., k+n-1?
                    # No, the indices shift.
                    # New word index j corresponds to old j if j < i, and old j+1 if j >= i.
                    # So new window at k is:
                    # Elements at new indices k, k+1, ..., k+n-1.
                    # If k+n-1 < i: Old k..k+n-1 (Unchanged, but this case is excluded by affected range logic? 
                    # If k+n-1 < i, then i is not in the window. But our range [start_k, end_k] is derived from i being in the window.
                    # So for k in [start_k, end_k], the window MUST include index i in the old word.
                    # Thus, the new window will be formed by old elements k..i-1 and i+1..k+n.
                    # Wait, if the window ends at k+n-1 in old word, and i <= k+n-1.
                    # The new window takes elements from old indices:
                    # k .. i-1 (count: i-k)
                    # i+1 .. k+n (count: k+n - (i+1) + 1 = k+n-i)
                    # Total: i-k + k+n-i = n. Correct.
                    
                    new_win = tuple(word[k:i] + word[i+1:k+n+1])
                    new_windows.append(new_win)
                    
                    if len(set(new_win)) == n and new_win in critical_perms:
                        critical_perms.discard(new_win)
                        
                if not critical_perms:
                    # All critical perms are still present in the new local windows
                    # Update counts: decrement old windows
                    for k, window in old_windows:
                        perm_count[window] -= 1
                        if perm_count[window] == 0:
                            del perm_count[window]
                            
                    # Perform deletion
                    word.pop(i)
                    L -= 1
                    improved = True
                    
                    # Update counts: increment new windows
                    for k in range(start_k, end_k + 1):
                        new_win = tuple(word[k:k+n])
                        if len(set(new_win)) == n:
                            perm_count[new_win] = perm_count.get(new_win, 0) + 1
                else:
                    # Cannot delete
                    pass

        if not improved:
            break # No more deletions possible in this pass
            
    return "".join(map(str, word))

def _standard_construct(n):
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
    return "".join(map(str, word))

def _standard_list(n):
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
    return word
