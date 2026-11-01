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
    # We iterate through positions. If a deletion is valid, we do it and restart the scan
    # or continue? Restarting is safer for correctness but slower. 
    # Given the O(n) check, we can try to be smarter.
    # Let's do a single pass first, then repeat until no changes or time limit.
    
    changed = True
    while changed and (time.time() < deadline):
        changed = False
        # We scan from left to right
        for i in range(len(word)):
            if time.time() > deadline:
                break
            
            # Determine which windows are affected by removing position i
            # Windows starting at s include position i if s <= i <= s+n-1
            # => i-n+1 <= s <= i
            # Also s must be valid: 0 <= s <= len(word) - n
            
            s_min = max(0, i - n + 1)
            s_max = min(i, len(word) - n)
            
            # Check if any of these windows is a permutation with count 1
            safe = True
            # We need to know the windows to check
            # Note: if we delete i, these windows disappear.
            # If any of them had count 1, it becomes 0 -> invalid.
            for s in range(s_min, s_max + 1):
                w = tuple(word[s:s+n])
                if w in perm_counts and perm_counts[w] == 1:
                    safe = False
                    break
            
            if safe:
                # Perform deletion
                # 1. Decrement counts of windows that are removed
                #    The windows removed are those starting in [s_min, s_max]
                #    BUT, deleting word[i] shifts the rest of the word left.
                #    The windows that *existed* and are *removed* are those that contained i.
                #    These are exactly the windows starting in [s_min, s_max].
                for s in range(s_min, s_max + 1):
                    w = tuple(word[s:s+n])
                    if w in perm_counts:
                        perm_counts[w] -= 1
                
                # 2. Remove the element
                word.pop(i)
                
                # 3. Increment counts of new windows formed by the shift
                #    The shift affects windows that *now* cover position i (which is the new word[i], formerly word[i+1])
                #    Actually, it's easier to think about which windows are *new*.
                #    Before deletion, windows were at indices s.
                #    After deletion, windows at indices s < i are unchanged.
                #    Windows at indices s >= i shift left.
                #    Specifically, the window that was at s+1 is now at s.
                #    The "new" windows are those that did not exist before at those positions.
                #    Actually, the set of windows changes.
                #    Let's identify the range of indices in the *new* word that are affected.
                #    The deletion happens at index i.
                #    New word length L' = L - 1.
                #    Windows starting at s in new word:
                #    If s < i: same as old word.
                #    If s >= i: this window is formed by elements from old word starting at s, 
                #    except the element at i is gone, so it's old word[s:i] + old word[i+1:s+n+1].
                #    Wait, the window at new index s corresponds to old indices s...s+n-1, 
                #    but with a gap at i.
                #    Essentially, the windows that are *newly formed* (i.e., their content changed or they are new positions)
                #    are those starting at s in [i, min(i+n-1, L'-n)]?
                #    Let's just re-evaluate the windows in the range [i, i+n-1] in the new word.
                #    These are the only ones that could have changed content or position relative to the count map.
                #    Actually, windows starting at s < i are identical to before.
                #    Windows starting at s >= i in the new word are formed by old word[s:i] + old word[i+1:s+n+1].
                #    We need to increment the count for these new windows.
                
                # Range of new windows to add:
                # New word indices s from i to min(i + n - 1, len(word) - n)
                # Note: len(word) is now L-1.
                new_s_min = i
                new_s_max = min(i + n - 1, len(word) - n)
                
                for s in range(new_s_min, new_s_max + 1):
                    w = tuple(word[s:s+n])
                    if w in perm_counts:
                        perm_counts[w] += 1
                
                changed = True
                break # Restart scan as indices changed
    
    return "".join(map(str, word))
