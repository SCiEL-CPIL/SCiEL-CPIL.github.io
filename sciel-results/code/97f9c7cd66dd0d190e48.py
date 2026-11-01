import time
import random
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # 1. Recursive Construction
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
    
    # If n is small, we can just return the recursive result if it's already optimal or close
    # But for n=7, it's not.
    
    # Precompute all permutations?
    # For n=7, 5040 perms.
    # We track counts of all 5040 perms.
    
    perm_count = defaultdict(int)
    lost = set()
    
    # Initialize counts
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        perm_count[p] += 1
    
    # Identify lost (should be empty initially for valid superperm)
    # Generate all perms to check if any are missing?
    # The recursive construction is valid, so lost should be empty.
    # But let's be safe.
    import itertools
    all_perms = set(itertools.permutations(range(1, n+1)))
    for p in all_perms:
        if perm_count.get(p, 0) == 0:
            lost.add(p)
            
    # Helper to update counts for a window change
    # We will manually manage the counts during deletion attempts.
    
    def try_delete(i):
        """
        Attempts to delete word[i].
        Returns True if successful, False otherwise.
        Modifies word, perm_count, lost in-place if successful.
        """
        L = len(word)
        if L <= n:
            return False
            
        # Windows affected in OLD word: j such that j <= i <= j+n-1
        # j ranges from max(0, i-n+1) to min(i, L-n)
        start_j = max(0, i - n + 1)
        end_j = min(i, L - n)
        
        old_windows = []
        for j in range(start_j, end_j + 1):
            old_windows.append(tuple(word[j:j+n]))
            
        # Tentatively remove
        # We need to know the NEW windows that span the deletion
        # New word length L-1.
        # New windows spanning the gap:
        # In new word, the gap is at index i (conceptually between old i-1 and i+1).
        # New window indices j' such that the window covers the junction.
        # The characters are word[0..i-1] + word[i+1..L-1]
        # A window in the new word is "new" if it wasn't present in the old word at the same relative position?
        # No, we just need to count the permutations that EXIST in the new word.
        # The permutations that disappear are the old_windows.
        # The permutations that appear are the windows in the new word that cover the deletion site.
        
        # Let's compute the new windows that cover the deletion site.
        # In the new word, the deletion is at index i.
        # The windows covering index i in the new word are j' such that j' <= i <= j'+n-1.
        # j' ranges from max(0, i-n+1) to min(i, (L-1)-n)
        start_j_new = max(0, i - n + 1)
        end_j_new = min(i, L - 1 - n)
        
        new_windows = []
        for j in range(start_j_new, end_j_new + 1):
            # Construct window from word with i removed
            # word_new[j:j+n]
            # If j < i: word[j:j+n] but skip i?
            # Easier: create a slice
            # Since n is small (7), we can just build the tuple
            w_slice = word[:j] + word[j+1:] if j < i else word[:j] + word[j+1:] # This is wrong logic for slice
            
            # Correct way to get new window content:
            # If the window in new word starts at j:
            # It contains elements from word.
            # Elements are word[j..i-1] and word[i+1..j+n-1]
            # Let's just build it
            new_w = []
            for k in range(n):
                idx = j + k
                if idx < i:
                    new_w.append(word[idx])
                else:
                    new_w.append(word[idx+1])
            new_windows.append(tuple(new_w))

        # Decrement old windows
        valid = True
        for p in old_windows:
            perm_count[p] -= 1
            if perm_count[p] == 0:
                lost.add(p)
                
        # Increment new windows
        for p in new_windows:
            if perm_count.get(p, 0) == 0:
                # It was lost, now it's back
                if p in lost:
                    lost.remove(p)
            perm_count[p] += 1
            
        # Check validity
        if len(lost) > 0:
            # Revert
            for p in new_windows:
                perm_count[p] -= 1
                if perm_count[p] == 0:
                    lost.add(p)
            for p in old_windows:
                if perm_count.get(p, 0) == 0:
                    if p in lost:
                        lost.remove(p)
                perm_count[p] += 1
            return False
        else:
            # Commit
            word.pop(i)
            return True

    # Greedy Deletion Loop
    while time.time() < deadline:
        progress = False
        # Iterate backwards to avoid index shifting issues? 
        # If we delete i, indices > i shift left.
        # If we iterate backwards, indices < i are unaffected.
        L = len(word)
        for i in range(L - 1, -1, -1):
            if time.time() > deadline:
                break
            if try_delete(i):
                progress = True
        if not progress:
            break
            
    return "".join(map(str, word))
