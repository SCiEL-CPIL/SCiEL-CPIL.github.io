import time
import sys
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 1. Generate the classic recursive superpermutation
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
            # Find max overlap
            max_ov = min(len(new), len(block))
            overlap = 0
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Initial length
    current_len = len(word)
    
    # 2. Greedy Deletion to minimize length
    # We need to check if all permutations of 1..n are present.
    # Total perms = n!
    
    def get_permutation_index(p):
        # Map permutation tuple to a unique index 0..n!-1
        # This allows fast counting
        # Using factorial number system or just a dict lookup
        pass

    # For simplicity and speed, we use a Counter of tuples.
    # n=7, 5040 perms.
    
    def check_and_delete(word_list, idx):
        """
        Attempts to delete word_list[idx].
        Returns True if deletion was successful (all perms still covered).
        Modifies word_list and perm_counts in place if successful.
        """
        n_val = len(set(word_list)) # n
        # Actually n is fixed from the argument. Pass it or use global.
        
        # Identify affected windows
        # Windows starting at j such that j <= idx < j + n
        # j in [max(0, idx - n + 1), min(idx, len(word_list) - n)]
        
        start_j = max(0, idx - n + 1)
        end_j = min(idx, len(word_list) - n)
        
        if start_j > end_j:
            # No windows affected? Impossible if idx is valid and n>=1
            # But if n=1, window is just the char.
            # If n > len(word_list), no windows.
            return False

        # Store old counts for affected windows
        affected_perms = []
        for j in range(start_j, end_j + 1):
            w = tuple(word_list[j:j+n])
            affected_perms.append((j, w))
            
        # Decrement counts
        for j, w in affected_perms:
            perm_counts[w] -= 1
            if perm_counts[w] == 0:
                # Revert and return False
                for j2, w2 in affected_perms:
                    perm_counts[w2] += 1
                return False

        # Perform deletion
        del word_list[idx]
        
        # Update counts for the new windows in the affected region
        # The region in the new list is roughly the same indices
        # The deletion shifted indices >= idx to the left by 1.
        # The windows that exist now in the range [start_j, end_j] (adjusted)
        
        new_len = len(word_list)
        # The affected range in the new list:
        # The deletion was at idx.
        # Windows that *overlap* the deletion point in the new list are those
        # that contain the new element at idx (which was idx+1) or the element at idx-1.
        # Essentially, we need to re-evaluate windows starting from 
        # max(0, idx - n + 1) to min(idx, new_len - n)
        
        new_start_j = max(0, idx - n + 1)
        new_end_j = min(idx, new_len - n)
        
        # Note: The windows we decremented were from the OLD list.
        # We need to increment the counts for the NEW windows in this region.
        # However, some new windows might be identical to old windows?
        # No, the list changed.
        
        # Wait, the logic "Decrement old, Delete, Increment new" works if
        # we correctly identify the set of windows that changed.
        # The set of windows in the old list that contained the deleted char is S_old.
        # The set of windows in the new list that occupy the "same physical space" (around the gap) is S_new.
        # Actually, it's easier to just say:
        # The windows that are valid in the new list and start in [idx-n+1, idx] are the ones we need to count.
        
        # But we already decremented S_old.
        # We must increment S_new.
        
        for j in range(new_start_j, new_end_j + 1):
            w = tuple(word_list[j:j+n])
            perm_counts[w] += 1
            
        return True

    # Initialize perm_counts
    perm_counts = defaultdict(int)
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        perm_counts[w] += 1
        
    # Check if initial word is valid (it should be)
    # All n! perms must have count > 0
    
    # Greedy deletion loop
    # We do multiple passes.
    # Pass 1: Left to Right
    # Pass 2: Right to Left
    # Repeat until no changes.
    
    changed = True
    passes = 0
    max_passes = 50 # Safety
    
    while changed and passes < max_passes:
        changed = False
        passes += 1
        if time.time() > deadline:
            break
            
        # Left to Right
        i = 0
        while i < len(word):
            if time.time() > deadline:
                break
            if check_and_delete(word, i):
                changed = True
                # Do not increment i, as the next element is now at i
            else:
                i += 1
                
        if time.time() > deadline:
            break
            
        # Right to Left
        i = len(word) - 1
        while i >= 0:
            if time.time() > deadline:
                break
            if check_and_delete(word, i):
                changed = True
                # Do not decrement i, as the next element (to the left) is now at i-1?
                # If we delete at i, the element at i-1 is still at i-1.
                # The element that was at i+1 is now at i.
                # So we should check i again?
                # If we delete at i, the list shrinks.
                # The element now at i is the one that was at i+1.
                # So we should NOT decrement i. We check i again.
                # But if we keep checking i, we might delete multiple chars at the same index?
                # Yes, that's fine.
            else:
                i -= 1

    # Convert to string
    return "".join(map(str, word))

