import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 140

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
            best_overlap = 0
            max_check = min(len(new), len(block))
            for o in range(max_check, 0, -1):
                if new[-o:] == block[:o]:
                    best_overlap = o
                    break
            new += block[best_overlap:]
        word = new

    if n <= 6:
        return "".join(map(str, word))

    # 2. Setup for local search
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts
    counts = [0] * num_perms
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        if p in perm_to_id:
            counts[perm_to_id[p]] += 1
            
    # Track number of permutations with zero count
    zero_count = sum(1 for c in counts if c == 0)
    
    # Check if valid initially
    if zero_count > 0:
        # Should not happen for standard construction
        return "".join(map(str, word))

    # Local Search: Deletion
    # We will repeatedly try to delete characters
    # To avoid O(N^2) complexity, we use the fast count update
    
    def try_delete(i):
        """
        Attempts to delete word[i]. Returns True if successful.
        Modifies word and counts if successful.
        """
        nonlocal zero_count
        L = len(word)
        if L == n: # Cannot delete if only one window
            return False
        
        # Affected windows in old word: start k in [i-n+1, i]
        # We need to decrement counts for these
        # And increment counts for new windows formed after deletion
        
        # Track changes to zero_count
        # 1. Decrement old windows
        for k in range(max(0, i - n + 1), min(i + 1, L - n + 1)):
            p = tuple(word[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                if counts[pid] == 1:
                    zero_count += 1
                counts[pid] -= 1
        
        # 2. Simulate deletion to find new windows
        # Affected range in new word: start k in [max(0, i-n+1), min(i, L-n)]
        
        for k in range(max(0, i - n + 1), min(i, L - n)): # k is start index in NEW word
            if k < i:
                w1 = word[k:i]
                w2 = word[i+1:k+n+1]
                p = tuple(w1 + w2)
            else: # k >= i
                p = tuple(word[k+1:k+1+n])
                
            pid = perm_to_id.get(p)
            if pid is not None:
                if counts[pid] == 0:
                    zero_count -= 1
                counts[pid] += 1
                
        # Check if all counts are >= 1
        if zero_count == 0:
            # Perform deletion
            word.pop(i)
            return True
        else:
            # Revert: we need to undo the count changes
            # Re-increment old windows
            for k in range(max(0, i - n + 1), min(i + 1, L - n + 1)):
                p = tuple(word[k:k+n])
                pid = perm_to_id.get(p)
                if pid is not None:
                    if counts[pid] == 0:
                        zero_count -= 1
                    counts[pid] += 1
            
            # Re-decrement new windows
            for k in range(max(0, i - n + 1), min(i, L - n)):
                if k < i:
                    w1 = word[k:i]
                    w2 = word[i+1:k+n+1]
                    p = tuple(w1 + w2)
                else:
                    p = tuple(word[k+1:k+1+n])
                    
                pid = perm_to_id.get(p)
                if pid is not None:
                    if counts[pid] == 1:
                        zero_count += 1
                    counts[pid] -= 1
            
            return False

    # Iterate deletion passes
    improved = True
    while improved:
        improved = False
        # Left to right
        i = 0
        while i < len(word):
            if time.time() > deadline:
                return "".join(map(str, word))
            if try_delete(i):
                improved = True
                # Do not increment i, check same position again (new character)
            else:
                i += 1
        # Right to left
        i = len(word) - 1
        while i >= 0:
            if time.time() > deadline:
                return "".join(map(str, word))
            if try_delete(i):
                improved = True
            else:
                i -= 1

    # 3. Local Search: Swap
    # Try swapping adjacent characters if it reduces length (by enabling further deletions?)
    # No, swap doesn't change length directly.
    # But swap might enable deletions that were previously blocked.
    # Let's do a pass of swaps and then try deletions again.
    
    def try_swap(i):
        """
        Attempts to swap word[i] and word[i+1].
        Returns True if successful and it enables at least one deletion.
        """
        nonlocal zero_count
        L = len(word)
        if i >= L - 1:
            return False
            
        # Check validity of swap
        # Affected windows: start k in [i-n+1, i+1]
        
        # Decrement old windows
        for k in range(max(0, i - n + 1), min(i + 2, L - n + 1)):
            p = tuple(word[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                if counts[pid] == 1:
                    zero_count += 1
                counts[pid] -= 1
        
        # Perform swap
        word[i], word[i+1] = word[i+1], word[i]
        
        # Increment new windows
        for k in range(max(0, i - n + 1), min(i + 2, L - n + 1)):
            p = tuple(word[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                if counts[pid] == 0:
                    zero_count -= 1
                counts[pid] += 1
        
        # Check validity
        if zero_count > 0:
            # Revert
            word[i], word[i+1] = word[i+1], word[i]
            # Re-increment old windows
            for k in range(max(0, i - n + 1), min(i + 2, L - n + 1)):
                p = tuple(word[k:k+n])
                pid = perm_to_id.get(p)
                if pid is not None:
                    if counts[pid] == 0:
                        zero_count -= 1
                    counts[pid] += 1
            # Re-decrement new windows
            for k in range(max(0, i - n + 1), min(i + 2, L - n + 1)):
                p = tuple(word[k:k+n])
                pid = perm_to_id.get(p)
                if pid is not None:
                    if counts[pid] == 1:
                        zero_count += 1
                    counts[pid] -= 1
            return False
        
        # If valid, try to delete at i and i+1
        saved_word = word.copy()
        saved_zero = zero_count
        
        if try_delete(i):
            return True
        else:
            # State is back to post-swap (try_delete reverted itself)
            if try_delete(i+1):
                return True
            
            # If no deletion, revert swap
            word[:] = saved_word
            # Revert counts: need to undo the swap's count changes
            # First, undo the new windows from post-swap state
            for k in range(max(0, i - n + 1), min(i + 2, L - n + 1)):
                p = tuple(word[k:k+n])
                pid = perm_to_id.get(p)
                if pid is not None:
                    if counts[pid] == 1:
                        zero_count += 1
                    counts[pid] -= 1
            # Then redo the old windows
            word[i], word[i+1] = word[i+1], word[i]
            for k in range(max(0, i - n + 1), min(i + 2, L - n + 1)):
                p = tuple(word[k:k+n])
                pid = perm_to_id.get(p)
                if pid is not None:
                    if counts[pid] == 0:
                        zero_count -= 1
                    counts[pid] += 1
            return False

    # Iterate swap passes
    improved = True
    while improved:
        improved = False
        i = 0
        while i < len(word) - 1:
            if time.time() > deadline:
                return "".join(map(str, word))
            if try_swap(i):
                improved = True
                # Don't increment i, as word changed
            else:
                i += 1

    return "".join(map(str, word))
