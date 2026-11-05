import time
import itertools
import random

def construct(n):
    """
    Construct the shortest known superpermutation for n symbols.
    Uses recursive construction followed by greedy deletion with count-based validity check.
    """
    start_time = time.time()
    deadline = 110.0  # 10 seconds margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # 1. Generate classic recursive superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            target_overlap = m - 2
            overlap = 0
            if len(new_word) >= target_overlap and len(block) >= target_overlap:
                if new_word[-target_overlap:] == block[:target_overlap]:
                    overlap = target_overlap
                else:
                    max_overlap = min(len(new_word), len(block))
                    for o in range(max_overlap, -1, -1):
                        if o == 0:
                            break
                        if new_word[len(new_word) - o:] == block[:o]:
                            overlap = o
                            break
            else:
                max_overlap = min(len(new_word), len(block))
                for o in range(max_overlap, -1, -1):
                    if o == 0:
                        break
                    if new_word[len(new_word) - o:] == block[:o]:
                        overlap = o
                        break
            
            new_word.extend(block[overlap:])
        
        word = new_word
        
    current_word = word[:]
    
    # Pre-compute all permutations
    all_perms = list(itertools.permutations(range(1, n+1)))
    perm_set = set(all_perms)
    
    # Initialize counts
    # Count occurrences of each permutation in current_word
    counts = {p: 0 for p in perm_set}
    
    def update_counts_add(w, start_idx, end_idx):
        """Add counts for windows starting in [start_idx, end_idx]"""
        for i in range(start_idx, min(end_idx, len(w) - n + 1)):
            p = tuple(w[i:i+n])
            if p in counts:
                counts[p] += 1
                
    def update_counts_remove(w, start_idx, end_idx):
        """Remove counts for windows starting in [start_idx, end_idx]"""
        for i in range(start_idx, min(end_idx, len(w) - n + 1)):
            p = tuple(w[i:i+n])
            if p in counts:
                counts[p] -= 1

    # Initial count
    # Windows start from 0 to L-n
    update_counts_add(current_word, 0, len(current_word) - n + 1)
    
    # Check initial validity (should be true)
    # for p in perm_set:
    #    if counts[p] == 0: return "Invalid Initial"
        
    # Greedy deletion
    # We maintain the list of counts.
    # When we delete index i, we remove windows starting at i-n+1 to i.
    # Specifically, windows $W_{i-n+1}, \dots, W_i$ (if they exist).
    # Note: A window $W_j$ is defined by $w[j:j+n]$.
    # It contains $w[i]$ if $j \le i \le j+n-1$.
    # So $j$ ranges from $i-n+1$ to $i$.
    
    # After deletion, the length decreases by 1.
    # The indices of windows after $i$ shift left by 1.
    # However, we don't need to update the *values* of the counts for windows that shift,
    # because the set of windows *starting* at specific indices changes,
    # but we are maintaining counts of *permutations*, not indices.
    # The only counts that change are those for the windows that are *removed* by the deletion.
    # The windows that shift left are still the same permutations, just at different indices.
    # So we only need to decrement the counts for the $n$ windows that are destroyed.
    
    passes = 0
    max_passes = 500
    
    while passes < max_passes:
        if time.time() > deadline:
            break
            
        improved = False
        L = len(current_word)
        last_valid_start = L - n  # last index j where w[j:j+n] is a valid window
        
        # Iterate backwards to handle index shifts
        for i in range(L - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Determine which windows are affected
            # Windows starting at j such that j <= i <= j+n-1
            # j in [i-n+1, i]
            start_j = max(0, i - n + 1)
            end_j = min(i, last_valid_start)
            
            # Quick pre-check: if any affected window has count == 1, deletion is impossible
            # This avoids the expensive decrement/increment cycle
            can_delete = True
            for j in range(start_j, end_j + 1):
                p = tuple(current_word[j:j+n])
                if counts.get(p, 0) <= 1:
                    can_delete = False
                    break
            
            if not can_delete:
                continue
            
            # Deletion is possible: perform it and update counts
            affected_perms = []
            for j in range(start_j, end_j + 1):
                p = tuple(current_word[j:j+n])
                affected_perms.append(p)
                counts[p] -= 1
            
            current_word.pop(i)
            improved = True
            last_valid_start -= 1  # length decreased by 1
            
        passes += 1
        if not improved:
            break
            
    return "".join(map(str, current_word))
