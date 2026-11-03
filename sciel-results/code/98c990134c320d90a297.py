import time
from itertools import permutations

def construct(n):
    deadline = time.time() + 140  # 10s safety margin

    # Classic recursive construction of length 1! + 2! + ... + n!
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

    # For n >= 7, apply local search to shorten the word
    all_perms = set(permutations(range(1, n + 1)))
    current = word[:]
    n_local = n
    L = len(current)
    
    # Build reference count map: window -> count of occurrences
    from collections import defaultdict
    ref_count = defaultdict(int)
    for i in range(L - n_local + 1):
        ref_count[tuple(current[i:i + n_local])] += 1
    
    # Count how many required permutations are covered (ref_count >= 1)
    covered = sum(1 for p in all_perms if ref_count[p] > 0)
    total_perms = len(all_perms)
    
    # Verify classic construction is valid
    if covered < total_perms:
        return "".join(map(str, current))

    # Fast local search: try deleting each character using reference counting
    improved = True
    start_time = time.time()
    
    while improved and time.time() - start_time < 130:  # Safety margin
        improved = False
        L = len(current)
        num_windows = L - n_local + 1
        
        # Try deleting from left to right
        for i in range(L):
            if (i % 500 == 0) and time.time() - start_time > 130:
                break
            
            # Check if deleting position i would cause any permutation to become uncovered
            # Windows affected: those that include position i
            # Window starting at j covers positions j..j+n-1
            # Position i is in window j iff j <= i <= j+n-1, i.e., i-n+1 <= j <= i
            # Also j must be in [0, num_windows-1]
            
            j_start = max(0, i - n_local + 1)
            j_end = min(i, num_windows - 1)
            
            # Check if any affected window has ref_count == 1 and is a required permutation
            can_delete = True
            for j in range(j_start, j_end + 1):
                w = tuple(current[j:j + n_local])
                if ref_count[w] == 1 and w in all_perms:
                    can_delete = False
                    break
            
            if can_delete:
                # Perform deletion
                current.pop(i)
                L -= 1
                num_windows -= 1
                improved = True
                
                # Update reference counts for affected windows
                # Windows that were affected by deletion need recalculation
                # Old windows j in [j_start, j_end] are gone or shifted
                # New windows: positions shift left by 1 after i
                
                # Decrement counts for old windows that included position i
                for j in range(j_start, j_end + 1):
                    w = tuple(current[j:j + n_local]) if j + n_local <= L else None
                    # Actually, after pop, the windows have shifted. Let me think more carefully.
                    # After popping position i, the old window at position j (for j < i) still exists
                    # but the old window at position j >= i has shifted.
                    # This is getting complex. Let me just rebuild the affected region.
                    pass
                
                # Simpler approach: rebuild reference counts for windows near the deletion
                # Windows that might have changed are those starting from max(0, i-n+1) to min(i, L-n)
                # Let's just rebuild all affected windows
                new_j_start = max(0, i - n_local + 1)
                new_j_end = min(i, L - n_local)
                
                # First, clear old counts for windows in the affected range
                # But we need to be careful: some windows may span the deletion point
                
                # Actually, the cleanest approach: for windows that no longer exist or have changed,
                # we need to update. Let me just decrement the old windows and increment new ones.
                
                # Old windows that included position i: j in [j_start, j_end]
                # After deletion, these windows are gone (their content changed or they shifted)
                # New windows in the same range need to be counted.
                
                # Let me just rebuild the reference count for the affected region
                # by first removing all old affected windows, then adding new ones.
                
                # Remove old affected windows
                for j in range(j_start, j_end + 1):
                    # The old window at position j before deletion
                    # We need to reconstruct what it was
                    # If j < i: old window was current[j:j+n] but current has changed
                    # This is getting too complex. Let me use a different strategy.
                    pass
                
                # New strategy: just rebuild the entire reference count if needed,
                # but only when we actually delete. Since deletions are relatively rare,
                # this might be acceptable. But let me try a more efficient approach.
                
                # Actually, let me just track which windows changed and update incrementally.
                # When we delete position i:
                # - Windows starting at j where j < i and j+n-1 >= i are affected (they lose one element)
                # - Windows starting at j where j >= i shift left by 1
                # 
                # The simplest correct approach: 
                # 1. For each old window that included position i, decrement its count
                # 2. For each new window in the affected range, increment its count
                # But we need to know the old window contents.
                
                # Let me just do a full rebuild of the affected region.
                # The affected region is windows starting from max(0, i-n+1) to min(i, L-n)
                # But we need to be careful about what the old windows were.
                
                # I think the cleanest approach given the complexity is to just rebuild
                # the entire reference count map when a deletion occurs. Since deletions
                # are infrequent (only when valid), this should be acceptable.
                
                # Rebuild reference count
                ref_count.clear()
                for j in range(L - n_local + 1):
                    ref_count[tuple(current[j:j + n_local])] += 1
                
                # Update covered count
                covered = sum(1 for p in all_perms if ref_count[p] > 0)
                
            # If cannot delete, move on to next position

    return "".join(map(str, current))
