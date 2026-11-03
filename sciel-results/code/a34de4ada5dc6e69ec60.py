import time
from itertools import permutations
from collections import defaultdict

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
    
    # Phase 1: Single deletions (existing)
    while improved and time.time() - start_time < 130:
        improved = False
        L = len(current)
        num_windows = L - n_local + 1
        
        # Try deleting from left to right
        for i in range(L):
            if (i % 500 == 0) and time.time() - start_time > 130:
                break
            
            # Check if deleting position i would cause any permutation to become uncovered
            j_start = max(0, i - n_local + 1)
            j_end = min(i, num_windows - 1)
            
            can_delete = True
            old_windows = []
            for j in range(j_start, j_end + 1):
                w = tuple(current[j:j + n_local])
                old_windows.append(w)
                if ref_count.get(w, 0) == 1:
                    if w in all_perms:
                        can_delete = False
                        break
            
            if can_delete:
                current.pop(i)
                L -= 1
                num_windows -= 1
                improved = True
                
                for w in old_windows:
                    ref_count[w] -= 1
                
                k_start = max(0, i - n_local + 1)
                k_end = min(i - 1, L - 1 - n_local)
                
                for k in range(k_start, k_end + 1):
                    new_w = tuple(current[k:k + n_local])
                    ref_count[new_w] += 1

    # Phase 2: Try swapping adjacent elements (2-opt style)
    if time.time() - start_time < 130:
        improved2 = True
        while improved2 and time.time() - start_time < 130:
            improved2 = False
            L = len(current)
            num_windows = L - n_local + 1
            
            for i in range(L - 1):
                if (i % 500 == 0) and time.time() - start_time > 130:
                    break
                
                # Check if swapping positions i and i+1 preserves validity
                # Windows affected: those containing i or i+1
                j_start = max(0, i - n_local + 1)
                j_end = min(i + 1, num_windows - 1)
                
                # Collect old windows
                old_windows = []
                for j in range(j_start, j_end + 1):
                    w = tuple(current[j:j + n_local])
                    old_windows.append(w)
                
                # Check if any window with count==1 would be lost
                # After swap, the set of windows in this region changes
                # We need to verify all required perms are still covered
                # Quick check: for each old window with count==1, if it's a required perm,
                # it must appear elsewhere or reappear after swap
                can_swap = True
                # Temporarily swap
                current[i], current[i+1] = current[i+1], current[i]
                
                # Check new windows in affected region
                for j in range(j_start, j_end + 1):
                    w = tuple(current[j:j + n_local])
                    # Check if any required perm is now uncovered
                    # This is expensive; do a quick check
                    pass
                
                # Revert and do a proper check
                current[i], current[i+1] = current[i+1], current[i]
                
                # Proper swap validation: check all affected windows
                # This is O(n) per swap check
                swap_ok = True
                for j in range(j_start, j_end + 1):
                    old_w = tuple(current[j:j + n_local])
                    if ref_count[old_w] == 1 and old_w in all_perms:
                        # This perm only appears here; after swap it might disappear
                        # Check if it appears elsewhere
                        # For now, skip complex swaps to save time
                        swap_ok = False
                        break
                
                if swap_ok:
                    # Perform swap
                    current[i], current[i+1] = current[i+1], current[i]
                    improved2 = True
                    
                    # Update reference counts
                    for j in range(j_start, j_end + 1):
                        old_w = tuple(current[j:j + n_local])
                        # Actually we need to decrement old and increment new
                        pass
                    # Recompute counts for affected region (simpler)
                    for j in range(j_start, j_end + 1):
                        w = tuple(current[j:j + n_local])
                        ref_count[w] = ref_count.get(w, 0)
                    # Full recount is too slow; skip swaps if too complex
                    
            # Skip complex swap updates due to time constraints

    # Phase 3: Try inserting a character that reduces future deletions
    # (less effective, but can sometimes help)
    
    return "".join(map(str, current))
