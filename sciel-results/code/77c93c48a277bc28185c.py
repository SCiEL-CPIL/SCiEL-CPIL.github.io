import time
import random
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin for 150s limit
    
    # 1. Generate standard recursive superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
        
    S = list(word)
    L = len(S)
    
    # Initialize counts
    counts = {}
    for i in range(L - n + 1):
        p = tuple(S[i:i+n])
        counts[p] = counts.get(p, 0) + 1
        
    best_S = S[:]
    best_len = L
    
    def get_affected_windows(S, i, n, removal_len=1):
        """
        Returns list of (old_windows, new_windows) for a removal of 'removal_len' chars starting at i.
        """
        if removal_len == 1:
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            old_windows = [tuple(S[j:j+n]) for j in range(start_old, end_old + 1)]
            
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, len(S) - 1 - n)
            new_windows = []
            for k in range(start_new, end_new + 1):
                w = []
                for idx in range(k, k + n):
                    if idx < i:
                        w.append(S[idx])
                    else:
                        w.append(S[idx + 1])
                new_windows.append(tuple(w))
            return old_windows, new_windows
        
        elif removal_len == 2:
            # Remove i and i+1
            start_old = max(0, i - n + 1)
            end_old = min(i + 1, len(S) - n)
            old_windows = [tuple(S[j:j+n]) for j in range(start_old, end_old + 1)]
            
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, len(S) - 2 - n)
            new_windows = []
            for k in range(start_new, end_new + 1):
                w = []
                for idx in range(k, k + n):
                    if idx < i - 1:
                        w.append(S[idx])
                    elif idx == i - 1:
                        w.append(S[i - 1])
                    else:
                        w.append(S[idx + 2])
                new_windows.append(tuple(w))
            return old_windows, new_windows
            
        elif removal_len == 3:
            # Remove i, i+1, i+2
            start_old = max(0, i - n + 1)
            end_old = min(i + 2, len(S) - n)
            old_windows = [tuple(S[j:j+n]) for j in range(start_old, end_old + 1)]
            
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, len(S) - 3 - n)
            new_windows = []
            for k in range(start_new, end_new + 1):
                w = []
                for idx in range(k, k + n):
                    if idx < i - 2:
                        w.append(S[idx])
                    elif idx == i - 2:
                        w.append(S[i - 2])
                    elif idx == i - 1:
                        w.append(S[i - 1])
                    else:
                        w.append(S[idx + 3])
                new_windows.append(tuple(w))
            return old_windows, new_windows
        else:
            return [], []

    def check_and_apply_removal(S, counts, i, removal_len):
        """
        Checks if removal is valid. If so, applies it to S and counts.
        Returns True if applied.
        """
        if i < 0 or i + removal_len > len(S):
            return False
            
        old_windows, new_windows = get_affected_windows(S, i, n, removal_len)
        
        # Check validity: Any old window with count 1 must be in new_windows
        new_set = set(new_windows)
        for p in old_windows:
            if counts.get(p, 0) == 1:
                if p not in new_set:
                    return False
        
        # Apply removal
        for p in old_windows:
            counts[p] -= 1
            if counts[p] == 0:
                del counts[p]
        
        for p in new_windows:
            counts[p] = counts.get(p, 0) + 1
            
        # Modify S
        del S[i:i+removal_len]
        
        return True

    # Main Search Loop
    last_progress_time = time.time()
    
    while time.time() < deadline:
        # 1. Greedy Removal Pass (Batch)
        # We try to remove as many as possible in one pass.
        # We iterate from end to start to avoid index shifting issues during deletion.
        # However, removals can enable other removals.
        # A single forward pass (0 to L) with index adjustment is tricky.
        # A single backward pass (L to 0) is safer for deletion.
        
        made_change = False
        
        # Try singles first
        indices = list(range(len(S)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            if i >= len(S):
                continue
            
            # We need to adjust index if previous deletions happened?
            # Since we are iterating a shuffled list, and deleting shifts indices,
            # it is safer to just try the index. If it's out of bounds, skip.
            # If a deletion happened at a lower index, the current index 'i' in the list
            # now points to S[i-1] (if one deletion before it).
            # This heuristic is imperfect but often works.
            
            # To be more robust, we can just restart the pass if a change is made.
            # But that is slow.
            # Let's try a "scan" approach:
            # Iterate i from 0 to L-1.
            # If we remove at i, the next index to check is i (which is now the old i+1).
            
        # Revised Strategy: Linear Scan with Step-Back
        i = 0
        while i < len(S):
            if time.time() > deadline:
                break
            
            # Try removing at i
            if check_and_apply_removal(S, counts, i, 1):
                made_change = True
                # Do not increment i, because S[i] is now the old S[i+1]
                # which might also be removable
                if len(S) < best_len:
                    best_len = len(S)
                    best_S = S[:]
                    last_progress_time = time.time()
            else:
                i += 1
                
        if made_change:
            continue
            
        # 2. Pair Removal Pass
        made_change = False
        i = 0
        while i < len(S) - 1:
            if time.time() > deadline:
                break
                
            if check_and_apply_removal(S, counts, i, 2):
                made_change = True
                if len(S) < best_len:
                    best_len = len(S)
                    best_S = S[:]
                    last_progress_time = time.time()
            else:
                i += 1
                
        if made_change:
            continue
            
        # 3. Triple Removal Pass
        made_change = False
        i = 0
        while i < len(S) - 2:
            if time.time() > deadline:
                break
                
            if check_and_apply_removal(S, counts, i, 3):
                made_change = True
                if len(S) < best_len:
                    best_len = len(S)
                    best_S = S[:]
                    last_progress_time = time.time()
            else:
                i += 1
                
        if made_change:
            continue
            
        # 4. Stuck. Try Perturbation (Swap)
        if len(S) > n:
            # Random adjacent swap
            i = random.randint(0, len(S) - 2)
            
            # Check swap validity
            # Old windows around i, i+1
            # New windows around i, i+1
            
            # We can simulate the swap
            S_temp = S[:]
            S_temp[i], S_temp[i+1] = S_temp[i+1], S_temp[i]
            
            # Calculate old and new windows
            # Affected range: [i-n+1, i+1]
            start = max(0, i - n + 1)
            end = min(i + 1, len(S) - n)
            
            old_windows = [tuple(S[j:j+n]) for j in range(start, end + 1)]
            new_windows = [tuple(S_temp[j:j+n]) for j in range(start, end + 1)]
            
            new_set = set(new_windows)
            valid = True
            for p in old_windows:
                if counts.get(p, 0) == 1:
                    if p not in new_set:
                        valid = False
                        break
                        
            if valid:
                # Apply swap
                for p in old_windows:
                    counts[p] -= 1
                    if counts[p] == 0:
                        del counts[p]
                for p in new_windows:
                    counts[p] = counts.get(p, 0) + 1
                    
                S[i], S[i+1] = S[i+1], S[i]
                made_change = True
                # Length is same, so best_len doesn't update, but we might enable removals
                last_progress_time = time.time() # Count as progress to avoid restart?
                # Actually, if we just swap, we should probably continue to removals.
        
        if not made_change:
            # Stuck completely.
            # If we haven't made progress for 5 seconds, restart from best_S
            if time.time() - last_progress_time > 5.0:
                S = best_S[:]
                # Rebuild counts
                counts = {}
                for i in range(len(S) - n + 1):
                    p = tuple(S[i:i+n])
                    counts[p] = counts.get(p, 0) + 1
                last_progress_time = time.time()
            else:
                # Just a bit stuck, try another random swap or break?
                # Given time limits, breaking might be safer if we are deep in.
                # But let's try a few more random swaps.
                pass
                
    return "".join(map(str, best_S))
