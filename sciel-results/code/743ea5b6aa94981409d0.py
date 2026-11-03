import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill

    # 1. Generate standard superpermutation using careful overlap
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
    
    L = len(word)
    num_windows = L - n + 1
    window_ids = [0] * num_windows
    counts = [0] * num_perms
    
    for i in range(num_windows):
        p = tuple(word[i:i+n])
        pid = perm_to_id.get(p, -1)
        window_ids[i] = pid
        if pid >= 0:
            counts[pid] += 1
            
    zero_count = sum(1 for c in counts if c == 0)
    
    if zero_count > 0:
        return "".join(map(str, word))

    # Helper: check if time is up
    def time_up():
        return time.time() > deadline

    # Local Search: Deletion
    def try_delete(i):
        """
        Attempts to delete word[i]. Returns True if successful.
        """
        nonlocal zero_count
        L = len(word)
        if L <= n:
            return False
        
        # Save state for potential rollback
        saved_zero = zero_count
        updates = []
        
        # Decrement old windows
        k_start = max(0, i - n + 1)
        k_end = min(i + 1, L - n + 1)
        for k in range(k_start, k_end):
            pid = window_ids[k]
            if pid >= 0:
                if counts[pid] == 1:
                    zero_count += 1
                counts[pid] -= 1
                updates.append((pid, -1))
        
        # Increment new windows (windows that span the deletion point)
        nk_start = max(0, i - n + 1)
        nk_end = min(i, L - n)
        for k in range(nk_start, nk_end):
            # Construct new window
            if k < i:
                w1 = word[k:i]
                w2 = word[i+1:k+n+1]
                p = tuple(w1 + w2)
            else:
                p = tuple(word[k+1:k+1+n])
            
            pid = perm_to_id.get(p)
            if pid is not None:
                if counts[pid] == 0:
                    zero_count -= 1
                counts[pid] += 1
                updates.append((pid, 1))
        
        # Check validity
        if zero_count == 0:
            word.pop(i)
            # Update window_ids
            new_L = L - 1
            new_num_windows = new_L - n + 1
            
            # Recompute affected windows (those that span the gap)
            for k in range(max(0, i - n + 1), min(i, new_num_windows)):
                p = tuple(word[k:k+n])
                window_ids[k] = perm_to_id.get(p, -1)
            
            # Shift remaining window_ids
            for k in range(i, new_num_windows):
                window_ids[k] = window_ids[k+1]
            window_ids.pop()
            
            return True
        else:
            # Revert
            for pid, delta in updates:
                counts[pid] -= delta
            zero_count = saved_zero
            return False

    # Local Search: Swap adjacent
    def try_swap(i):
        """
        Attempts to swap word[i] and word[i+1]. Returns True if successful and leads to improvement.
        """
        nonlocal zero_count
        L = len(word)
        if i >= L - 1:
            return False
            
        saved_zero = zero_count
        updates = []
        
        # Decrement old windows
        k_start = max(0, i - n + 1)
        k_end = min(i + 2, L - n + 1)
        for k in range(k_start, k_end):
            pid = window_ids[k]
            if pid >= 0:
                if counts[pid] == 1:
                    zero_count += 1
                counts[pid] -= 1
                updates.append((pid, -1))
        
        # Perform swap
        word[i], word[i+1] = word[i+1], word[i]
        
        # Increment new windows
        for k in range(k_start, k_end):
            p = tuple(word[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                if counts[pid] == 0:
                    zero_count -= 1
                counts[pid] += 1
                updates.append((pid, 1))
                window_ids[k] = pid
        
        if zero_count == 0:
            # Valid swap. Try to delete to see if it helps.
            saved_word = word[:]
            saved_zero2 = zero_count
            saved_window_ids = window_ids[:]
            
            # Try delete i
            if try_delete(i):
                return True
            # Try delete i+1
            if try_delete(i+1):
                return True
            
            # No deletion possible. Revert swap.
            word[:] = saved_word
            zero_count = saved_zero2
            window_ids[:] = saved_window_ids
            for pid, delta in updates:
                counts[pid] -= delta
            zero_count = saved_zero
            return False
        else:
            # Invalid swap. Revert.
            word[i], word[i+1] = word[i+1], word[i]
            for pid, delta in updates:
                counts[pid] -= delta
            zero_count = saved_zero
            for k in range(k_start, k_end):
                p = tuple(word[k:k+n])
                window_ids[k] = perm_to_id.get(p, -1)
            return False

    # Local Search: Block deletion (delete 2 adjacent chars)
    def try_delete_block(i, length=2):
        """
        Attempts to delete word[i:i+length]. Returns True if successful.
        """
        nonlocal zero_count
        L = len(word)
        if L - length < n:
            return False
        
        saved_zero = zero_count
        updates = []
        
        # Decrement old windows affected by the deletion
        # Windows that include any position in [i, i+length-1]
        k_start = max(0, i - n + 1)
        k_end = min(i + length, L - n + 1)
        for k in range(k_start, k_end):
            pid = window_ids[k]
            if pid >= 0:
                if counts[pid] == 1:
                    zero_count += 1
                counts[pid] -= 1
                updates.append((pid, -1))
        
        # Increment new windows (windows that span the deletion gap)
        # In the new word, positions after i+length shift left by length
        # New windows starting at k where the window spans the gap
        nk_start = max(0, i - n + 1)
        nk_end = min(i + length, L - length - n + 1)
        for k in range(nk_start, nk_end):
            # Construct new window
            if k < i:
                w1 = word[k:i]
                w2 = word[i+length:k+n+length]
                p = tuple(w1 + w2)
            elif k < i + length:
                # This shouldn't happen given our range, but just in case
                w1 = word[k:i+length] if k <= i+length else []
                w2 = word[i+length:k+n+length]
                p = tuple(w1 + w2)
            else:
                p = tuple(word[k+length:k+length+n])
            
            pid = perm_to_id.get(p)
            if pid is not None:
                if counts[pid] == 0:
                    zero_count -= 1
                counts[pid] += 1
                updates.append((pid, 1))
        
        # Check validity
        if zero_count == 0:
            # Perform deletion
            del word[i:i+length]
            # Update window_ids
            new_L = L - length
            new_num_windows = new_L - n + 1
            
            # Recompute affected windows
            for k in range(max(0, i - n + 1), min(i + length, new_num_windows)):
                p = tuple(word[k:k+n])
                window_ids[k] = perm_to_id.get(p, -1)
            
            # Shift remaining window_ids
            for k in range(i + length, new_num_windows):
                window_ids[k] = window_ids[k+length]
            for _ in range(length):
                window_ids.pop()
            
            return True
        else:
            # Revert
            for pid, delta in updates:
                counts[pid] -= delta
            zero_count = saved_zero
            return False

    # Phase 1: Deletion passes
    improved = True
    passes = 0
    while improved and not time_up():
        improved = False
        passes += 1
        # Left to right
        i = 0
        while i < len(word) and not time_up():
            if try_delete(i):
                improved = True
            else:
                i += 1
        # Right to left
        i = len(word) - 1
        while i >= 0 and not time_up():
            if try_delete(i):
                improved = True
            else:
                i -= 1

    # Phase 2: Swap passes
    improved = True
    while improved and not time_up():
        improved = False
        i = 0
        while i < len(word) - 1 and not time_up():
            if try_swap(i):
                improved = True
            else:
                i += 1

    # Phase 3: Block deletion passes (delete 2 adjacent)
    improved = True
    while improved and not time_up():
        improved = False
        i = 0
        while i < len(word) - 1 and not time_up():
            if try_delete_block(i, 2):
                improved = True
            else:
                i += 1

    # Phase 4: More deletion passes
    improved = True
    while improved and not time_up():
        improved = False
        i = 0
        while i < len(word) and not time_up():
            if try_delete(i):
                improved = True
            else:
                i += 1
        i = len(word) - 1
        while i >= 0 and not time_up():
            if try_delete(i):
                improved = True
            else:
                i -= 1

    # Phase 5: More swap passes
    improved = True
    while improved and not time_up():
        improved = False
        i = 0
        while i < len(word) - 1 and not time_up():
            if try_swap(i):
                improved = True
            else:
                i += 1

    # Final polish: alternating deletion and swap
    for _ in range(5):
        if time_up():
            break
        improved = False
        i = 0
        while i < len(word) and not time_up():
            if try_delete(i):
                improved = True
            else:
                i += 1
        if not improved:
            break
        improved = False
        i = 0
        while i < len(word) - 1 and not time_up():
            if try_swap(i):
                improved = True
            else:
                i += 1
        if not improved:
            break

    return "".join(map(str, word))
