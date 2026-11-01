import time
import random
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # 1. Generate initial superpermutation
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
                if o == 0:
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    if n <= 4:
        return "".join(map(str, word))
    
    # 2. Local Search with Kicks
    current = list(word)
    L = len(current)
    
    perm_count = defaultdict(int)
    for i in range(L - n + 1):
        p = tuple(current[i:i+n])
        perm_count[p] += 1
        
    target_perms = 1
    for i in range(1, n + 1):
        target_perms *= i
        
    best_word = list(current)
    best_len = L
    
    def update_counts_remove(i):
        # Remove windows containing i
        start_w = max(0, i - n + 1)
        end_w = min(i, L - n)
        for w in range(start_w, end_w + 1):
            p = tuple(current[w:w+n])
            perm_count[p] -= 1
            if perm_count[p] == 0:
                del perm_count[p]

    def update_counts_add(i, symbol):
        # Add windows containing i (after insertion at i)
        # Windows start from i-n+1 to i
        start_w = max(0, i - n + 1)
        end_w = min(i, L) # L is new length? No, L is current length before insertion
        # If we insert at i, the new element is at i.
        # Windows ending at i, i-1, ..., i-n+1
        # Start indices: i-n+1, ..., i
        for w in range(start_w, end_w + 1):
            if w + n <= L + 1: # New length is L+1
                p = tuple(current[w:w+n])
                perm_count[p] += 1

    def try_delete(i):
        nonlocal current, L
        if i < 0 or i >= L:
            return False
            
        start_w = max(0, i - n + 1)
        end_w = min(i, L - n)
        
        old_windows = []
        for w in range(start_w, end_w + 1):
            p = tuple(current[w:w+n])
            old_windows.append(p)
            
        # New windows after deletion
        # Window starting at j, for j in [max(0, i-n+1), i-1]
        # It consists of current[j:i] + current[i+1:i+1+(n-(i-j))]
        new_windows = []
        for j in range(max(0, i - n + 1), i):
            if j + n > L - 1:
                continue
            prefix = current[j:i]
            suffix_len = n - (i - j)
            suffix = current[i+1:i+1+suffix_len]
            if len(suffix) == suffix_len:
                p = tuple(prefix + suffix)
                new_windows.append(p)
                
        new_set = set(new_windows)
        
        for p in old_windows:
            if perm_count[p] == 1 and p not in new_set:
                return False
                
        # Perform deletion
        for p in old_windows:
            perm_count[p] -= 1
            if perm_count[p] == 0:
                del perm_count[p]
                
        for p in new_windows:
            perm_count[p] += 1
            
        current = current[:i] + current[i+1:]
        L -= 1
        return True

    def kick():
        nonlocal current, L
        # Pick random position and symbol
        pos = random.randint(0, L)
        sym = random.randint(1, n)
        
        # Save state
        old_current = list(current)
        old_L = L
        old_counts = dict(perm_count)
        
        # Insert
        current.insert(pos, sym)
        L += 1
        
        # Update counts for insertion
        # This is tricky to do incrementally in the helper. 
        # For simplicity and speed, let's just rebuild counts if we kick often? 
        # No, rebuild is O(L*n). 
        # Let's just rebuild counts after kick if we decide to keep it?
        # Or just do a full scan for deletions after kick.
        
        # Try to delete some symbols
        attempts = 0
        while attempts < 20 and time.time() < deadline:
            deleted = False
            # Randomized scan
            indices = list(range(L))
            random.shuffle(indices)
            for i in indices:
                if try_delete(i):
                    deleted = True
                    break
            if not deleted:
                break
            attempts += 1
            
        # Check if we improved
        if L < best_len:
            best_len = L
            best_word = list(current)
            # Update perm_count to match current? 
            # We need perm_count to be consistent with current for next iterations.
            # Rebuild perm_count
            perm_count.clear()
            for i in range(L - n + 1):
                p = tuple(current[i:i+n])
                perm_count[p] += 1
        else:
            # Revert
            current = old_current
            L = old_L
            perm_count.clear()
            perm_count.update(old_counts)

    # Main Loop
    while time.time() < deadline:
        deleted_any = False
        # Forward scan
        for i in range(L):
            if time.time() > deadline:
                break
            if try_delete(i):
                deleted_any = True
                if L < best_len:
                    best_len = L
                    best_word = list(current)
        
        if not deleted_any:
            # Stuck, try a kick
            if time.time() < deadline - 5: # Reserve time
                kick()
            else:
                break
                
    return "".join(map(str, best_word))
