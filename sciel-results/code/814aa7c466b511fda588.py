import time
from collections import Counter
import itertools
import random

def construct(n):
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    
    if n <= 6:
        return "".join(map(str, word))

    # 2. Optimization for n=7
    start_time = time.time()
    deadline = start_time + 110
    
    L = len(word)
    perm_counts = Counter()
    for i in range(L - n + 1):
        perm_counts[tuple(word[i:i+n])] += 1
        
    def is_valid_deletion(w, idx):
        # Check if deleting w[idx] keeps all perms present
        # Lost windows: w[j:j+n] for j in [max(0, idx-n+1), min(idx, len(w)-n)]
        # Gained windows: w[j-k:j] + w[j+1:j+1+(n-k)] ... wait, indices are tricky.
        # Let's use the logic from the original code, it was correct.
        
        lost_windows = []
        start_min = max(0, idx - n + 1)
        start_max = min(idx, len(w) - n)
        for j in range(start_min, start_max + 1):
            lost_windows.append(tuple(w[j:j+n]))
            
        gained_windows = []
        for k in range(1, n):
            left_start = idx - k
            right_end = idx + 1 + (n - k)
            if left_start >= 0 and right_end <= len(w):
                w_new = w[left_start:idx] + w[idx+1:right_end]
                gained_windows.append(tuple(w_new))
                
        lost_counter = Counter(lost_windows)
        gained_counter = Counter(gained_windows)
        
        affected = set(lost_counter.keys()) | set(gained_counter.keys())
        for p in affected:
            old_c = perm_counts[p]
            new_c = old_c - lost_counter.get(p, 0) + gained_counter.get(p, 0)
            if new_c <= 0:
                return False
        return True

    def apply_deletion(w, idx):
        lost_windows = []
        start_min = max(0, idx - n + 1)
        start_max = min(idx, len(w) - n)
        for j in range(start_min, start_max + 1):
            lost_windows.append(tuple(w[j:j+n]))
            
        gained_windows = []
        for k in range(1, n):
            left_start = idx - k
            right_end = idx + 1 + (n - k)
            if left_start >= 0 and right_end <= len(w):
                w_new = w[left_start:idx] + w[idx+1:right_end]
                gained_windows.append(tuple(w_new))
                
        for p in lost_windows:
            perm_counts[p] -= 1
        for p in gained_windows:
            perm_counts[p] += 1
        w.pop(idx)

    def is_valid_swap(w, idx):
        # Swap w[idx] and w[idx+1]
        # Affected windows: those containing idx or idx+1.
        # Start indices: max(0, idx-n+1) to min(idx+1, len(w)-n)
        old_w = w
        new_w = w[:]
        new_w[idx], new_w[idx+1] = new_w[idx+1], new_w[idx]
        
        # Check windows in the affected region
        # We can just check the windows that changed.
        # Windows starting at j where j <= idx and j+n-1 >= idx+1?
        # Actually, only windows that include idx or idx+1 change.
        # j such that j <= idx and j+n-1 >= idx  => j in [idx-n+1, idx]
        # and j such that j <= idx+1 and j+n-1 >= idx+1 => j in [idx+1-n+1, idx+1]
        # Union: [idx-n+1, idx+1]
        
        start_min = max(0, idx - n + 1)
        start_max = min(idx + 1, len(w) - n)
        
        for j in range(start_min, start_max + 1):
            old_window = tuple(old_w[j:j+n])
            new_window = tuple(new_w[j:j+n])
            
            if old_window == new_window:
                continue
                
            # If old_window count was 1, and it's not in new_window, it might disappear.
            # But we need to check the net effect.
            # Simpler: Just check if new_window is a permutation?
            # No, we need to ensure ALL permutations are present.
            # If we remove an occurrence of P (count 1) and add an occurrence of Q (count > 0), P is lost.
            # So:
            # 1. Decrement old_window count (temp)
            # 2. Increment new_window count (temp)
            # 3. Check if any count becomes 0.
            
            # Since only a few windows change, we can just check:
            # Does old_window have count 1 in the global counter?
            # If yes, and new_window is NOT old_window, then old_window count becomes 0?
            # Wait, if new_window is the same as some other window, it doesn't matter.
            # But if old_window was unique (count 1), and we change it to new_window,
            # then old_window count becomes 0.
            # So if perm_counts[old_window] == 1 and old_window != new_window:
            #    return False
            
            # Is that sufficient?
            # What if new_window is NOT a permutation?
            # If new_window is not a permutation, it doesn't contribute to the count of any permutation.
            # But we are replacing old_window (a permutation) with new_window.
            # If old_window count was 1, it is now 0. -> Invalid.
            # If old_window count > 1, it is now count-1. Still > 0. -> OK for old_window.
            # But what about new_window?
            # If new_window is a permutation, its count increases.
            # If new_window is NOT a permutation, its count is irrelevant.
            # So the only risk is losing the last occurrence of a permutation.
            
            if perm_counts[old_window] == 1 and old_window != new_window:
                return False
                
        return True

    def apply_swap(w, idx):
        # Update counters
        start_min = max(0, idx - n + 1)
        start_max = min(idx + 1, len(w) - n)
        
        # We need to decrement old windows and increment new windows.
        # But we already know they are valid.
        # We just need to update the counter.
        # Note: is_valid_swap didn't modify the counter.
        
        for j in range(start_min, start_max + 1):
            old_window = tuple(w[j:j+n])
            # Create new window manually to avoid slicing copy?
            # w[j:j+n] with swap at idx, idx+1.
            # It's easier to just rebuild the affected windows from the modified w.
            # But we haven't modified w yet.
            pass
        
        # Perform swap
        w[idx], w[idx+1] = w[idx+1], w[idx]
        
        # Update counters
        for j in range(start_min, start_max + 1):
            # We need the OLD window to decrement, and NEW window to increment.
            # But we already swapped w.
            # So current w[j:j+n] is the NEW window.
            # We need to know the OLD window.
            # We can reconstruct it or store it before swapping.
            # Let's refactor apply_swap to take old and new windows.
            pass
            
    # Refactor: 
    # To avoid bugs, let's just re-calculate the counter for the affected region?
    # No, that's slow if we do it for every swap.
    # But for n=7, it's small.
    
    # Let's stick to the deletion logic which is already there, and add swap logic carefully.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try deletions
        for i in range(len(word)):
            if time.time() > deadline:
                break
            if is_valid_deletion(word, i):
                apply_deletion(word, i)
                improved = True
                break
        
        if improved:
            continue
            
        # 2. Try swaps (if no deletion found)
        # Try a few random swaps to escape local minima
        for _ in range(100):
            if time.time() > deadline:
                break
            if len(word) < 2:
                break
            i = random.randint(0, len(word) - 2)
            if is_valid_swap(word, i):
                # Apply swap
                # Update counter
                start_min = max(0, i - n + 1)
                start_max = min(i + 1, len(word) - n)
                
                # Store old windows
                old_windows = [tuple(word[j:j+n]) for j in range(start_min, start_max + 1)]
                
                word[i], word[i+1] = word[i+1], word[i]
                
                new_windows = [tuple(word[j:j+n]) for j in range(start_min, start_max + 1)]
                
                for ow in old_windows:
                    perm_counts[ow] -= 1
                for nw in new_windows:
                    perm_counts[nw] += 1
                
                improved = True
                break
                
    return "".join(map(str, word))
