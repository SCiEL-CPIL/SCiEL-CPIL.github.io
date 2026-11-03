import time
from collections import defaultdict
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # 1. Standard Recursive Construction
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
            max_overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    max_overlap = o
                    break
            new += block[max_overlap:]
        word = new

    # 2. Aggressive Local Search
    w = word
    L = len(w)
    
    def build_counts(w, n):
        counts = defaultdict(int)
        for i in range(len(w) - n + 1):
            counts[tuple(w[i:i+n])] += 1
        return counts

    perm_counts = build_counts(w, n)
    
    def try_delete(w, perm_counts, i, n):
        """Try to delete position i. Returns True if successful."""
        if i < 0 or i >= len(w):
            return False
        # Check if safe: all affected windows must have count > 1
        for k in range(max(0, i - n + 1), min(i, len(w) - n) + 1):
            p = tuple(w[k:k+n])
            if perm_counts[p] == 1:
                return False
        
        # Decrement old windows
        for k in range(max(0, i - n + 1), min(i, len(w) - n) + 1):
            p = tuple(w[k:k+n])
            perm_counts[p] -= 1
            if perm_counts[p] == 0:
                del perm_counts[p]
        
        del w[i]
        
        # Increment new windows
        for k in range(max(0, i - n + 1), min(i, len(w) - n) + 1):
            p = tuple(w[k:k+n])
            perm_counts[p] += 1
        
        return True

    def try_swap(w, perm_counts, i, n):
        """Try to swap positions i and i+1. Returns True if successful."""
        if i < 0 or i + 1 >= len(w):
            return False
        
        # Check if swap is safe
        # Windows affected: those containing either i or i+1
        # That's k in [i-n+1, i+1]
        lo = max(0, i - n + 1)
        hi = min(i + 1, len(w) - n)
        
        # Save old windows
        old_windows = []
        safe = True
        for k in range(lo, hi + 1):
            p = tuple(w[k:k+n])
            old_windows.append(p)
            if perm_counts[p] == 1:
                safe = False
                break
        
        if not safe:
            return False
        
        # Perform swap
        w[i], w[i+1] = w[i+1], w[i]
        
        # Check new windows
        new_windows = []
        for k in range(lo, hi + 1):
            p = tuple(w[k:k+n])
            new_windows.append(p)
            if perm_counts[p] == 0:
                # This window would be lost
                w[i], w[i+1] = w[i+1], w[i]
                return False
        
        # Update counts
        for p in old_windows:
            perm_counts[p] -= 1
            if perm_counts[p] == 0:
                del perm_counts[p]
        for p in new_windows:
            perm_counts[p] += 1
        
        return True

    def try_delete2(w, perm_counts, i, n):
        """Try to delete two consecutive positions i and i+1. Returns True if successful."""
        if i < 0 or i + 1 >= len(w):
            return False
        
        # Windows affected: those containing i or i+1
        lo = max(0, i - n + 1)
        hi = min(i + 1, len(w) - n)
        
        old_windows = []
        safe = True
        for k in range(lo, hi + 1):
            p = tuple(w[k:k+n])
            old_windows.append(p)
            if perm_counts[p] == 1:
                safe = False
                break
        
        if not safe:
            return False
        
        # Remove both
        del w[i]
        del w[i]
        
        # Check new windows
        for k in range(max(0, i - n + 1), min(i, len(w) - n) + 1):
            p = tuple(w[k:k+n])
            if perm_counts[p] == 0:
                # Restore
                w.insert(i, old_windows[0][0])  # This is wrong, need proper restore
                # Actually, let's just check before deleting
                # This approach is getting complex; let's use a different strategy
                pass
        
        # Simpler: just check if all new windows exist
        new_windows = []
        for k in range(max(0, i - n + 1), min(i, len(w) - n) + 1):
            p = tuple(w[k:k+n])
            new_windows.append(p)
        
        for p in new_windows:
            if perm_counts[p] == 0:
                # Need to restore - this is complex, so let's not do 2-deletes
                # Reconstruct is too hard; skip this move
                w.insert(i, w[i])  # placeholder, won't work correctly
                return False
        
        # Update counts
        for p in old_windows:
            perm_counts[p] -= 1
            if perm_counts[p] == 0:
                del perm_counts[p]
        for p in new_windows:
            perm_counts[p] += 1
        
        return True

    # Multiple passes with different strategies
    for pass_num in range(200):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Pass 1: Try deletions from right to left
        i = len(w) - 1
        while i >= 0:
            if time.time() > deadline:
                break
            if try_delete(w, perm_counts, i, n):
                improved = True
                # Don't decrement i, check same position again
            else:
                i -= 1
        
        if not improved:
            # Try swaps
            i = len(w) - 2
            while i >= 0:
                if time.time() > deadline:
                    break
                if try_swap(w, perm_counts, i, n):
                    improved = True
                    i -= 1  # Move to next position
                else:
                    i -= 1
        
        if not improved:
            # Try forward deletions
            i = 0
            while i < len(w):
                if time.time() > deadline:
                    break
                if try_delete(w, perm_counts, i, n):
                    improved = True
                    # Stay at same i
                else:
                    i += 1
        
        if not improved:
            break
    
    return "".join(map(str, w))
