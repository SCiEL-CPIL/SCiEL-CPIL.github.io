import time
import random
import sys

def construct(n):
    start_time = time.time()
    # 150s limit. Use 145s for construction + search to be safe.
    deadline = start_time + 145.0

    # For small n, return known optimal constructions
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "12341231423124312"
    
    # Build the standard superpermutation via recursive insertion
    # This produces a valid superpermutation of length sum_{k=1}^n k!
    # For n=7, this is 1+2+6+24+120+720+5040 = 5913.
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
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
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    
    if n <= 4:
        return "".join(map(str, word))
    
    target_count = 1
    for i in range(2, n + 1):
        target_count *= i
        
    current_word = list(word)
    best_word = list(current_word)
    best_len = len(current_word)
    
    # Precompute window counts
    def compute_counts(w):
        counts = {}
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                counts[window] = counts.get(window, 0) + 1
        return counts

    perm_count = compute_counts(current_word)
    
    def try_remove(w, counts, idx):
        """
        Attempts to remove index idx from w.
        Returns (new_w, new_counts, success)
        """
        if idx < 0 or idx >= len(w):
            return w, counts, False
        
        n_local = len(w[0:1]) * 0 + 7 # n is 7, but let's use the outer n
        # Actually, pass n explicitly or use closure.
        
        # Check validity
        # Windows affected are those starting from max(0, idx-n+1) to min(idx, len(w)-n)
        old_start = max(0, idx - n + 1)
        old_end = min(idx, len(w) - n)
        
        affected_old = []
        for j in range(old_start, old_end + 1):
            window = tuple(w[j:j+n])
            if len(set(window)) == n:
                if counts.get(window, 0) == 1:
                    return w, counts, False
                affected_old.append((j, window))
        
        # Perform removal
        new_w = w[:idx] + w[idx+1:]
        
        # Update counts
        for j, window in affected_old:
            counts[window] -= 1
            if counts[window] == 0:
                del counts[window]
        
        # Add new windows
        # New windows span the junction.
        # Junction is between new_w[idx-1] and new_w[idx] (which is w[idx+1])
        # New windows start at k in [idx-n+1, idx-1] in new_w coordinates
        # But only if they are within bounds and valid
        new_start = max(0, idx - n + 1)
        new_end = min(idx - 1, len(new_w) - n)
        
        for k in range(new_start, new_end + 1):
            new_window = tuple(new_w[k:k+n])
            if len(set(new_window)) == n:
                counts[new_window] = counts.get(new_window, 0) + 1
        
        return new_w, counts, True

    # Optimization Loop with Restarts
    # We will run multiple passes. If we get stuck, we restart with a fresh random order.
    # To increase diversity, we can also try "swaps" if we get stuck, but deletions are primary.
    
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline - 2:
            break
            
        # Shuffle indices for randomization
        indices = list(range(len(current_word)))
        random.shuffle(indices)
        
        improved = False
        for i in indices:
            if time.time() > deadline - 2:
                break
            if i >= len(current_word):
                continue
            
            # Attempt removal
            new_w, new_counts, success = try_remove(current_word, perm_count, i)
            
            if success:
                current_word = new_w
                perm_count = new_counts
                improved = True
                if len(current_word) < best_len:
                    best_len = len(current_word)
                    best_word = list(current_word)
                # After a removal, indices shift. 
                # We can continue with the same loop but indices are now stale.
                # It's safer to restart the loop or adjust. 
                # Given the small shift, restarting the inner loop is often cleaner 
                # but might waste time. 
                # Let's just continue; if index is out of bounds, we skip.
                # However, the list `indices` is now incorrect for the new word length.
                # We break the inner loop and restart the outer loop (next restart) 
                # or simply break and let the outer loop handle it?
                # Actually, a single pass with stale indices is inefficient.
                # Let's break and do a new pass.
                break
        
        if not improved:
            # Stuck. Try a different random seed or just continue to next restart.
            # We can also try to "jitter" the word: swap two adjacent chars, 
            # then try to remove. But this is complex.
            # For now, just restart with a new shuffle.
            pass

    # Final verification
    if time.time() < deadline - 1:
        seen = set()
        for i in range(len(best_word) - n + 1):
            window = tuple(best_word[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        if len(seen) != target_count:
            # Revert to original if invalid
            best_word = list(word)
    
    return "".join(map(str, best_word))
