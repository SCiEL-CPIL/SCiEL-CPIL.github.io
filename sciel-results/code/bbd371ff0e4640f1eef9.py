import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    # 150s limit. Use 140s for construction + search to be safe.
    deadline = start_time + 140.0
    
    # For small n, return known optimal constructions
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "12341231423124312" # Standard is 33, optimal is 33.
        # Actually, let's just let the algorithm run for n=4, it's fast.
    
    # Build the standard superpermutation via recursive insertion
    word = [1]
    for m in range(2, n + 1):
        # Find all distinct (m-1)-permutations in current word
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Insert m into each permutation
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap with current new_word
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    
    # For n <= 4, the standard construction is optimal
    if n <= 4:
        return "".join(map(str, word))
    
    # Precompute target number of permutations
    target_count = 1
    for i in range(2, n + 1):
        target_count *= i
        
    # Current word state
    current_word = word
    current_len = len(current_word)
    
    # Helper to compute initial window counts
    def compute_window_counts(w):
        counts = {}
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                counts[window] = counts.get(window, 0) + 1
        return counts

    perm_count = compute_window_counts(current_word)
    
    # Helper to check if a specific window is covered elsewhere
    # This is implicitly handled by the count check in the removal logic.
    
    # Main optimization loop
    improved = True
    pass_count = 0
    max_passes = 50 # Increased passes to allow more random shuffles
    
    while improved and pass_count < max_passes and time.time() < deadline:
        improved = False
        pass_count += 1
        
        # Randomize scan order to avoid bias
        indices = list(range(len(current_word)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            if i >= len(current_word):
                continue
            
            # Determine which windows are affected by removing position i
            # Old windows containing i: start indices j such that j <= i <= j+n-1
            # j ranges from max(0, i-n+1) to min(i, len(current_word)-n)
            old_start = max(0, i - n + 1)
            old_end = min(i, len(current_word) - n)
            
            affected_old = []
            for j in range(old_start, old_end + 1):
                window = tuple(current_word[j:j+n])
                if len(set(window)) == n:
                    affected_old.append((j, window))
            
            # New windows after removal:
            # The word is W[:i] + W[i+1:].
            # New windows starting at j in the new word.
            # A new window at index j in the new word corresponds to:
            # If j < i: it uses W[j:i] + W[i+1:j+n] (length n)
            # If j >= i: it uses W[j+1:j+1+n] (shifted)
            # But we only care about windows that *change*.
            # Windows starting at j < i and ending at >= i are the ones that change.
            # Specifically, new windows starting at j in [max(0, i-n+1), i-1]
            # Wait, if we remove index i, the new window starting at j (where j < i)
            # is composed of current_word[j:i] + current_word[i+1:j+n].
            # This is only a valid window if j+n > i (so it crosses the gap).
            # j+n-1 >= i => j >= i-n+1.
            # Also j must be a valid start index in the new word.
            # New word length is L-1. Max start index is (L-1)-n.
            # So j ranges from max(0, i-n+1) to min(i-1, len(current_word)-n-1).
            
            affected_new = []
            new_start = max(0, i - n + 1)
            new_end = min(i - 1, len(current_word) - n - 1)
            
            for j in range(new_start, new_end + 1):
                # Construct the new window
                # It consists of current_word[j:i] and current_word[i+1:j+n]
                # Length of first part: i - j
                # Length of second part: (j+n) - (i+1) = j + n - i - 1
                # Total length: i - j + j + n - i - 1 = n - 1 ?
                # Wait.
                # Original window at j: W[j...j+n-1]
                # If i is in this window, i.e., j <= i <= j+n-1.
                # After removing W[i], the new window starting at j is:
                # W[j...i-1] + W[i+1...j+n-1] + W[j+n] ? No.
                # The new word is W' = W[0...i-1] + W[i+1...].
                # Window starting at j in W' (for j < i):
                # It takes W'[j...j+n-1].
                # W'[j...i-1] corresponds to W[j...i-1].
                # W'[i...j+n-1] corresponds to W[i+1...j+n].
                # So the new window is W[j:i] + W[i+1:j+n+1].
                # Let's verify length: (i-j) + (j+n+1 - (i+1)) = i-j + j+n-i = n. Correct.
                
                new_window = tuple(current_word[j:i] + current_word[i+1:j+n+1])
                if len(set(new_window)) == n:
                    affected_new.append((j, new_window))
            
            # Check if removal is safe
            can_remove = True
            for j, window in affected_old:
                if perm_count.get(window, 0) <= 1:
                    # This window would be lost.
                    # Check if it appears in affected_new
                    still_covered = False
                    for nj, nw in affected_new:
                        if nw == window:
                            still_covered = True
                            break
                    if not still_covered:
                        can_remove = False
                        break
            
            if not can_remove:
                continue
            
            # Perform removal
            removed_symbol = current_word[i]
            new_word = current_word[:i] + current_word[i+1:]
            
            # Update perm_count
            for j, window in affected_old:
                perm_count[window] -= 1
                if perm_count[window] == 0:
                    del perm_count[window]
            
            for j, window in affected_new:
                perm_count[window] = perm_count.get(window, 0) + 1
            
            current_word = new_word
            current_len = len(current_word)
            improved = True
            
            # Safety check: ensure we still cover all perms
            # We can do this occasionally to prevent drift
            if pass_count % 10 == 0 and time.time() < deadline - 5:
                if len(perm_count) < target_count:
                    # Revert
                    current_word = current_word[:i] + [removed_symbol] + current_word[i:]
                    for j, window in affected_old:
                        perm_count[window] = perm_count.get(window, 0) + 1
                    for j, window in affected_new:
                        perm_count[window] -= 1
                        if perm_count[window] == 0:
                            del perm_count[window]
                    improved = False
                    continue

    # Final verification
    if time.time() < deadline - 2:
        # Quick check
        seen = set()
        for i in range(len(current_word) - n + 1):
            window = tuple(current_word[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        if len(seen) != target_count:
            # Fallback to original
            current_word = word
    
    return "".join(map(str, current_word))
