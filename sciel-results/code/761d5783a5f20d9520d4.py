import time
import itertools
import math
from collections import defaultdict

def _rank_perm(p, n):
    """Compute lexicographic rank of permutation p (0-indexed) in 0..n!-1."""
    used = [False] * (n + 1)
    rank = 0
    for i in range(n):
        # Count unused elements smaller than p[i]
        smaller_unused = 0
        for v in range(1, p[i]):
            if not used[v]:
                smaller_unused += 1
        rank += smaller_unused * math.factorial(n - i - 1)
        used[p[i]] = True
    return rank

def construct(n):
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
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
            overlap = 0
            # Find max overlap
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # 2. Efficient Greedy Deletion
    target_perms = math.factorial(n)
    W = len(word)
    
    # Pre-compute initial window ranks and counts
    perm_counts = [0] * target_perms
    window_ranks = [0] * (W - n + 1)
    for i in range(W - n + 1):
        r = _rank_perm(tuple(word[i:i+n]), n)
        window_ranks[i] = r
        perm_counts[r] += 1
    
    # Helper to update a single window rank at index j in word
    # Returns the new rank
    def get_window_rank(j):
        return _rank_perm(tuple(word[j:j+n]), n)

    # Deletion passes
    # We scan left-to-right, then right-to-left, repeating until no changes or time limit
    while True:
        if time.time() > deadline:
            break
        
        W = len(word)
        changed = False
        
        # Left-to-right pass
        i = 0
        while i < W:
            if time.time() > deadline:
                break
            
            # Indices of windows containing position i
            # Window j contains i if j <= i <= j+n-1 => i-n+1 <= j <= i
            min_j = max(0, i - n + 1)
            max_j = min(i, W - n)
            
            valid = True
            for j in range(min_j, max_j + 1):
                if perm_counts[window_ranks[j]] <= 1:
                    valid = False
                    break
            
            if valid:
                # Remove word[i]
                word.pop(i)
                W -= 1
                changed = True
                
                # Update window_ranks and perm_counts
                # The windows that contained i are now "invalid" in their old form.
                # The new windows are formed by the shift.
                
                # Old affected windows: j in [min_j, max_j]
                # We need to decrement their counts.
                for j in range(min_j, max_j + 1):
                    perm_counts[window_ranks[j]] -= 1
                
                # Determine which windows in the new array correspond to the affected region.
                # After removing index i, the array shrinks by 1.
                # Windows starting at index k in the new array:
                # If k < min_j: unchanged.
                # If k >= max_j: the content is shifted. Specifically, new window k corresponds to old window k+1?
                # Let's trace carefully.
                # Old word: w[0] ... w[i-1] w[i] w[i+1] ...
                # New word: w[0] ... w[i-1] w[i+1] ...
                # 
                # Affected old windows: start indices min_j ... max_j.
                # These windows are "destroyed".
                # New windows that cover the gap:
                # The gap is at index i (in new coordinates, the element w[i+1] is now at i).
                # The windows that include the new position i (which is old i+1) are:
                # New start indices j' such that j' <= i <= j'+n-1 => i-n+1 <= j' <= i.
                # Note: i is the same index in new array.
                # So new affected indices are also [max(0, i-n+1), min(i, W-n)].
                # Let's call this new_min_j, new_max_j.
                
                new_min_j = max(0, i - n + 1)
                new_max_j = min(i, W - n)
                
                # We need to recompute ranks for these new windows.
                # But we also need to handle the list `window_ranks` structure.
                # The list `window_ranks` has length W-n+1.
                # Old length: W_old - n + 1.
                # New length: W_new - n + 1 = W_old - n.
                # So we remove one element from window_ranks. Which one?
                # The windows are indexed by their start position.
                # Removing word[i] shifts all elements after i to the left.
                # Windows starting at j >= i shift their content?
                # No, windows starting at j < i are unaffected in content?
                # Wait.
                # If j < i, the window word[j:j+n] does NOT include word[i] if j+n-1 < i.
                # If j+n-1 >= i, it includes word[i].
                # This is exactly the condition j >= i-n+1.
                # So windows with start j in [i-n+1, i] include word[i].
                
                # Let's update `window_ranks` list.
                # We remove the window starting at i? No.
                # The mapping is:
                # New window j corresponds to old window j if j < min_j?
                # If j < min_j, the window content is identical to old window j.
                # If j >= new_max_j?
                # Let's just rebuild the affected part of window_ranks.
                
                # Actually, it's safer to:
                # 1. Decrement counts for old affected windows.
                # 2. Remove the window at index i from window_ranks?
                #    No, the indices shift.
                #    Old window j (j > i) becomes new window j-1.
                #    Old window j (j <= i) remains new window j?
                #    Wait.
                #    New word[k] = Old word[k] for k < i.
                #    New word[k] = Old word[k+1] for k >= i.
                #    
                #    New window j:
                #    If j + n - 1 < i: New window j == Old window j.
                #    If j >= i: New window j == Old window j+1 (shifted).
                #    If j < i <= j+n-1: New window j is a mix.
                #    
                #    The set of indices j where content changes is [i-n+1, i-1]?
                #    Let's check j = i.
                #    New window i: word[i:i+n].
                #    Old word[i] is gone. New word[i] is Old word[i+1].
                #    So New window i = Old word[i+1 : i+n+1]. This is Old window i+1.
                #    So New window i corresponds to Old window i+1.
                #    
                #    Let's check j = i-1.
                #    New window i-1: word[i-1 : i-1+n].
                #    Includes New word[i] (Old i+1).
                #    Does not include Old word[i].
                #    Old window i-1: word[i-1 : i-1+n]. Includes Old word[i].
                #    So New window i-1 is different from Old window i-1.
                #    
                #    So windows j in [max(0, i-n+1), i] are changed.
                #    Windows j < i-n+1 are unchanged.
                #    Windows j > i are unchanged in content, but their index in the array shifts?
                #    No, `window_ranks` is an array indexed by start position.
                #    The array length decreases by 1.
                #    The element at index i in `window_ranks` (which corresponds to old window i) is removed?
                #    Actually, we can just rebuild the `window_ranks` array for the affected indices.
                
                # To avoid complex list splicing errors, let's just update the specific indices in `window_ranks`.
                # But `window_ranks` itself is a list that shrinks.
                # We can pop the element at index i?
                # If we pop index i from `window_ranks`, then for j > i, `window_ranks[j]` becomes `window_ranks[j-1]` (old).
                # This matches the shift!
                # So:
                # 1. Decrement counts for old affected windows (indices min_j to max_j in OLD array).
                # 2. Pop `window_ranks[i]`? 
                #    Wait, if we pop i, the indices > i shift down.
                #    The old window i+1 is now at index i.
                #    This matches "New window i is Old window i+1".
                #    So yes, pop i from `window_ranks`.
                # 3. Now update the new affected windows.
                #    The new affected indices are new_min_j to new_max_j.
                #    For each j in this range, compute new rank and update `window_ranks[j]` and `perm_counts`.
                
                # Step 1: Decrement old counts
                for j in range(min_j, max_j + 1):
                    perm_counts[window_ranks[j]] -= 1
                
                # Step 2: Pop window_ranks[i]
                # Note: i must be a valid index in window_ranks.
                # window_ranks length is W-n+1.
                # i is in [0, W-1].
                # Is i <= W-n?
                # If i > W-n, then i is not a valid start index for a window in the old array.
                # But we only consider deletion if valid.
                # If i > W-n, then max_j = W-n.
                # The windows affected are min_j to W-n.
                # If we pop i from word, and i > W-n, it means i is in the "tail" where no window starts.
                # Wait, if i > W-n, then i is not a start of any window.
                # But i is contained in windows starting at W-n ... i.
                # Specifically, window W-n ends at W-1.
                # If i is close to end, it's in the last few windows.
                # Does `window_ranks` have an entry for i?
                # `window_ranks` has indices 0 to W-n.
                # If i > W-n, we cannot pop `window_ranks[i]`.
                # In this case, the shift is different?
                # If i > W-n, removing word[i] does not shift any window STARTS?
                # The windows are defined by start index.
                # If we remove a character at the end, the starts 0..W-n are still valid?
                # No, the word length decreases. The valid starts are 0..W-n-1.
                # So the last window (start W-n) is removed?
                # Or does the window at start W-n change?
                # Old word length W. New length W-1.
                # Old windows: 0..W-n.
                # New windows: 0..W-n-1.
                # Window j (j < W-n) in new word:
                # If j+n-1 < i, unchanged.
                # If j+n-1 >= i, changed.
                # Since i > W-n, i is in the last n-1 positions.
                # The window starting at W-n-1 ends at W-2.
                # If i = W-1, it is not in any window?
                # Wait, window W-n-1 is word[W-n-1 : W-1].
                # Window W-n is word[W-n : W].
                # If we remove word[W-1] (i=W-1), the new last window is word[W-n-1 : W-1] (old indices).
                # This is the old window W-n-1.
                # So the last window (old W-n) is gone.
                # So we should pop the last element of `window_ranks`.
                # Generally, we pop the element at index `min(i, W-n)`?
                # If i <= W-n, we pop i.
                # If i > W-n, we pop W-n (the last one).
                
                pop_idx = min(i, W - n)
                # Decrement count for the popped window?
                # We already decremented all affected old windows.
                # The window at pop_idx was one of the affected ones?
                # If i <= W-n, pop_idx = i. i is in [min_j, max_j]?
                # max_j = min(i, W-n) = i. Yes.
                # If i > W-n, pop_idx = W-n.
                # max_j = W-n.
                # min_j = max(0, i-n+1).
                # Since i > W-n, and W-n is the last start, i is near the end.
                # Is W-n in [min_j, max_j]?
                # min_j <= W-n?
                # i-n+1 <= W-n => i <= W-1. True.
                # So yes, pop_idx is in the affected range.
                
                # We already decremented perm_counts for window_ranks[pop_idx].
                # Now remove it from the list.
                window_ranks.pop(pop_idx)
                
                # Step 3: Update new affected windows
                new_min_j = max(0, i - n + 1)
                new_max_j = min(i, W - n) # W is new length here? No, W was decremented before?
                # We did W -= 1 earlier.
                # So W is the new length.
                # Valid starts: 0 to W-n.
                
                for j in range(new_min_j, new_max_j + 1):
                    new_r = get_window_rank(j)
                    window_ranks[j] = new_r
                    perm_counts[new_r] += 1
                
                # Do not increment i, check the new element at i
                continue
            else:
                i += 1
        
        if not changed:
            # Try right-to-left
            i = W - 1
            while i >= 0:
                if time.time() > deadline:
                    break
                
                min_j = max(0, i - n + 1)
                max_j = min(i, W - n)
                
                valid = True
                for j in range(min_j, max_j + 1):
                    if perm_counts[window_ranks[j]] <= 1:
                        valid = False
                        break
                
                if valid:
                    word.pop(i)
                    W -= 1
                    changed = True
                    
                    # Decrement old counts
                    for j in range(min_j, max_j + 1):
                        perm_counts[window_ranks[j]] -= 1
                    
                    pop_idx = min(i, W - n)
                    window_ranks.pop(pop_idx)
                    
                    new_min_j = max(0, i - n + 1)
                    new_max_j = min(i, W - n)
                    
                    for j in range(new_min_j, new_max_j + 1):
                        new_r = get_window_rank(j)
                        window_ranks[j] = new_r
                        perm_counts[new_r] += 1
                    
                    i -= 1
                else:
                    i -= 1
            
            if not changed:
                break

    return "".join(map(str, word))
