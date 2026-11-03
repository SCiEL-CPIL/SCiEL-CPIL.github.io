<<<<<<< SEARCH
def _build_required_perms(n):
    """Build the set of all n! permutations as tuples."""
    return set(itertools.permutations(range(1, n + 1)))

def _count_windows(word, n, required_perms):
    """Count how many required permutations appear as windows in word."""
    count = 0
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if window in required_perms:
            count += 1
    return count

def _try_remove_fast(word, n, required_perms, deadline):
    """Try to remove symbols from word using fast counting. Returns shortened word."""
    word = list(word)
    target = len(required_perms)
    current_count = _count_windows(word, n, required_perms)
    if current_count < target:
        return word
    
    i = 0
    while i < len(word) - n + 1 and time.time() < deadline:
        # Try removing word[i]
        test = word[:i] + word[i+1:]
        new_count = _count_windows(test, n, required_perms)
        if new_count >= target:
            word = test
            current_count = new_count
            # Don't increment i since we removed an element
        else:
            i += 1
    return word

def _try_remove_multi_pass(word, n, required_perms, deadline):
    """Multiple passes of removal until no improvement or time limit."""
    word = list(word)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        i = 0
        while i < len(word) - n + 1 and time.time() < deadline:
            test = word[:i] + word[i+1:]
            if _count_windows(test, n, required_perms) >= len(required_perms):
                word = test
                improved = True
                # Don't increment i
            else:
                i += 1
    return word
=======
def _perm_to_id(perm, n):
    """Map a permutation tuple to an integer ID using factorial number system."""
    # Precompute factorials
    # This is called many times, so we should cache or use a dict.
    # For n=7, we can just use a dict mapping tuple to int.
    pass

def _fast_remove(word, n, deadline):
    """
    Efficiently remove redundant symbols from the superpermutation.
    Uses a counter array to track permutation occurrences.
    """
    if n < 2:
        return word
    
    # Map all permutations to IDs 0..n!-1
    perms = list(itertools.permutations(range(1, n + 1)))
    perm_id = {p: i for i, p in enumerate(perms)}
    num_perms = len(perms)
    
    word = list(word)
    L = len(word)
    
    # Initialize counts
    # window_start i corresponds to word[i:i+n]
    # Number of windows is L - n + 1
    counts = [0] * num_perms
    distinct = 0
    
    # Precompute window IDs for current word
    window_ids = [0] * (L - n + 1)
    for i in range(L - n + 1):
        window = tuple(word[i:i+n])
        wid = perm_id[window]
        window_ids[i] = wid
        if counts[wid] == 0:
            distinct += 1
        counts[wid] += 1
    
    if distinct < num_perms:
        return word # Should not happen for valid superperm
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        i = 0
        L = len(word)
        # We need to be careful with indices when removing
        # If we remove word[i], windows affected are those starting from max(0, i-n+1) to min(i, L-n)
        
        while i < L - n + 1 and time.time() < deadline:
            # Try removing word[i]
            # Windows covering index i:
            # Start index s such that s <= i <= s+n-1 => i-n+1 <= s <= i
            start_min = max(0, i - n + 1)
            start_max = min(i, L - n)
            
            lost_perm = -1
            affected_windows = []
            
            for s in range(start_min, start_max + 1):
                wid = window_ids[s]
                if counts[wid] == 1:
                    lost_perm = wid
                    break
                # We don't need to update counts yet, just check if it's the last one
                affected_windows.append((s, wid))
            
            if lost_perm != -1:
                # Removing this char would lose a permutation
                i += 1
            else:
                # Safe to remove
                # Update counts and window_ids
                # Decrement counts for affected windows
                for s, wid in affected_windows:
                    counts[wid] -= 1
                    # If count becomes 0, distinct decreases (but we checked it won't)
                
                # Remove word[i]
                word.pop(i)
                L -= 1
                
                # Update window_ids
                # Windows starting after i are shifted left by 1
                # Windows starting before i are unchanged
                # The window starting at i is gone? No.
                # Original windows: ... W_{i-1}, W_i, W_{i+1} ...
                # W_i covers i..i+n-1. Removing i, the new window at i is old W_{i+1} shifted?
                # Let's trace carefully.
                # Word: ... w_{i-1}, w_i, w_{i+1}, ...
                # Window at i: w_i...w_{i+n-1}
                # Window at i+1: w_{i+1}...w_{i+n}
                # Remove w_i.
                # New word: ... w_{i-1}, w_{i+1}, ...
                # New window at i-1 (if exists): w_{i-1}, w_{i+1}, ... (shifted)
                # New window at i: w_{i+1}, w_{i+2}, ... (this is old window i+1)
                
                # Actually, it's easier to recompute window_ids for the affected range.
                # The indices of windows change.
                # Old windows: 0 .. L-n
                # New windows: 0 .. L-1-n
                # Windows 0..i-1 are unchanged in content?
                # Window s < i: word[s:s+n]. Since removal is at i > s+n-1? No.
                # If s < i, the window might include i?
                # If s+n-1 >= i, then the window content changes.
                # This logic is getting complex to update incrementally without errors.
                
                # Given the small size (n=7, L~6000), and that we only remove ~8 chars,
                # the overhead of recomputing window_ids for the whole word is:
                # 6000 * 7 ops = 42000 ops.
                # If we do this for every removal attempt, and we have 6000 attempts,
                # 6000 * 42000 = 252,000,000 ops.
                # In Python, this might take ~10-20 seconds.
                # This is acceptable for 120s limit if we are careful.
                
                # Let's just recompute window_ids and counts from scratch after each successful removal?
                # No, that's O(L*n) per removal. 6000 * 42000 = 252M.
                # But we only do this when we *successfully* remove a char.
                # We expect to remove ~8 chars.
                # The *failed* attempts just check counts (O(n)).
                # So:
                # 1. Loop i from 0 to L-n.
                # 2. Check if removal is safe (O(n)).
                # 3. If safe, remove, recompute all window_ids and counts (O(L*n)).
                # 4. Repeat.
                
                # This seems viable.
                
                # Recompute
                L = len(word)
                window_ids = [0] * (L - n + 1)
                counts = [0] * num_perms
                distinct = 0
                for s in range(L - n + 1):
                    window = tuple(word[s:s+n])
                    wid = perm_id[window]
                    window_ids[s] = wid
                    if counts[wid] == 0:
                        distinct += 1
                    counts[wid] += 1
                
                improved = True
                # Do not increment i, check new position i
                
    return word
>>>>>>> REPLACE

Wait, the logic in `_fast_remove` above has a flaw: I update `window_ids` and `counts` inside the loop but the `window_ids` variable is local to the recompute block, and the outer loop uses the old `window_ids` for the safety check?
No, in the code I wrote:
`for s in range(start_min, start_max + 1): wid = window_ids[s]`
This uses the `window_ids` from the *previous* iteration.
If I recompute `window_ids` after a removal, the next iteration of `i` will use the *new* `window_ids`.
However, the `i` loop logic `i += 1` vs `i` stays same needs to be consistent with the new length.
If I remove `word[i]`, the new `word` has length `L-1`.
The window at index `i` in the new word corresponds to the window starting at `i` in the new word.
The old window at `i` is gone. The old window at `i+1` becomes the new window at `i`.
So checking the new `word[i]` (which was `word[i+1]` before) is correct.
The `window_ids` array is updated to reflect the new word.
So the safety check for the new `word[i]` will use the correct `window_ids`.

One detail: `perm_id` dictionary lookup `perm_id[window]` inside the recompute loop is $O(1)$ average.
Total work per removal: $O(L \cdot n)$.
Number of removals: small.
Number of failed checks: $O(L \cdot n)$ per pass? No, failed check is $O(n)$.
Total work per pass: $O(L \cdot n)$ (for the successful removals, which are few) + $O(L \cdot n)$ (for the failed checks? No, failed checks are $O(n)$ each, so $L \cdot n$ total for the pass).
So one pass is roughly $O(L \cdot n)$.
$L \approx 6000, n=7$. $42000$ operations.
Python can do ~10-50 million simple ops per second.
So one pass is ~1ms.
We can do thousands of passes in 120s.
This is much faster than the original.

Let's refine the `_fast_remove` function to be clean.

