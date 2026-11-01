import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    # The evaluator kills after 150s. We use a 140s deadline to be safe.
    deadline = start_time + 140.0

    # 1. Standard recursive construction
    # This gives a valid superpermutation of length sum_{k=1}^n k!
    # For n=7, this is 1+2+6+24+120+720+5040 = 5913.
    # The known optimum is 5905.
    
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
            # Find max overlap
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # For small n, standard is optimal
    if n < 5:
        return "".join(map(str, word))

    # 2. Optimization Phase
    # Goal: Remove redundant symbols to shorten the word while keeping all permutations.
    
    current_word = word
    current_len = len(current_word)
    
    target_count = 1
    for i in range(2, n+1):
        target_count *= i
        
    # Pre-compute the set of all permutations as a set of tuples for fast lookup
    # Actually, we just need to check if the set of windows in the word has size target_count.
    # But to be robust against non-permutation windows (which shouldn't happen if we start valid and only remove),
    # we should ensure all windows are valid permutations.
    # In a superpermutation, all windows of length n are permutations by definition if the construction is standard.
    # However, when we remove symbols, we might create windows that are not permutations?
    # No, if we remove a symbol from a word where all windows were permutations, the new windows
    # are subsequences of the old windows. Wait, no.
    # If we have ... A B C ... and remove B, we get ... A C ...
    # The new windows are formed by shifting.
    # Do they remain permutations?
    # Example: 1 2 3. Remove 2 -> 1 3.
    # If n=3, windows were (1,2,3). Now we have length 2.
    # The problem requires windows of length n.
    # If we remove a symbol, the length decreases.
    # The new windows are the old windows with one element removed, plus new windows formed by the shift.
    # Actually, let's look at the indices.
    # Old: w[0..L-1]. Windows: w[i..i+n-1].
    # Remove w[k]. New word w'[0..L-2].
    # New windows: w'[i..i+n-1].
    # If i+k < n, the window w'[i..i+n-1] corresponds to w[i..i+n-1] with w[k] removed?
    # Not exactly.
    # Let's just check validity by counting unique windows that are permutations.
    
    def is_valid(w):
        if len(w) < n:
            return False
        seen = set()
        # Optimization: use a local variable for n
        nn = n
        ww = w
        L = len(ww)
        for i in range(L - nn + 1):
            window = tuple(ww[i:i+nn])
            # Check if it's a permutation
            # Since we start with a valid superperm, and only remove, 
            # it's possible to create non-permutations?
            # Example: 1 2 3 1 2. Remove 3. 1 2 1 2.
            # Windows of length 3: (1,2,1) - not a perm, (2,1,2) - not a perm.
            # So we MUST check if it's a permutation.
            if len(set(window)) == nn:
                seen.add(window)
        return len(seen) == target_count

    improved = True
    # We will do multiple passes.
    # In each pass, we try to remove every symbol in a random order.
    # If a removal is valid, we keep it and restart the pass (or continue with adjusted indices).
    # Restarting is safer but slower. Given the small size, restarting is fine.
    
    # To speed up is_valid, we can cache the set of windows?
    # No, the word changes.
    
    # Alternative: Incremental update?
    # Removing symbol at index i affects windows starting at max(0, i-n+1) to min(i, L-n).
    # There are at most n windows affected.
    # We can maintain the count of each permutation.
    # Let's try this.
    
    # Map permutation to index?
    # There are 5040 perms. We can use a dictionary {perm_tuple: count}.
    
    # Initialize counts
    counts = {}
    for i in range(len(current_word) - n + 1):
        window = tuple(current_word[i:i+n])
        if len(set(window)) == n:
            counts[window] = counts.get(window, 0) + 1
    
    # Check if valid initially
    if len(counts) != target_count:
        # This should not happen with the standard construction
        pass
        
    def try_remove(i):
        """
        Try to remove symbol at index i.
        Returns True if valid, False otherwise.
        Modifies current_word and counts if valid.
        """
        L = len(current_word)
        if i < 0 or i >= L:
            return False
            
        # Identify affected windows
        # A window starting at j is affected if i is in [j, j+n-1]
        # So j is in [i-n+1, i]
        start_j = max(0, i - n + 1)
        end_j = min(i, L - n)
        
        old_windows = []
        for j in range(start_j, end_j + 1):
            w = tuple(current_word[j:j+n])
            if len(set(w)) == n:
                old_windows.append(w)
        
        # Remove the symbol
        old_word = current_word
        new_word = current_word[:i] + current_word[i+1:]
        
        # Calculate new windows for the affected range
        # The new word is shorter.
        # The windows that "shift" into the gap are the ones that were to the right.
        # Actually, it's easier to just recount the affected range in the new word.
        # The affected range in the new word is indices [start_j, end_j] (approx).
        # Specifically, windows in the new word that overlap with the removed index i.
        # In the new word, the index i is gone.
        # Windows starting at j in the new word:
        # If j < i, the window is new_word[j:j+n]. This corresponds to old_word[j:j+n] with old_word[i] removed?
        # No, if j < i, new_word[j:j+n] contains elements from old_word[j..i-1] and old_word[i+1..j+n].
        # So it's a different window.
        
        # Let's just recalculate the counts for the affected windows in the new word.
        # Which windows in the new word are different from the old word?
        # Any window in the new word that covers the position where i was removed.
        # In the new word, the "gap" is between i-1 and i.
        # So windows starting at j in [i-n+1, i-1] are affected.
        # Also, the window that ended at i-1 in the old word?
        
        # Let's be precise.
        # Old word: ... A B C ... (B at index i)
        # New word: ... A C ...
        # Windows in old word involving B:
        # Start j such that j <= i <= j+n-1 => i-n+1 <= j <= i.
        # These are the windows we removed from counts.
        
        # Windows in new word that are "new" or "changed":
        # Any window in new word that includes the junction.
        # The junction is between new_word[i-1] and new_word[i] (which was old_word[i+1]).
        # So windows starting at j in [i-n+1, i-1] in the NEW word.
        # Note: the range of starts is slightly shifted.
        
        # Let's just identify the set of windows to update.
        # Removed windows: old_word[j:j+n] for j in [max(0, i-n+1), min(i, L-n)]
        # Added windows: new_word[j:j+n] for j in [max(0, i-n+1), min(i-1, L-1-n)]
        # Note: L-1-n is the last valid start index in new_word.
        
        removed_set = []
        for j in range(start_j, end_j + 1):
            w = tuple(old_word[j:j+n])
            if len(set(w)) == n:
                removed_set.append(w)
        
        added_set = []
        new_L = L - 1
        new_start_j = max(0, i - n + 1)
        new_end_j = min(i - 1, new_L - n)
        if new_end_j >= new_start_j:
            for j in range(new_start_j, new_end_j + 1):
                w = tuple(new_word[j:j+n])
                if len(set(w)) == n:
                    added_set.append(w)
        
        # Check if any removed window had count 1
        # If so, removing it makes it disappear, which is invalid.
        for w in removed_set:
            if counts[w] == 1:
                # We cannot remove this symbol because it destroys a unique permutation.
                # Rollback? We haven't modified current_word yet.
                return False
        
        # If we get here, it's safe to remove.
        # Update counts
        for w in removed_set:
            counts[w] -= 1
            if counts[w] == 0:
                del counts[w]
        
        for w in added_set:
            counts[w] = counts.get(w, 0) + 1
        
        # Commit change
        current_word[:] = new_word
        return True

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate through indices in a randomized order
        L = len(current_word)
        indices = list(range(L))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            # The index i might be out of bounds if we've already removed some elements in this pass?
            # No, we restart the pass (break) if we remove an element.
            # So indices are valid at the start of the pass.
            # But wait, if we don't break, and just continue, the indices become stale.
            # So we MUST break and restart if we remove an element.
            
            if try_remove(i):
                improved = True
                break
                
    return "".join(map(str, current_word))
