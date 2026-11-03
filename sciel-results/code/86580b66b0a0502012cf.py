"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

# EVOLVE-BLOCK-START
def construct(n):
    if n == 1:
        return "1"
    
    deadline = time.time() + 140  # Safety margin before 150s limit

    # 1. Generate standard superpermutation of length 1! + ... + n!
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all unique (m-1)-permutations in order of appearance
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap with the end of 'new'
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, 0, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    # For n <= 6, the standard construction is often optimal or close to it, 
    # and local search might be overkill or risky if time is tight, 
    # but for n=7 we definitely want to try to improve.
    if n <= 6:
        return "".join(map(str, word))

    # 2. Setup for optimized local search
    # Map permutations to IDs for fast counting
    from itertools import permutations
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts of each permutation in the current word
    counts = [0] * num_perms
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        pid = perm_to_id.get(p)
        if pid is not None:
            counts[pid] += 1
            
    # Check if valid initially (should be)
    if any(c == 0 for c in counts):
        return "".join(map(str, word))

    # Local Search: Repeatedly try to delete characters
    # We iterate until no more deletions are possible or deadline is reached
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate backwards so indices remain valid after deletion
        for i in range(len(word) - 1, -1, -1):
            if time.time() > deadline:
                break
                
            # Only consider deletion if it doesn't remove the last occurrence of a perm
            # Optimization: If the character at i is part of a unique window, 
            # we can skip, but checking all windows is O(n).
            
            # Attempt deletion at index i
            # Affected old windows: start k in [max(0, i-n+1), min(i, len(word)-n)]
            # Note: len(word) is L. Last start index is L-n.
            # k must be such that k <= i <= k+n-1 => i-n+1 <= k <= i
            # Also k >= 0 and k <= L-n
            
            L = len(word)
            start_k = max(0, i - n + 1)
            end_k = min(i, L - n)
            
            # Store changes to revert if invalid
            changes = [] # list of (pid, delta)
            
            valid = True
            
            # 1. Decrement counts for old windows
            for k in range(start_k, end_k + 1):
                p = tuple(word[k:k+n])
                pid = perm_to_id.get(p)
                if pid is not None:
                    counts[pid] -= 1
                    changes.append((pid, -1))
                    if counts[pid] < 0:
                        # This shouldn't happen if counts are correct, 
                        # but if it does, revert immediately
                        for cid, delta in changes:
                            counts[cid] -= delta
                        valid = False
                        break
                    if counts[pid] == 0:
                        # This permutation is now lost. 
                        # We must check if it is restored by new windows.
                        # We don't fail yet, we just mark it as "at risk"
                        pass
            
            if not valid:
                continue

            # 2. Increment counts for new windows
            # New word is word with i removed.
            # New windows affected are those starting at k' in [start_k, end_k] in the NEW word?
            # Actually, the set of window *contents* changes.
            # Let's just calculate the new windows in the affected region of the NEW word.
            # The deletion shifts indices.
            # Old indices < i are unchanged.
            # Old indices >= i are shifted left by 1.
            
            # New windows that include the "gap" at i:
            # A new window starts at k' in the new word.
            # If k' < i:
            #   The window covers new_word[k':k'+n].
            #   Since i is removed, this corresponds to old_word[k':i] + old_word[i+1:k'+n+1].
            #   This window exists in the new word if k' <= i-1 (in new word coords, which is same as old for k'<i)
            #   and k'+n > i-1 (so it crosses the deletion point).
            #   Actually, simpler: 
            #   The windows in the new word that are *different* from old word are those that span the deletion point.
            #   In the new word, the deletion point is between index i-1 and i (if we think of the gap).
            #   Wait, let's just construct the new windows for k' in range(start_k, end_k+1) carefully.
            
            # The range of start indices in the NEW word that are affected:
            # The deletion is at index i in OLD word.
            # In NEW word, indices < i are same. Indices >= i are old_index - 1.
            # The "seam" is between new index i-1 and i.
            # Windows crossing this seam start at k' in [i-n, i-1] in NEW word.
            # Also, windows starting at k' >= i in NEW word are just old windows starting at k'+1.
            # But we already handled "old windows" by decrementing.
            # We need to increment the *new* windows.
            
            # Let's define the affected range in NEW word coordinates.
            # New length is L-1.
            # Max start index in new word is (L-1) - n = L - n - 1.
            
            new_start_k = max(0, i - n) # In new word, the gap is after i-1. So windows starting at i-n ... i-1 cross it?
            # Let's verify.
            # If i=0, new word starts at old 1.
            # If i=L-1, new word ends at old L-2.
            
            # Let's just iterate k' from max(0, i-n) to min(i-1, L-n-1) for windows crossing the seam?
            # And what about windows that were "shifted"?
            # Actually, it is safer to just re-calculate the windows in the range [start_k, end_k] 
            # but mapping them to the new word structure.
            
            # Alternative: 
            # The new windows that appear are those starting at k' in [max(0, i-n), i-1] (crossing the gap) 
            # PLUS the windows that were previously starting at k+1 for k in [start_k, end_k]?
            # No, that's getting confusing.
            
            # Let's use the explicit construction of the new window content.
            # For each potential start index k' in the NEW word that could be different from any old window:
            # The only windows that change are those that include the position where the deletion happened.
            # In the OLD word, position i was deleted.
            # In the NEW word, the characters at indices < i are the same.
            # The characters at indices >= i are shifted left.
            
            # Windows in NEW word starting at k':
            # If k' < i:
            #   The window is new_word[k':k'+n].
            #   This consists of old_word[k':i] (length i-k') and old_word[i+1:k'+n+1] (length n-(i-k')).
            #   This window is "new" if it wasn't identical to an old window? 
            #   Actually, we just need to add its count.
            #   We must ensure we don't double count or miss.
            #   The old windows we decremented were those starting at k in [start_k, end_k].
            #   The new windows we should increment are those starting at k' in the NEW word 
            #   that occupy the same "physical space" in the sequence, i.e., they cover the deletion site.
            
            # Let's iterate k' from max(0, i-n) to min(i-1, L-n-1).
            # These are the windows in the new word that span the junction between old[i-1] and old[i+1].
            
            new_end_k = min(i - 1, L - n - 1)
            
            for k_prime in range(max(0, i - n), new_end_k + 1):
                # Construct the new window
                # Part 1: new_word[k_prime : i] which is old_word[k_prime : i]
                # Part 2: new_word[i : k_prime+n] which is old_word[i+1 : k_prime+n+1]
                
                w1 = word[k_prime:i]
                w2 = word[i+1:k_prime+n+1]
                p = tuple(w1 + w2)
                pid = perm_to_id.get(p)
                if pid is not None:
                    counts[pid] += 1
                    changes.append((pid, 1))
            
            # Check if any count is 0
            # We only need to check the pids that were touched?
            # No, a pid could have been 1, decremented to 0, and NOT incremented back.
            # So we must check if ANY count is 0.
            # Since we only touched a small set of pids, we can check if any of the touched pids is 0?
            # No, if a pid was 1, and we decremented it, it became 0. If we didn't increment it back, it stays 0.
            # So yes, we only need to check the pids in `changes`?
            # Wait, if a pid was 2, decremented to 1, it's fine.
            # If a pid was 1, decremented to 0, and then incremented to 1, it's fine.
            # If a pid was 1, decremented to 0, and NOT incremented, it's 0.
            # So we just need to check if any `counts[pid]` for `pid` in `changes` is 0?
            # Actually, if a pid was not in `changes`, its count didn't change, so if it was >0, it's still >0.
            # So we only need to check the pids in `changes`.
            
            for pid, _ in changes:
                if counts[pid] == 0:
                    # Invalid deletion
                    # Revert
                    for cid, delta in changes:
                        counts[cid] -= delta
                    valid = False
                    break
            
            if valid:
                # Perform deletion
                word.pop(i)
                improved = True
                # Continue loop, next i will be adjusted by range? 
                # We are iterating backwards with range, so indices are fine.
                # However, if we delete i, the next iteration is i-1.
                # The word length decreased by 1.
                # This is safe.
            else:
                pass # Already reverted

    return "".join(map(str, word))
# EVOLVE-BLOCK-END
