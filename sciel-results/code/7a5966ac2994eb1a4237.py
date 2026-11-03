import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # 1. Generate standard superpermutation
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
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n < 3:
        return "".join(map(str, word))

    # Precompute permutation to index mapping
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts
    counts = [0] * num_perms
    L = len(word)
    
    def build_counts(w):
        c = [0] * num_perms
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            if p in perm_to_idx:
                c[perm_to_idx[p]] += 1
        return c

    counts = build_counts(word)
    
    # Helper to check if a specific window tuple is a valid permutation and get its index
    # We assume all windows in a superperm are valid permutations initially, 
    # but after deletions/swaps, some windows might be invalid (repeated chars).
    # However, our deletion logic only removes chars, it doesn't create invalid windows 
    # unless the "new" windows formed have duplicates.
    # Wait, if we delete a char, the new windows are formed by existing chars.
    # If the original word was a superperm, all chars are distinct in any window of size n.
    # After deletion, a window of size n in the new word is a subsequence of length n 
    # from the old word that skips one char.
    # It is possible that the new window has duplicate characters?
    # Example: 1 2 1 2 3. Delete 2. 1 1 2 3. Window 1 1 2 3 is invalid.
    # So we MUST check if the new windows are valid permutations.
    
    # We will handle this in the incremental check.

    def try_delete(k, w, c):
        """
        Simulates deletion of index k.
        Returns (is_valid, new_counts, new_covered)
        """
        L = len(w)
        # 1. Identify old windows containing k
        # Windows start at j such that j <= k <= j+n-1
        # j_min = k - n + 1, j_max = k
        j_min = max(0, k - n + 1)
        j_max = min(L - n, k)
        
        # Decrement counts for these windows
        # We need to be careful: if a window is invalid (not in perm_to_idx), it wasn't counted.
        # So we only decrement if it was counted.
        
        new_c = c[:] # Copy is O(num_perms) = 5040. Acceptable?
        # 5040 * 6000 deletions * 1000 passes = 30 billion ops? No.
        # We only do this if we want to commit.
        # For checking, we can just track the changes.
        
        # Let's track deltas.
        # affected_old_windows = []
        # affected_new_windows = []
        
        # Decrement old windows
        valid_deletion = True
        # We need to know if any count goes to 0.
        # Let's just simulate on a copy of counts?
        # 5040 ints copy is fast.
        
        # Actually, let's just do the copy.
        temp_counts = c[:]
        
        for j in range(j_min, j_max + 1):
            p = tuple(w[j:j+n])
            idx = perm_to_idx.get(p)
            if idx is not None:
                temp_counts[idx] -= 1
                if temp_counts[idx] == 0:
                    # We can't early exit easily because we need to restore? 
                    # No, if any count hits 0, it's invalid.
                    # But we might have other windows that also decrement.
                    # Just mark invalid and break?
                    # Yes, if any permutation count becomes 0, the word is not a superperm.
                    valid_deletion = False
                    # We can break here? Yes, we don't need exact counts if invalid.
                    break

        if not valid_deletion:
            return False, c, sum(1 for x in c if x > 0) # Return original counts
        
        # 2. Identify new windows formed by deletion
        # The new word w' has length L-1.
        # The junction is between w[k-1] and w[k+1].
        # New windows are those starting at j' in w' such that they include the junction.
        # In w', the characters are:
        # 0 .. k-1 : w[0] .. w[k-1]
        # k .. L-2 : w[k+1] .. w[L-1]
        
        # A window in w' starting at j' covers w'[j' .. j'+n-1].
        # It is "new" if it wasn't a window in w.
        # Windows in w that are preserved are those not containing k.
        # So we only need to add counts for windows in w' that DO contain the junction.
        
        # Which j' in w' contain the junction?
        # The junction is between index k-1 and k in w'.
        # So j' <= k-1 and j'+n-1 >= k.
        # j' <= k-1 and j' >= k - n + 1.
        
        j_prime_min = max(0, k - n + 1)
        j_prime_max = min(L - 1 - n, k - 1) # Max start index in w' is L-1-n
        
        for j_prime in range(j_prime_min, j_prime_max + 1):
            # Construct the window in w'
            # Part 1: w[j_prime : k] (if j_prime < k)
            # Part 2: w[k+1 : j_prime+n] (if k+1 < j_prime+n)
            
            # Note: indices in w'
            # w'[i] = w[i] for i < k
            # w'[i] = w[i+1] for i >= k
            
            # Window covers w'[j_prime] to w'[j_prime+n-1]
            
            # Map to w indices:
            # If i < k, w_idx = i
            # If i >= k, w_idx = i+1
            
            # The window in w' corresponds to a subsequence of w.
            # We need to check if this subsequence is a valid permutation.
            
            # Let's build the tuple
            # It consists of:
            # w[j_prime : k] + w[k+1 : j_prime+n]
            # Wait, length must be n.
            # (k - j_prime) + (j_prime + n - k - 1) ?
            # Let's check lengths.
            # Left part: w[j_prime] to w[k-1]. Length k - j_prime.
            # Right part: w[k+1] to w[j_prime+n-1+1] = w[j_prime+n]. Length j_prime+n - k - 1.
            # Total length: k - j_prime + j_prime + n - k - 1 = n - 1.
            # Missing 1 char?
            
            # Let's re-evaluate.
            # w' indices: j_prime ... j_prime+n-1
            # If j_prime+n-1 < k: all left. (But we said it contains junction, so j_prime+n-1 >= k)
            # If j_prime >= k: all right. (But j_prime <= k-1)
            
            # So it spans the junction.
            # Left part in w: j_prime to k-1. (Count: k - j_prime)
            # Right part in w: k+1 to j_prime+n. (Count: j_prime+n - k)
            # Total: n.
            
            left_part = w[j_prime:k]
            right_part = w[k+1:j_prime+n+1]
            
            # Note: if j_prime == k-1, left_part is w[k-1:k] (len 1).
            # if j_prime+n-1 == k, right_part is w[k+1:k+1] (len 0).
            
            # Construct tuple
            # Since n is small (7), we can just concatenate lists
            new_window_list = left_part + right_part
            p_new = tuple(new_window_list)
            
            idx = perm_to_idx.get(p_new)
            if idx is not None:
                temp_counts[idx] += 1
            else:
                # If the new window is invalid (e.g. duplicates), it doesn't contribute to counts.
                # This is fine, it just means we lost a valid permutation (if it was the only one)
                # or we gained an invalid window (which is ignored).
                # But wait, if we lose a valid permutation, its count might drop to 0?
                # We already decremented the old windows.
                # If the old window was valid and the new one is invalid, we effectively lost it.
                # This is handled by the decrement step.
                pass

        # Check if any count is 0
        # We can check min(temp_counts) == 0?
        # Or sum(1 for x in temp_counts if x == 0) == 0?
        # Since we only changed O(n) entries, we can just check those?
        # No, other counts are unchanged.
        # We only need to check if ANY count is 0.
        # If the original counts had no 0s, and we only decremented/incremented some,
        # the only way to get a 0 is if we decremented a count that was 1.
        
        # So we just need to check the indices we touched.
        # But we didn't store them.
        # Easier: just check if min(temp_counts) == 0.
        # min() on 5040 ints is fast.
        
        if min(temp_counts) == 0:
            return False, c, sum(1 for x in c if x > 0)
        
        return True, temp_counts, sum(1 for x in temp_counts if x > 0)

    # Main optimization loop
    while time.time() < deadline:
        L = len(word)
        if L < n:
            break
            
        improved = False
        # Try to delete
        for k in range(L):
            if time.time() > deadline - 5:
                break
            
            valid, new_counts, new_covered = try_delete(k, word, counts)
            if valid:
                # Perform deletion
                word = word[:k] + word[k+1:]
                counts = new_counts
                improved = True
                break # Restart scan
        
        if not improved:
            # Try swaps
            swapped = False
            for i in range(L - 1):
                if time.time() > deadline - 5:
                    break
                # Swap i and i+1
                word[i], word[i+1] = word[i+1], word[i]
                
                # Check validity
                # Rebuild counts? O(L).
                # Or incremental?
                # Swapping affects windows containing i or i+1.
                # Similar to deletion but complex.
                # Given L is small, let's just rebuild.
                new_counts = build_counts(word)
                
                if min(new_counts) > 0:
                    counts = new_counts
                    swapped = True
                    break
                else:
                    # Revert
                    word[i], word[i+1] = word[i+1], word[i]
            
            if not swapped:
                # Try 2-swaps?
                # For now, stop if no single swap works.
                # We could add 2-swaps if time permits.
                break

    return "".join(map(str, word))
