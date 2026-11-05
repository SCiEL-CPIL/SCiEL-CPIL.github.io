import time
import itertools
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (evaluator kills at 150s)

    if n == 1:
        return "1"
    if n == 2:
        return "121"

    # 1. Generate standard superpermutation
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline - 5:
            break
        # Extract unique (m-1)-permutations from word
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build new word
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
    
    S = list(word)
    
    # 2. Initial Validation and Count Setup
    all_perms = set(itertools.permutations(range(1, n + 1)))
    perm_counts = {}
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i + n])
        perm_counts[p] = perm_counts.get(p, 0) + 1
        
    # Optimization: Precompute a set of all valid permutations for O(1) lookup
    # We already have all_perms.
    
    # Helper to check if removal at index i is valid
    # Returns (is_valid, new_S, deltas)
    def check_removal(S, i, perm_counts):
        L = len(S)
        if L <= n:
            return False, None, None
        
        # Windows that will be destroyed (removed)
        # A window starting at j is destroyed if it includes index i
        # i.e., j <= i < j+n  =>  i-n+1 <= j <= i
        start_min = max(0, i - n + 1)
        start_max = min(i, L - n)
        
        deltas = {}
        valid = True
        
        # Check old windows
        for j in range(start_min, start_max + 1):
            w = tuple(S[j:j+n])
            if w in all_perms: # Only care about valid perms for validity
                if perm_counts.get(w, 0) == 1:
                    valid = False
                    break
                else:
                    deltas[w] = deltas.get(w, 0) - 1
        
        if not valid:
            return False, None, None
            
        # Check new windows (formed by joining S[:i] and S[i+1:])
        # New windows start at j in [i-n+1, i-1] relative to new string?
        # Let's map indices carefully.
        # New string S' = S[:i] + S[i+1:]
        # A window in S' at index k is formed by S'[k:k+n]
        # If k < i: S'[k:k+n] = S[k:k+n] (if k+n <= i) or straddles (if k+n > i)
        # If k >= i: S'[k:k+n] = S[k+1:k+1+n]
        
        # The straddling windows are those where the window crosses the deletion point i.
        # In S', the window starts at k such that it includes the join.
        # The join is between S[i-1] and S[i+1].
        # The windows that change are those that included S[i].
        # These are windows starting at j in [i-n+1, i] in the OLD string.
        # In the NEW string, the corresponding windows start at j' where:
        # If j < i, j' = j.
        # If j >= i, j' = j - 1.
        
        # We already checked the old windows for validity (counts > 1).
        # Now we need to add the counts for the new windows that were formed.
        # Note: The new windows are distinct from the old ones.
        
        for j in range(start_min, start_max + 1):
            # Determine start index in new string
            if j < i:
                new_j = j
            else:
                new_j = j - 1
            
            # Construct the new window from S
            # It spans from new_j to new_j+n in S'
            # Map back to S indices:
            # S' index x corresponds to S index x if x < i, else x+1
            window_chars = []
            for x in range(new_j, new_j + n):
                s_idx = x if x < i else x + 1
                window_chars.append(S[s_idx])
            
            w_new = tuple(window_chars)
            # We don't need to check validity for new windows (they only add to count)
            # But we must record the delta
            deltas[w_new] = deltas.get(w_new, 0) + 1
            
            # Wait, if w_new is the same as an old window that was decremented,
            # the net delta might be 0.
            
        # Optimization: If a window is removed and the same window is added, net delta is 0.
        # This can happen if the window content doesn't change?
        # If we remove S[i], the windows change.
        # Example: S = ... A B C ... remove B.
        # Old window: A B C
        # New window: A C ?
        # They are different tuples.
        
        # One edge case: if the new window is identical to an old window that was NOT destroyed?
        # No, we only process windows that were destroyed.
        # The new windows are formed *in place* of the destroyed ones.
        
        return True, S[:i] + S[i+1:], deltas

    # Pruning Loop
    # Strategy: Scan from right to left.
    # If a removal is valid, perform it and restart the scan from the same index (or i-1).
    # Restarting from i-1 is safe and allows finding multiple removals in a cluster.
    
    pass_count = 0
    while time.time() < deadline - 2:
        removed_in_pass = False
        # Scan from right to left
        for i in range(len(S) - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            
            if len(S) <= n:
                break
                
            valid, new_S, deltas = check_removal(S, i, perm_counts)
            
            if valid:
                # Apply removal
                S = new_S
                for p, delta in deltas.items():
                    if delta == 0:
                        continue
                    if p in perm_counts:
                        perm_counts[p] += delta
                        if perm_counts[p] == 0:
                            del perm_counts[p]
                    else:
                        # If p was not in counts (count 0), and delta > 0, add it
                        # If delta < 0, it should have been in counts.
                        perm_counts[p] = delta
                
                removed_in_pass = True
                # Continue scanning from i-1 (next iteration)
                # Since we are iterating downwards, the next i is i-1.
                # This is safe because indices < i are unchanged.
                # Indices > i are shifted, but we've already passed them.
                # However, removing i might make a removal at i-1 valid that wasn't before?
                # Yes. So continuing to i-1 is good.
                
        if not removed_in_pass:
            # Stuck. No more single deletions possible.
            break
            
    # 3. Local Search: 2-Opt style swaps
    # Try to swap adjacent elements or small blocks to create new overlaps.
    # Since we are stuck on deletions, we try to rearrange to make deletions possible.
    # A simple heuristic: try swapping S[i] and S[i+1].
    # If the swap is valid (all perms still covered), and it allows a deletion nearby, we might improve.
    # But checking validity of a swap is expensive.
    # Alternative: Random restarts of the deletion process? No, we have limited time.
    
    # Let's try a few random swaps in the latter half of the string.
    import random
    random.seed(42)
    
    best_S = S
    best_len = len(S)
    
    # We will try to improve by swapping pairs (i, i+1) or (i, i+2)
    # and checking if it allows more deletions.
    # To save time, we only check if the swap itself is valid (doesn't remove unique perms).
    # Then we run a quick deletion pass.
    
    # Precompute a helper for fast validity check of a swap
    def check_swap_validity(S, i, j, perm_counts, all_perms):
        # Swap S[i] and S[j]
        # Windows affected are those containing i or j.
        # If |i-j| is small, the set of affected windows is small.
        # For now, only handle adjacent swaps (j = i+1).
        if j != i + 1:
            return False
        
        L = len(S)
        if i < 0 or i+1 >= L:
            return False
            
        # Affected windows: start in [i-1, i] (for i) and [i, i+1] (for i+1)
        # Union: [i-1, i+1]
        starts = set()
        for k in range(max(0, i - n + 1), min(L - n, i + 1) + 1):
            starts.add(k)
            
        # Check if any affected window is unique and would be destroyed
        # We need to simulate the swap.
        # S' = S[:i] + [S[i+1], S[i]] + S[i+2:]
        
        # Old windows
        old_windows = {}
        for k in starts:
            w = tuple(S[k:k+n])
            if w in all_perms:
                old_windows[w] = old_windows.get(w, 0) + 1
                
        # New windows
        new_windows = {}
        for k in starts:
            # Construct new window
            # S' index x:
            # if x < i: S[x]
            # if x == i: S[i+1]
            # if x == i+1: S[i]
            # if x > i+1: S[x]
            w_chars = []
            for x in range(k, k+n):
                if x < i:
                    w_chars.append(S[x])
                elif x == i:
                    w_chars.append(S[i+1])
                elif x == i+1:
                    w_chars.append(S[i])
                else:
                    w_chars.append(S[x])
            w_new = tuple(w_chars)
            if w_new in all_perms:
                new_windows[w_new] = new_windows.get(w_new, 0) + 1
                
        # Check validity:
        # For every perm in old_windows, if count == 1, it must appear in new_windows with count >= 1.
        # Actually, we need to check global counts.
        # If a perm has global count 1, and it is in old_windows (so it is affected),
        # it must be in new_windows.
        
        for w, cnt in old_windows.items():
            if perm_counts.get(w, 0) == 1:
                if w not in new_windows:
                    return False
            # If count > 1, it's fine as long as it doesn't become 0.
            # But since we are just swapping, the total number of windows is the same.
            # If a perm is removed from one window and added to another, count stays same.
            # If a perm is removed and NOT added, count decreases by 1.
            # If count was 1, it becomes 0 -> Invalid.
            # If count was >1, it becomes count-1 -> Valid.
            
        return True

    # We will try a limited number of swaps
    # Focus on the tail end where the standard construction is less efficient?
    # Or random locations.
    
    # To ensure we don't spend too much time, we limit the number of swap attempts.
    # Each swap attempt involves a check (O(n) roughly) and if valid, a deletion pass (O(L*n)).
    # If the deletion pass finds a deletion, we keep the swap.
    
    # Let's try 500 random swaps.
    for _ in range(500):
        if time.time() > deadline - 2:
            break
            
        i = random.randint(0, len(S) - 2)
        
        if check_swap_validity(S, i, i+1, perm_counts, all_perms):
            # Perform swap
            S[i], S[i+1] = S[i+1], S[i]
            
            # Update perm_counts?
            # It's complex to update incrementally for a swap.
            # Instead, we can just run the deletion loop on the new S.
            # But we need accurate perm_counts for the deletion loop.
            # Recalculating perm_counts takes O(L*n).
            # If we do this for 500 swaps, it might be too slow.
            
            # Alternative: Keep perm_counts consistent.
            # The swap changes the windows in `starts`.
            # We can update perm_counts by removing old windows and adding new ones.
            
            # Let's do that.
            # First, revert the swap in S to calculate old windows? No, S is already swapped.
            # So S currently has the new configuration.
            # We need to subtract the new windows and add the old windows?
            # No, we want to update perm_counts to reflect the new S.
            
            # Current perm_counts reflects S_before_swap.
            # We need perm_counts to reflect S_after_swap.
            
            # Let's re-calculate the deltas.
            # Old windows (from S_before_swap) are in `old_windows` from check_swap_validity?
            # No, check_swap_validity used S_after_swap to calculate new_windows, and S_before_swap implicitly for old?
            # In check_swap_validity, I calculated `old_windows` from `S` which was passed in.
            # Wait, in the loop above:
            # `S` is the current list.
            # I call `check_swap_validity(S, i, i+1, ...)`.
            # Inside, I construct `old_windows` from `S`. But `S` is the state BEFORE swap?
            # No, I am checking validity BEFORE swapping.
            # So `S` inside check_swap_validity is the original S.
            # So `old_windows` are from original S.
            # And `new_windows` are constructed by simulating the swap on original S.
            
            # So if valid:
            # 1. Update perm_counts:
            #    Subtract old_windows
            #    Add new_windows
            # 2. Swap S[i], S[i+1]
            
            # Let's implement this carefully.
            
            # Re-fetch old/new windows?
            # It's safer to just recalculate perm_counts if we are unsure.
            # Given L ~ 6000, n=7. L*n = 42000 ops.
            # 500 swaps * 42000 = 21 million ops.
            # Python can do ~10M simple ops/sec? Maybe 50M?
            # 21M might take ~0.5s.
            # Plus deletion pass.
            # This seems feasible.
            
            # Let's just recalculate perm_counts for simplicity and safety.
            new_perm_counts = {}
            for k in range(len(S) - n + 1):
                # S is still original (not swapped yet)
                pass
            
            # Wait, I haven't swapped S yet.
            # So:
            S[i], S[i+1] = S[i+1], S[i]
            
            # Recalculate perm_counts
            perm_counts = {}
            for k in range(len(S) - n + 1):
                p = tuple(S[k:k+n])
                perm_counts[p] = perm_counts.get(p, 0) + 1
                
            # Run deletion pass
            removed = False
            for j in range(len(S) - 1, -1, -1):
                if time.time() > deadline - 2:
                    break
                valid, new_S, deltas = check_removal(S, j, perm_counts)
                if valid:
                    S = new_S
                    for p, delta in deltas.items():
                        if delta == 0: continue
                        if p in perm_counts:
                            perm_counts[p] += delta
                            if perm_counts[p] == 0: del perm_counts[p]
                        else:
                            perm_counts[p] = delta
                    removed = True
                    # Continue from j-1
                    # The for loop will go to j-1 automatically.
                    # But indices shifted.
                    # If we removed at j, indices < j are same.
                    # So continuing to j-1 is correct.
                    
            if removed:
                # Keep the swap and deletions
                pass
            else:
                # No deletion found, revert swap?
                # If we revert, we need to revert perm_counts too.
                # It's easier to just keep the swap if it didn't break validity,
                # even if no deletion was found immediately.
                # It might enable a future deletion.
                # However, if we keep a swap that doesn't reduce length,
                # we might wander.
                # Let's revert if no deletion was found to avoid drift.
                S[i], S[i+1] = S[i+1], S[i]
                # Recalculate perm_counts again?
                # Or just revert perm_counts?
                # Reverting perm_counts is hard.
                # Let's just recalculate.
                perm_counts = {}
                for k in range(len(S) - n + 1):
                    p = tuple(S[k:k+n])
                    perm_counts[p] = perm_counts.get(p, 0) + 1

    return "".join(map(str, S))

# Test locally if needed
if __name__ == "__main__":
    # Small tests
    print(construct(3))
    print(len(construct(3)))
    print(construct(4))
    print(len(construct(4)))
    # print(construct(7))
    # print(len(construct(7)))
