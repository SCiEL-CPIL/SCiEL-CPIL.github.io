import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # 1. Generate standard recursive superpermutation
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
        
    # Convert to list of ints for easier manipulation
    S = list(word)
    
    # 2. Initialize counts
    # n is small (<=7), so tuples are fine
    counts = {}
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i+n])
        counts[p] = counts.get(p, 0) + 1
        
    best_S = S[:]
    best_len = len(S)
    
    # 3. Local Search / Removal
    # We will try to remove characters.
    # Strategy:
    # 1. Repeatedly scan from right to left (or left to right) and remove any removable character.
    # 2. When stuck, try to "swap" adjacent characters to create a removable character, then remove it.
    #    This is a 2-step move: Swap(i, i+1) then Remove(j).
    #    We need to check if the swap is valid (doesn't break any permutation) and if it enables a removal.
    
    def is_valid_removal(S, counts, i, n):
        """Check if removing S[i] is valid."""
        # Old windows that will be destroyed
        start_old = max(0, i - n + 1)
        end_old = min(i, len(S) - n)
        
        for j in range(start_old, end_old + 1):
            p = tuple(S[j:j+n])
            if counts.get(p, 0) == 1:
                return False
        return True

    def perform_removal(S, counts, i, n):
        """Perform removal of S[i] and update counts."""
        # Decrement old windows
        start_old = max(0, i - n + 1)
        end_old = min(i, len(S) - n)
        
        for j in range(start_old, end_old + 1):
            p = tuple(S[j:j+n])
            counts[p] -= 1
            if counts[p] == 0:
                del counts[p]
        
        # Increment new windows (straddling)
        # New string is S[:i] + S[i+1:]
        # New windows are those that straddle the gap
        # They correspond to windows in the new string starting at k such that k < i < k+n
        # In terms of old S indices, these windows use S[k:k+n] but with S[i] removed.
        # The window in the new string at index k uses:
        # S[k:k+i] + S[i+1:k+i+n] (if k <= i-1)
        # Actually, let's just construct them.
        
        # The new windows that are "new" (didn't exist before) are those that straddle the deletion point.
        # In the new string, these are windows starting at indices k where k < i and k+n > i.
        # i.e., i-n < k < i.
        # So k ranges from max(0, i-n+1) to min(i-1, len(S)-1-n).
        
        start_new = max(0, i - n + 1)
        end_new = min(i - 1, len(S) - 1 - n)
        
        for k in range(start_new, end_new + 1):
            # Construct window from S (before removal)
            # Window in new string at k uses:
            # S[k:k+i] and S[i+1:k+i+n]
            # Wait, the window length is n.
            # It uses indices k ... k+n-1 in the NEW string.
            # Mapping to OLD string:
            # If idx < i: old_idx = idx
            # If idx >= i: old_idx = idx + 1
            
            window_chars = []
            for idx in range(k, k+n):
                s_idx = idx if idx < i else idx + 1
                window_chars.append(S[s_idx])
            
            p_new = tuple(window_chars)
            counts[p_new] = counts.get(p_new, 0) + 1
        
        # Remove from S
        S.pop(i)

    # Main search loop
    while time.time() < deadline:
        # Phase 1: Greedy removal pass
        removed_any = False
        # Try from right to left first, then left to right?
        # Let's alternate or just do one pass.
        # Right to left is often better for this structure.
        for i in range(len(S) - 1, -1, -1):
            if time.time() > deadline:
                break
            if is_valid_removal(S, counts, i, n):
                perform_removal(S, counts, i, n)
                removed_any = True
                best_S = S[:]
                best_len = len(S)
        
        if not removed_any:
            # Phase 2: Try to swap to enable removal
            # We look for a swap (i, i+1) such that:
            # 1. The swap itself is valid (doesn't destroy any unique permutation).
            # 2. After the swap, at least one removal is valid.
            
            # To speed this up, we can just try random swaps or scan.
            # Scanning is O(N^2) which might be slow for N~6000.
            # Let's try a limited number of random swaps or a specific heuristic.
            
            # Heuristic: Try to swap adjacent characters that are "similar" or part of a pattern.
            # Or just try all swaps but with a time limit.
            
            swap_found = False
            # Try a few random swaps
            for _ in range(100): # Limit random attempts
                if time.time() > deadline:
                    break
                i = random.randint(0, len(S) - 2)
                
                # Check if swap (i, i+1) is valid
                # A swap changes windows in the range [i-n+1, i+n-1] roughly.
                # Specifically, windows that contain either i or i+1.
                
                # Let's just perform the swap, check validity, and rollback if invalid.
                # But "validity" of swap means no permutation count drops to 0.
                
                # We need to check all windows affected by the swap.
                # Affected windows are those that include index i or i+1.
                # i.e., start indices j such that j <= i < j+n OR j <= i+1 < j+n.
                # j in [i-n+1, i] U [i-n+2, i+1] = [i-n+1, i+1].
                
                start_check = max(0, i - n + 1)
                end_check = min(i + 1, len(S) - n)
                
                # Check if any of these windows has count 1
                swap_valid = True
                for j in range(start_check, end_check + 1):
                    p = tuple(S[j:j+n])
                    if counts.get(p, 0) == 1:
                        swap_valid = False
                        break
                
                if not swap_valid:
                    continue
                
                # Perform swap
                S[i], S[i+1] = S[i+1], S[i]
                
                # Update counts for affected windows
                # This is complex. We need to decrement old and increment new.
                # Let's just rebuild counts? No, too slow.
                # Let's just check if a removal is now possible without fully updating counts?
                # No, we need accurate counts.
                
                # Given the complexity and time limit, let's skip complex swap logic for now
                # and rely on the greedy removal which might be sufficient for small improvements.
                # Or, we can try a simpler "swap and remove" if we can afford to recalculate counts locally.
                
                # Let's rollback for now to keep it safe.
                S[i], S[i+1] = S[i+1], S[i]
            
            # If we are stuck and time is running out, break.
            # If we have time, we might try more aggressive moves, but for now,
            # if no removal was found in the greedy pass, we are likely at a local minimum.
            # Let's break to avoid wasting time.
            # Actually, let's try one more pass from left to right.
            removed_any = False
            for i in range(len(S)):
                if time.time() > deadline:
                    break
                if is_valid_removal(S, counts, i, n):
                    perform_removal(S, counts, i, n)
                    removed_any = True
                    best_S = S[:]
                    best_len = len(S)
            
            if not removed_any:
                # Stuck
                # Try to break the local minimum with a random perturbation?
                # Or just break.
                # Let's break to save time for the next evaluation or just return.
                # Since we are in a single construct call, we just return best_S.
                break

    return "".join(map(str, best_S))
