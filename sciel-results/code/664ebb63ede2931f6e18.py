import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    if n == 1:
        return "1"
    
    # 1. Generate standard superpermutation
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
            
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
    
    S = list(word)
    
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Initialize counts
    perm_counts = {}
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i + n])
        perm_counts[p] = perm_counts.get(p, 0) + 1
        
    def is_valid_incremental(S, i):
        """
        Check if removing S[i] keeps all permutations.
        Returns (is_valid, new_S, new_counts) if valid, else (False, None, None).
        """
        L = len(S)
        if L <= n:
            return False, None, None
        
        # Windows affected:
        # Old windows that contain index i: start from max(0, i-n+1) to min(i, L-n)
        # These windows will be removed.
        # New windows that bridge the gap: start from max(0, i-n+1) to min(i-1, L-n-1)
        # Wait, if we remove S[i], the new string is S' = S[:i] + S[i+1:]
        # The windows in S' that are different from S are those that cross the boundary i-1 | i.
        # In S', the index i is now S[i+1].
        # The windows in S' starting at j (where j < i) that end after i-1 in S' are the new ones.
        
        # Let's just compute the affected windows in the NEW string S'
        # and compare with the OLD string S.
        
        # Actually, easier:
        # 1. Identify old windows to remove.
        # 2. Identify new windows to add.
        # 3. Check if any permutation count drops to 0.
        
        old_windows_to_remove = []
        # In S, windows that contain index i
        start_min = max(0, i - n + 1)
        start_max = min(i, len(S) - n)
        for j in range(start_min, start_max + 1):
            old_windows_to_remove.append(tuple(S[j:j+n]))
            
        new_windows_to_add = []
        # In S', the removal of S[i] shifts S[i+1:] left.
        # New windows are formed at indices j in S' such that the window in S'
        # corresponds to a window in S that was broken or a new combination.
        # The new windows in S' are at indices j from max(0, i-n+1) to min(i-1, len(S')-n)
        # Let's just generate the specific new windows.
        # The gap is between i-1 and i+1 (in S).
        # New windows in S' start at j in [i-n+1, i-1].
        # For each such j, the window in S' is S'[j:j+n].
        # S'[j:j+n] corresponds to S[j:i] + S[i+1:i+n].
        # Note: if j == i, the window in S' is S'[i:i+n] = S[i+1:i+n+1], which was a window in S starting at i+1.
        # So windows starting at j >= i in S' are the same as windows starting at j+1 in S.
        # Windows starting at j < i in S' are the "new" ones if they cross the gap.
        
        # Let's just construct the new string slice and extract windows?
        # No, that's O(n^2) or O(n).
        
        # Let's do it explicitly.
        # New windows start at j in S'.
        # If j >= i, S'[j:j+n] = S[j+1:j+1+n]. (Same as old window at j+1)
        # If j < i, S'[j:j+n] = S[j:i] + S[i+1:i+n-j+ (n - (i-j))]?
        # Length of S[j:i] is i-j.
        # Length of S[i+1:...] needed is n - (i-j).
        # So S'[j:j+n] = S[j:i] + S[i+1:i+1+n-(i-j)] = S[j:i] + S[i+1:j+n+1].
        
        new_windows_to_add = []
        for j in range(max(0, i - n + 1), i):
            # Check if this window index j is valid in S'
            # S' length is L-1. Max start index is L-1-n.
            if j > len(S) - 1 - n:
                continue
            # Construct the tuple
            # S[j:i] + S[i+1:j+n+1]
            part1 = S[j:i]
            part2 = S[i+1:j+n+1]
            new_win = tuple(part1 + part2)
            new_windows_to_add.append(new_win)
            
        # Now update counts virtually
        # We need to check if removing old_windows and adding new_windows
        # results in any perm_count becoming 0.
        
        # Get current counts for affected perms
        # We can just simulate the change on a copy of the counts for affected keys?
        # Or just check the minimum count among the removed windows?
        # No, because a window might be removed and a different window (same perm) added.
        
        # Let's collect the delta for each permutation.
        deltas = {}
        for w in old_windows_to_remove:
            deltas[w] = deltas.get(w, 0) - 1
        for w in new_windows_to_add:
            deltas[w] = deltas.get(w, 0) + 1
            
        # Check validity
        for p, delta in deltas.items():
            if p in all_perms:
                new_count = perm_counts.get(p, 0) + delta
                if new_count < 0:
                    return False, None, None
                if new_count == 0:
                    # If it becomes 0, it's invalid because we need at least 1
                    # Wait, if it was 1 and we remove it, and don't add it, it becomes 0. Invalid.
                    # If it was 1, we remove it, and add it (delta 0), it stays 1. Valid.
                    # If it was 1, we remove it (delta -1), it becomes 0. Invalid.
                    # So if new_count < 1, invalid.
                    return False, None, None
                    
        # If we reach here, it's valid.
        # We need to return the new S and update perm_counts.
        # But we don't want to modify perm_counts until we are sure.
        # The caller will commit.
        # So we return True, and the caller updates.
        # But the caller needs to know the deltas to update perm_counts.
        # Or the caller can just re-calculate the counts for the affected windows?
        # Easier: Return the new S. The caller updates perm_counts by applying deltas.
        
        return True, S[:i] + S[i+1:], deltas

    # Pruning Loop
    for _pass in range(100):
        if time.time() > deadline - 5:
            break
        removed = False
        for i in range(len(S) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            if len(S) <= n:
                break
            valid, new_S, deltas = is_valid_incremental(S, i)
            if valid:
                S = new_S
                # Update perm_counts
                for p, delta in deltas.items():
                    perm_counts[p] = perm_counts.get(p, 0) + delta
                removed = True
                break
        if not removed:
            break
            
    return "".join(map(str, S))
