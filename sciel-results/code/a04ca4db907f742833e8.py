import time
import itertools
import sys
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (evaluator kills at 150s)

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1312131"
    if n == 4:
        return "1413121413124131241"

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
        for j in range(start_min, start_max + 1):
            # Determine start index in new string
            if j < i:
                new_j = j
            else:
                new_j = j - 1
            
            # Construct the new window from S
            window_chars = []
            for x in range(new_j, new_j + n):
                s_idx = x if x < i else x + 1
                window_chars.append(S[s_idx])
            
            w_new = tuple(window_chars)
            deltas[w_new] = deltas.get(w_new, 0) + 1
            
        return True, S[:i] + S[i+1:], deltas

    # Pruning Loop
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
                        perm_counts[p] = delta
                
                removed_in_pass = True
                # Continue scanning from i-1 (next iteration)
                # Since we are iterating downwards, the next i is i-1.
                
        if not removed_in_pass:
            # Stuck. No more single deletions possible.
            break
            
    # 3. Local Search: 2-Opt style swaps
    # Try to swap adjacent elements or small blocks to create new overlaps.
    # Since we are stuck on deletions, we try to rearrange to make deletions possible.
    
    import random
    random.seed(42)
    
    best_S = S
    best_len = len(S)
    
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
