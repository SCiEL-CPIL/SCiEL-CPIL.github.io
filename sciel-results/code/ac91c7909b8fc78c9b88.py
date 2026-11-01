import time
import random
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 145.0
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # Build standard superpermutation
    word = [1]
    for m in range(2, n + 1):
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
    
    if n <= 4:
        return "".join(map(str, word))
    
    target_count = 1
    for i in range(2, n + 1):
        target_count *= i
        
    # Precompute permutation IDs for efficiency
    # Map tuple of symbols to integer ID
    perm_id = {}
    id_to_perm = []
    counter = 0
    for p in range(target_count):
        # Generate all permutations
        pass
    
    # Efficiently generate permutations and map them
    from itertools import permutations
    all_perms = list(permutations(range(1, n+1)))
    for p in all_perms:
        perm_id[p] = counter
        id_to_perm.append(p)
        counter += 1
    
    current_word = list(word)
    L = len(current_word)
    
    # Initialize counts array
    counts = [0] * target_count
    
    # Helper to get ID of window at i, or -1 if invalid
    def get_window_id(i):
        # Window is current_word[i : i+n]
        # Check if all distinct
        # Since n is small (7), we can just check set length
        w = tuple(current_word[i:i+n])
        if len(set(w)) == n:
            return perm_id.get(w, -1)
        return -1

    # Build initial counts
    for i in range(L - n + 1):
        wid = get_window_id(i)
        if wid != -1:
            counts[wid] += 1
            
    # Verify initial state
    # sum(counts) should be number of valid windows
    # We assume the standard construction is valid.
    
    # Local Search: Remove positions
    # A position i can be removed if:
    # 1. All windows containing i that are valid have count > 1 (so they survive)
    # 2. All new windows formed by removal are valid permutations
    
    def try_remove(i):
        L_cur = len(current_word)
        if i < 0 or i >= L_cur:
            return False, [], []
        
        # Windows affected by removal of i
        # Old windows: [i-n+1, i] (start indices)
        # New windows: [i-n+1, i-1] (start indices in new string, but content shifts)
        
        # 1. Check old windows
        # A window starting at j contains i if j <= i <= j+n-1 => i-n+1 <= j <= i
        # Valid old windows must have count > 1
        valid_old = []
        for j in range(max(0, i - n + 1), min(i, L_cur - n) + 1):
            wid = get_window_id(j)
            if wid != -1:
                if counts[wid] <= 1:
                    return False, [], []
                valid_old.append((j, wid))
        
        # 2. Check new windows
        # After removing i, the string becomes W' = W[:i] + W[i+1:]
        # New windows start at j in W'.
        # The content of window starting at j in W' is W'[j:j+n]
        # This corresponds to W[j:i] + W[i+1:j+n+1]
        # Valid new windows must be valid permutations.
        valid_new = []
        # Range of j in W' that are affected:
        # j such that j < i < j+n (in W coordinates, roughly)
        # Actually, windows starting at j in W' for j in [i-n+1, i-1] are the ones that change content.
        # Windows starting at j >= i in W' are just windows starting at j+1 in W (unchanged content).
        # Windows starting at j < i-n+1 in W' are just windows starting at j in W (unchanged content).
        
        # So we only need to check j in [max(0, i-n+1), min(i-1, L_cur-n-1)]
        # Note: L_cur decreases by 1, so max start index is L_cur-1-n.
        # In W', length is L_cur-1. Max start is L_cur-1-n.
        
        for j in range(max(0, i - n + 1), min(i - 1, L_cur - n - 1) + 1):
            # Construct new window
            # W'[j:j+n] = W[j:i] + W[i+1:j+n+1]
            # Length of W[j:i] is i-j
            # Length of W[i+1:j+n+1] is j+n+1 - (i+1) = j+n-i
            # Total length n. Correct.
            
            # To avoid creating a list, we can check distinctness directly
            # Or construct tuple and check
            new_w = tuple(current_word[j:i] + current_word[i+1:j+n+1])
            
            if len(set(new_w)) == n:
                wid = perm_id.get(new_w, -1)
                if wid != -1:
                    valid_new.append((j, wid))
                else:
                    # Should not happen if perm_id is complete
                    return False, [], []
            else:
                # New window is not a valid permutation
                # This is allowed? 
                # The problem requires the *final* string to contain all permutations.
                # It does NOT require every window in the intermediate string to be a permutation.
                # However, if a new window is NOT a permutation, it doesn't contribute to the count.
                # Is it a problem? No.
                # So we don't need to reject based on invalid new windows.
                # We just don't add to valid_new.
                pass
        
        # If we get here, removal is valid
        return True, valid_old, valid_new

    def apply_removal(i, valid_old, valid_new):
        nonlocal current_word, counts
        # Update counts
        for j, wid in valid_old:
            counts[wid] -= 1
        for j, wid in valid_new:
            counts[wid] += 1
            
        # Remove from word
        current_word = current_word[:i] + current_word[i+1:]

    def revert_removal(i, valid_old, valid_new, old_word):
        nonlocal current_word, counts
        for j, wid in valid_old:
            counts[wid] += 1
        for j, wid in valid_new:
            counts[wid] -= 1
        current_word = old_word

    improved = True
    pass_count = 0
    L_cur = len(current_word)
    
    while improved and time.time() < deadline - 5:
        improved = False
        pass_count += 1
        L_cur = len(current_word)
        
        # Randomize order
        indices = list(range(L_cur))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 5:
                break
            if i >= len(current_word):
                continue
            
            success, valid_old, valid_new = try_remove(i)
            if success:
                old_word = current_word
                apply_removal(i, valid_old, valid_new)
                improved = True
                L_cur -= 1

    # Final verification
    if time.time() < deadline - 2:
        seen = set()
        for i in range(len(current_word) - n + 1):
            window = tuple(current_word[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        if len(seen) != target_count:
            # Fallback to original if broken (should not happen)
            current_word = list(word)

    return "".join(map(str, current_word))
