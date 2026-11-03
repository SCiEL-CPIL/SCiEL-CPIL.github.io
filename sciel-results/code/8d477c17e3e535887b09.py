import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    if n <= 1:
        return "1"
    
    # 1. Generate standard superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms = []
        seen = set()
        # Find all unique permutations of 1..m-1 in the current word
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Max overlap
            overlap = 0
            max_ov = min(len(new_word), len(block))
            for o in range(max_ov, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    
    L = len(word)
    
    # Precompute permutation to ID mapping
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts and window_ids
    counts = [0] * num_perms
    window_ids = [0] * (L - n + 1)
    
    pid = perm_to_id.get
    
    for j in range(L - n + 1):
        t = tuple(word[j:j+n])
        id_val = pid(t, -1)
        if id_val != -1:
            counts[id_val] += 1
            window_ids[j] = id_val
        else:
            window_ids[j] = -1 # Should not happen for valid superperm
    
    # Greedy Deletion Loop
    # We will alternate between forward and backward passes
    # And use the optimized window_ids update
    
    while time.time() < deadline:
        changed = False
        L = len(word)
        
        # Forward pass
        i = 0
        while i < L:
            if time.time() > deadline - 1:
                break
                
            # Check safety at i
            # Affected old windows: j in [max(0, i-n+1), min(i, L-n)]
            start_min = max(0, i - n + 1)
            start_max = min(i, L - n)
            
            safe = True
            affected_old_ids = []
            
            for j in range(start_min, start_max + 1):
                id_val = window_ids[j]
                if id_val != -1:
                    affected_old_ids.append(id_val)
                    if counts[id_val] == 1:
                        safe = False
                        break
            
            if safe:
                # Perform deletion
                for id_val in affected_old_ids:
                    counts[id_val] -= 1
                
                word.pop(i)
                L -= 1
                changed = True
                
                # Update window_ids
                # New length L
                # New number of windows: L - n + 1
                # Old number of windows: L + 1 - n + 1 = L - n + 2
                
                # 1. Handle seam windows: j in [max(0, i-n+1), min(i-1, L-n)]
                # Note: L is the NEW length.
                # The windows that change are those that covered index i in the OLD word.
                # In the NEW word, these are indices j such that the window [j, j+n-1] covers the gap.
                # The gap is at i.
                # So j <= i-1 and j+n-1 >= i => j >= i-n+1.
                # Also j must be a valid index in the new word: 0 <= j <= L-n.
                
                seam_min = max(0, i - n + 1)
                seam_max = min(i - 1, L - n)
                
                # Update these specific windows
                for j in range(seam_min, seam_max + 1):
                    # Compute new window ID
                    t = tuple(word[j:j+n])
                    new_id = pid(t, -1)
                    if new_id != -1:
                        counts[new_id] += 1
                        window_ids[j] = new_id
                    else:
                        window_ids[j] = -1
                
                # 2. Shift the rest of window_ids
                # Old indices > i-1 correspond to new indices >= i
                # Specifically, new_window_ids[k] = old_window_ids[k+1] for k >= i
                # The range of k is [i, L-n]
                # The range of old indices is [i+1, L-n+1]
                
                if i <= L - n:
                    # window_ids[i : L-n+1] = window_ids[i+1 : L-n+2]
                    # Since window_ids is a list, we can do slice assignment
                    window_ids[i : L - n + 1] = window_ids[i + 1 : L - n + 2]
                    # And remove the last element to maintain length L-n+1
                    window_ids.pop()
                else:
                    # If i > L-n, no windows to shift?
                    # Actually, if i is at the end, the number of windows decreases by 1.
                    # The last window was removed.
                    # window_ids length should be L-n+1.
                    # Old length was L-n+2.
                    # We popped one from word, so L decreased by 1.
                    # We need to reduce window_ids length by 1.
                    # The affected windows were handled above.
                    # If i > L-n, it means the deletion was near the end.
                    # The windows that were shifted are none?
                    # Let's check.
                    # If i > L-n, then the max index of old windows is L-n.
                    # The deletion is at i.
                    # The windows affected are j in [i-n+1, L-n].
                    # These are handled in the seam loop?
                    # seam_max = min(i-1, L-n).
                    # So yes, they are handled.
                    # The windows j < i-n+1 are unchanged.
                    # The windows j > L-n don't exist.
                    # So we just need to pop the last element if the length is too big?
                    # Wait, the number of windows is always L-n+1.
                    # We started with L_old-n+1.
                    # We need L_new-n+1 = L_old-n.
                    # So we must reduce length by 1.
                    # The slice assignment `window_ids[i : L - n + 1] = ...`
                    # If i > L-n, the slice is empty.
                    # So we just pop the last element?
                    # But which element?
                    # The last element of the old window_ids corresponded to the last window.
                    # If the deletion was at the very end, the last window might be affected or removed.
                    # Actually, the logic `window_ids[i : L - n + 1] = window_ids[i + 1 : L - n + 2]`
                    # effectively shifts and drops the last element.
                    # If i is large, the slice is small.
                    # Let's just ensure the length is correct.
                    # The slice assignment handles the shift.
                    # If i > L-n, `window_ids[i : L-n+1]` is empty.
                    # `window_ids[i+1 : L-n+2]` is empty.
                    # So nothing happens.
                    # But we need to pop one element from window_ids.
                    # The last element `window_ids[-1]` corresponds to the old last window.
                    # If the deletion was at the end, this window is gone or changed.
                    # If it's changed, it's in the seam.
                    # If it's gone, we just pop.
                    # So `window_ids.pop()` is safe.
                    window_ids.pop()

            else:
                i += 1
                
        # If no changes in forward pass, try backward pass
        if not changed:
            i = L - 1
            while i >= 0:
                if time.time() > deadline - 1:
                    break
                    
                start_min = max(0, i - n + 1)
                start_max = min(i, L - n)
                
                safe = True
                affected_old_ids = []
                
                for j in range(start_min, start_max + 1):
                    id_val = window_ids[j]
                    if id_val != -1:
                        affected_old_ids.append(id_val)
                        if counts[id_val] == 1:
                            safe = False
                            break
                
                if safe:
                    for id_val in affected_old_ids:
                        counts[id_val] -= 1
                    
                    word.pop(i)
                    L -= 1
                    changed = True
                    
                    seam_min = max(0, i - n + 1)
                    seam_max = min(i - 1, L - n)
                    
                    for j in range(seam_min, seam_max + 1):
                        t = tuple(word[j:j+n])
                        new_id = pid(t, -1)
                        if new_id != -1:
                            counts[new_id] += 1
                            window_ids[j] = new_id
                        else:
                            window_ids[j] = -1
                    
                    if i <= L - n:
                        window_ids[i : L - n + 1] = window_ids[i + 1 : L - n + 2]
                        window_ids.pop()
                    else:
                        window_ids.pop()
                else:
                    i -= 1
            
            if not changed:
                # If neither forward nor backward pass changes anything, we are at a local minimum.
                # We can break, as further single-deletion moves won't help.
                break

    return "".join(map(str, word))
