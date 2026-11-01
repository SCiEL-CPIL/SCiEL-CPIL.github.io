import time
import random
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Hardcoded for small n
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412314231243121342132413214321"
    
    if n != 7:
        # Fallback for other n (though task implies n=7)
        # Use recursive construction
        word = "1"
        for m in range(2, n + 1):
            perms = []
            seen = set()
            for i in range(len(word) - (m - 1) + 1):
                window = word[i:i + m - 1]
                if len(set(window)) == m - 1:
                    if window not in seen:
                        seen.add(window)
                        perms.append(window)
            
            new_word = ""
            for p in perms:
                block = p + str(m) + p
                max_overlap = 0
                limit = min(len(new_word), len(block))
                for o in range(limit, 0, -1):
                    if new_word[-o:] == block[:o]:
                        max_overlap = o
                        break
                new_word += block[max_overlap:]
            word = new_word
        return word

    # n = 7 specific logic
    # 1. Generate recursive superpermutation
    word_str = "1"
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word_str) - (m - 1) + 1):
            window = word_str[i:i + m - 1]
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word_str = ""
        for p in perms:
            block = p + str(m) + p
            max_overlap = 0
            limit = min(len(new_word_str), len(block))
            for o in range(limit, 0, -1):
                if new_word_str[-o:] == block[:o]:
                    max_overlap = o
                    break
            new_word_str += block[max_overlap:]
        word_str = new_word_str
    
    arr = [int(c) for c in word_str]
    L = len(arr)
    
    # Precompute permutation mapping
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Helper to get perm index for a window in arr
    def get_perm_idx(arr, start):
        if start < 0 or start + n > len(arr):
            return -1
        # Check if it's a permutation
        # Since n=7, checking set size is fast, but we can just map tuple
        # We know all windows in a valid superpermutation are permutations,
        # but after deletions, some might not be.
        # However, we only delete if valid.
        # To be safe, check if tuple is in map.
        p = tuple(arr[start:start+n])
        return perm_to_idx.get(p, -1)

    # State initialization
    perm_idx = [-1] * L
    perm_count = [0] * num_perms
    
    for i in range(L - n + 1):
        p_idx = get_perm_idx(arr, i)
        if p_idx != -1:
            perm_idx[i] = p_idx
            perm_count[p_idx] += 1
            
    # Criticality: pos_critical[i] = number of unique perms covering i
    pos_critical = [0] * L
    for i in range(L - n + 1):
        p_idx = perm_idx[i]
        if p_idx != -1 and perm_count[p_idx] == 1:
            for k in range(n):
                pos_critical[i+k] += 1

    best_len = L
    best_arr = arr[:]
    target_len = 5905

    # DFS with backtracking
    # Stack entries: (arr_copy, perm_idx_copy, perm_count_copy, pos_critical_copy, L)
    # This is memory heavy. 
    # Alternative: Record deletions and undo them.
    # Since depth is small, we can just copy arrays. L=5913, copying 4 arrays is ~23k ints. 
    # 1000 restarts * 23k * 4 bytes = 92 MB. Might be okay.
    # But we can do better: just store the list of deleted indices in order?
    # Undoing a deletion requires re-inserting and updating state.
    # Let's stick to copying state for simplicity, but limit restarts.
    
    # Actually, a simpler randomized greedy with many restarts is faster to implement and likely sufficient.
    # Let's try that first.
    
    attempts = 0
    max_attempts = 10000 # High number, but each attempt is fast
    
    while time.time() < deadline - 5 and attempts < max_attempts:
        # Reset state
        curr_arr = arr[:]
        curr_L = L
        curr_perm_idx = perm_idx[:]
        curr_perm_count = perm_count[:]
        curr_pos_critical = pos_critical[:]
        
        while time.time() < deadline - 5:
            # Find deletable indices
            candidates = []
            for i in range(curr_L):
                if curr_pos_critical[i] == 0:
                    candidates.append(i)
            
            if not candidates:
                break
            
            # Pick random candidate
            del_idx = random.choice(candidates)
            
            # Perform deletion
            # 1. Update perm_count for windows ending at or before del_idx
            # Windows affected: j in [del_idx - n + 1, del_idx]
            # These windows will disappear or shift.
            # Specifically, window starting at j contains del_idx if j <= del_idx < j+n.
            # So j in [del_idx - n + 1, del_idx].
            
            # We need to decrement counts for these windows.
            for j in range(max(0, del_idx - n + 1), min(del_idx, curr_L - n) + 1):
                p_idx = curr_perm_idx[j]
                if p_idx != -1:
                    curr_perm_count[p_idx] -= 1
                    # If this perm count becomes 0, it's invalid? 
                    # No, we only delete non-critical, so count was > 1.
                    # So count becomes >= 1. It remains a valid perm.
            
            # 2. Pop element
            curr_arr.pop(del_idx)
            curr_L -= 1
            
            # 3. Update perm_idx for the new array
            # The indices >= del_idx shift left by 1.
            # We need to recalculate perm_idx for the whole array? 
            # Or just the affected region?
            # The shift affects all indices >= del_idx.
            # It is safer and fast enough to recalculate perm_idx for the whole array.
            # O(L) per deletion. 8 deletions * 5900 = 47k ops. Very fast.
            
            # Resize perm_idx? No, just overwrite.
            # We only care about indices 0..curr_L-n
            for i in range(curr_L - n + 1):
                curr_perm_idx[i] = get_perm_idx(curr_arr, i)
            
            # 4. Recalculate perm_count
            # Reset counts
            for i in range(num_perms):
                curr_perm_count[i] = 0
            for i in range(curr_L - n + 1):
                if curr_perm_idx[i] != -1:
                    curr_perm_count[curr_perm_idx[i]] += 1
            
            # 5. Recalculate criticality
            curr_pos_critical = [0] * curr_L
            for i in range(curr_L - n + 1):
                if curr_perm_idx[i] != -1 and curr_perm_count[curr_perm_idx[i]] == 1:
                    for k in range(n):
                        curr_pos_critical[i+k] += 1
            
            if curr_L < best_len:
                best_len = curr_L
                best_arr = curr_arr[:]
                
            if curr_L == target_len:
                break
        
        if best_len == target_len:
            break
            
        attempts += 1
        # Small sleep to avoid burning CPU if stuck instantly? No, just loop.
        
    return ''.join(str(x) for x in best_arr)
