import time
import random
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    # 1. Generate initial superpermutation (Recursive Construction)
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract unique permutations of length m-1
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
                if o == 0:
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    if n <= 4:
        return "".join(map(str, word))
    
    current = list(word)
    L = len(current)
    
    # 2. Setup for Local Search
    # Map permutations to IDs for speed
    perm_to_id = {}
    id_to_perm = {}
    next_id = 0
    
    def get_id(perm):
        nonlocal next_id
        if perm in perm_to_id:
            return perm_to_id[perm]
        perm_to_id[perm] = next_id
        id_to_perm[next_id] = perm
        next_id += 1
        return next_id - 1

    # Initialize perm_ids and perm_count
    perm_ids = [0] * (L - n + 1)
    perm_count = defaultdict(int)
    
    for i in range(L - n + 1):
        p = tuple(current[i:i+n])
        pid = get_id(p)
        perm_ids[i] = pid
        perm_count[pid] += 1
        
    target_perms = 1
    for i in range(1, n + 1):
        target_perms *= i
        
    if len(perm_count) < target_perms:
        return "".join(map(str, current))
        
    # 3. Greedy Deletion Loop
    # We repeatedly scan the string and delete the first valid index found.
    # This allows cascading deletions.
    
    while time.time() < deadline:
        deleted = False
        # Scan from 0 to L-1
        for i in range(L):
            # Check if i can be deleted
            # Old windows containing i: start indices [max(0, i-n+1), min(i, L-n)]
            start_w = max(0, i - n + 1)
            end_w = min(i, L - n)
            
            if start_w > end_w:
                # i is not in any window? Impossible for valid string, but handle edge cases
                continue
                
            # Identify unique old windows
            unique_old_ids = set()
            for w in range(start_w, end_w + 1):
                pid = perm_ids[w]
                if perm_count[pid] == 1:
                    unique_old_ids.add(pid)
            
            # If no unique old windows, deletion is safe (counts just decrease, but don't hit 0)
            if not unique_old_ids:
                # Perform deletion
                _perform_deletion(i, current, perm_ids, perm_count, n, L, get_id)
                L -= 1
                deleted = True
                break # Restart scan
            
            # Check if new windows cover unique_old_ids
            # New windows are formed at start positions j in [max(0, i-n+1), i-1]
            # Note: In the NEW string, these windows start at j.
            # In the OLD string, the window at j consists of current[j:i] + current[i+1 : ...]
            
            new_ids = set()
            for j in range(max(0, i - n + 1), i):
                # Check bounds for suffix in old string
                # The window in new string starts at j.
                # It takes j..i-1 from old, and i+1..i+suffix_len from old.
                # suffix_len = n - (i - j)
                # End index in old = i + suffix_len = i + n - i + j = n + j
                # We need n + j <= L (old length) ? No, indices are 0..L-1.
                # Last index accessed is i + suffix_len - 1 = n + j - 1.
                # We need n + j - 1 < L => n + j <= L.
                if n + j > L:
                    continue
                
                prefix_len = i - j
                suffix_len = n - prefix_len
                
                new_p = tuple(current[j:i] + current[i+1 : i+1+suffix_len])
                new_pid = get_id(new_p)
                new_ids.add(new_pid)
                
                # Early exit
                if unique_old_ids.issubset(new_ids):
                    break
            
            if unique_old_ids.issubset(new_ids):
                # Perform deletion
                _perform_deletion(i, current, perm_ids, perm_count, n, L, get_id)
                L -= 1
                deleted = True
                break # Restart scan
        
        if not deleted:
            break
            
    return "".join(map(str, current))

def _perform_deletion(i, current, perm_ids, perm_count, n, L, get_id):
    """
    Deletes index i from current and perm_ids, updates perm_count.
    Note: perm_ids is a list of length L-n+1.
    After deletion, length becomes L-n.
    """
    
    # 1. Update perm_count for old windows containing i
    start_w = max(0, i - n + 1)
    end_w = min(i, L - n)
    
    for w in range(start_w, end_w + 1):
        pid = perm_ids[w]
        perm_count[pid] -= 1
        if perm_count[pid] == 0:
            # Optional: clean up, but defaultdict handles it
            pass

    # 2. Calculate new pids for the gap
    # New windows start at j in [max(0, i-n+1), i-1]
    # These correspond to new indices j.
    new_pids = {}
    for j in range(max(0, i - n + 1), i):
        if n + j > L:
            continue
        prefix_len = i - j
        suffix_len = n - prefix_len
        new_p = tuple(current[j:i] + current[i+1 : i+1+suffix_len])
        new_pid = get_id(new_p)
        new_pids[j] = new_pid
        perm_count[new_pid] += 1

    # 3. Update perm_ids
    # New length of perm_ids is L - n
    # 
    # Part 1: Unchanged prefix
    # Old indices 0 to start_w-1 are unchanged.
    # Note: start_w = max(0, i-n+1).
    # So indices 0 to start_w-1 are unchanged.
    
    # Part 2: New windows
    # New indices j in [start_w, i-1] (where j is the start index in new string)
    # Wait, the new windows are at indices j in the NEW string.
    # The range of j is [max(0, i-n+1), i-1].
    # Let's call this range [new_start, new_end].
    
    # Part 3: Shifted suffix
    # Old indices end_w+1 to L-n are shifted to new indices end_w to L-n-1.
    # Wait, let's look at the mapping again.
    # Old index $k$ maps to New index $k-1$ if $k > i$.
    # The windows affected are those containing $i$.
    # Old windows $w \in [start_w, end_w]$ are destroyed.
    # Old windows $w < start_w$ are unchanged.
    # Old windows $w > end_w$ are shifted.
    # 
    # New windows are formed at indices $j \in [start_w, i-1]$.
    # Note: $end_w = \min(i, L-n)$.
    # If $i \le L-n$, then $end_w = i$.
    # If $i > L-n$, then $end_w = L-n$.
    
    # Let's construct the new list.
    new_perm_ids = [0] * (L - n)
    
    # 1. Prefix: old indices 0 to start_w - 1
    # These map to new indices 0 to start_w - 1.
    if start_w > 0:
        for j in range(start_w):
            new_perm_ids[j] = perm_ids[j]
            
    # 2. Mid: New windows
    # These are at new indices j in [start_w, i-1].
    # But we must ensure j < L-n (new length).
    # The max j is i-1.
    # If i-1 >= L-n, we only go up to L-n-1.
    # However, the condition `n+j > L` in the calculation loop handles validity.
    # Let's just iterate the valid new window starts.
    for j in range(start_w, min(i, L-n)):
        if j in new_pids:
            new_perm_ids[j] = new_pids[j]
            
    # 3. Suffix: Old indices end_w + 1 to L-n
    # These map to new indices end_w to L-n-1.
    # Old index k maps to new index k-1.
    # So new index j gets old index j+1.
    # Range of j: end_w to L-n-1.
    # Corresponding old indices: end_w+1 to L-n.
    
    # We need to fill new_perm_ids from index end_w to end.
    # The values come from perm_ids[end_w+1 : L-n+1]
    
    suffix_start_new = end_w
    suffix_start_old = end_w + 1
    suffix_len = (L - n) - suffix_start_new
    if suffix_len > 0:
        # Copy slice
        new_perm_ids[suffix_start_new : suffix_start_new + suffix_len] = perm_ids[suffix_start_old : suffix_start_old + suffix_len]
        
    # Update global perm_ids reference? 
    # Python passes lists by reference, but we are reassigning inside the function?
    # No, we need to update the list in place or return it.
    # Let's update in place to avoid returning.
    perm_ids[:] = new_perm_ids
    
    # Update current word
    current.pop(i)
