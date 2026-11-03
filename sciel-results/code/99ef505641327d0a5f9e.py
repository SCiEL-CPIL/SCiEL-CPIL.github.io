"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct(n):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Generate classic superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract permutations of length m-1 from the current word
        # The current word contains all permutations of 1..m-1
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find overlap
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # 2. Initialize counts of all windows of length n
    n_len = n
    counts = {}
    for i in range(len(word) - n_len + 1):
        w = tuple(word[i:i + n_len])
        counts[w] = counts.get(w, 0) + 1
        
    # Ensure all permutations are present (sanity check)
    # The classic construction guarantees this.
    
    # 3. Greedy deletion loop
    # We try to delete characters one by one.
    # If a deletion maintains validity (all counts > 0), we accept it.
    # We repeat until no more deletions are possible or time runs out.
    
    max_iter = 1000  # Safety break
    iter_count = 0
    
    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1
        deleted = False
        # Try deleting each index
        # We iterate from 0 to len(word)-1
        # If we delete, indices shift, so we break and restart
        for i in range(len(word)):
            if time.time() > deadline:
                return "".join(map(str, word))
                
            # Identify old windows to remove
            # Windows containing index i start at i-6 to i (for n=7)
            # Generally i-(n-1) to i
            old_windows = []
            for start in range(max(0, i - n_len + 1), min(i + 1, len(word) - n_len + 1)):
                old_windows.append(tuple(word[start:start + n_len]))
            
            # Identify new windows to add
            # New string is word[:i] + word[i+1:]
            # Junction is between i-1 and i in new string
            # New windows start at i-6 to i-1 in new string (for n=7)
            # Generally i-(n-1) to i-1
            # We need to construct these tuples from the old word
            new_windows = []
            # The new string indices map to old:
            # new[j] = old[j] for j < i
            # new[j] = old[j+1] for j >= i
            # New windows start at s in new string.
            # s ranges from max(0, i-n+1) to i-1
            # Wait, if i=0, new string starts with old[1]. Junction is at start?
            # If i=0, we remove first char. New windows are old[1:1+n], old[2:2+n]... no.
            # If i=0, new string is old[1:].
            # Windows in new string:
            # Start 0: old[1:1+n]
            # This window existed in old string at start 1.
            # So it's not "new" in value?
            # My previous logic:
            # S_new = S_old - {old windows containing i} + {new windows containing junction}
            # If i=0, old windows containing 0: start 0.
            # New string junction: between -1 and 0? No, it's the start.
            # The "new" windows are those that span the cut.
            # If cut is at start, there is no left part.
            # So the windows starting at 0 in new string are old windows starting at 1.
            # They are already in S_old (at index 1).
            # So we just remove the window at start 0.
            # No new windows are added?
            # Let's check.
            # Old: A B. Remove A. New: B.
            # Windows in B are subset of windows in A B?
            # Yes, any window in B is a window in A B starting at index >= 1.
            # So S_new = S_old - {windows starting at 0 in old}.
            # So new_windows is empty.
            
            # General case:
            # Cut is between i-1 and i+1.
            # New windows are those that include both i-1 and i+1 (from old).
            # In new string, these are windows starting at i-6 to i-1.
            # Let's construct them.
            for s_new in range(max(0, i - n_len + 1), i):
                # Window in new string starting at s_new
                # It consists of:
                # Part from old left: s_new to i-1 (length i - s_new)
                # Part from old right: i+1 to ...
                # Total length n_len
                # Left part: word[s_new : i]
                # Right part: word[i+1 : i+1 + (n_len - (i - s_new))]
                left_part = word[s_new:i]
                right_len = n_len - len(left_part)
                right_part = word[i+1:i+1+right_len]
                if len(left_part) + len(right_part) == n_len:
                    new_windows.append(tuple(left_part + right_part))
            
            # Check validity
            valid = True
            # Temporarily decrement
            for w in old_windows:
                counts[w] -= 1
                if counts[w] == 0:
                    valid = False
                    break
            
            if valid:
                # Increment new windows
                for w in new_windows:
                    counts[w] = counts.get(w, 0) + 1
            
            if valid:
                # Accept deletion
                word.pop(i)
                deleted = True
                break
            else:
                # Rollback decrements
                for w in old_windows:
                    counts[w] += 1
                # No need to rollback increments as they weren't applied if valid was False before increment?
                # Wait, I only increment if valid. So if valid is False, I didn't increment.
                # But I did decrement. So I just rollback decrements.
        
        if not deleted:
            break
            
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
