import time
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

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
            # Find max overlap
            overlap = 0
            max_check = min(len(new), len(block))
            for o in range(max_check, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    L = len(word)
    
    # Helper to get window ID
    # Using base-n encoding is fast enough for n=7
    def get_window_id(start):
        val = 0
        for k in range(n):
            val = val * n + word[start+k]
        return val

    # Initialize counts
    window_counts = {}
    for i in range(L - n + 1):
        wid = get_window_id(i)
        window_counts[wid] = window_counts.get(wid, 0) + 1
    
    # Deletion Loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        i = 0
        while i < L:
            if time.time() > deadline:
                break
            
            # Check if deletion at i is valid
            # Windows containing i are at start indices j in [i-n+1, i]
            valid = True
            affected_ids = []
            
            start_min = max(0, i - n + 1)
            start_max = min(i, L - n)
            
            for j in range(start_min, start_max + 1):
                wid = get_window_id(j)
                affected_ids.append(wid)
                if window_counts[wid] == 1:
                    valid = False
                    break
            
            if valid:
                # Accept deletion
                # 1. Decrement counts for affected windows
                for wid in affected_ids:
                    window_counts[wid] -= 1
                    if window_counts[wid] == 0:
                        del window_counts[wid]
                
                # 2. Calculate new windows
                # The join happens between i-1 and i (new indices)
                # Old: ... A[i-1] A[i] A[i+1] ...
                # New: ... A[i-1] A[i+1] ...
                # New windows are those crossing the join.
                # In the new word of length L-1, the join is at index i.
                # Windows containing index i in new word:
                # Start indices k in [i-n+1, i] in new word.
                # But we only need to add the ones that are NEW.
                # The windows completely to the left (k+n <= i) are unchanged (just shifted? No, indices < i are same).
                # The windows completely to the right (k >= i) are shifted left by 1.
                # The windows crossing the join are new.
                
                # Actually, it's easier to just recompute the counts for the "new" windows that cross the gap.
                # The gap is between old index i-1 and i+1.
                # In the new word, these are adjacent at index i-1 and i.
                # The windows that include this new adjacency are those starting at k such that
                # k <= i-1 < k+n and k <= i < k+n?
                # Let's just iterate k from max(0, i-n+1) to min(i-1, L-1-n) in the NEW word context?
                # No, let's map new indices to old indices.
                # New word W' length L-1.
                # W'[k] = W[k] if k < i, else W[k+1].
                
                # We need to add counts for windows in W' that were NOT in W.
                # The windows in W' that are identical to windows in W are those that don't cross the deletion point.
                # Windows crossing the deletion point in W' are those where the window covers index i-1 and i in W'.
                # In W, index i-1 is W[i-1], index i is W[i+1].
                # So the new windows are formed by substrings of W that skip W[i].
                
                # Which start indices k in W' correspond to windows crossing the gap?
                # The gap is between W'[i-1] and W'[i].
                # A window W'[k:k+n] crosses this gap if k <= i-1 and k+n > i-1.
                # i.e., k <= i-1 < k+n.
                # Also k must be valid: 0 <= k <= L-1-n.
                
                new_start_min = max(0, i - n + 1)
                new_start_max = min(i - 1, L - 1 - n)
                
                for k in range(new_start_min, new_start_max + 1):
                    # Construct window ID from W'
                    # W'[k:k+n]
                    val = 0
                    # Left part: indices k to i-1 in W' (same as W)
                    # Length: i - k
                    for idx in range(k, i):
                        val = val * n + word[idx]
                    
                    # Right part: indices i to k+n-1 in W' (which are i+1 to ... in W)
                    # Number of elements: n - (i - k)
                    # Start in W: i+1
                    for idx in range(i + 1, i + 1 + (n - (i - k))):
                        val = val * n + word[idx]
                    
                    window_counts[val] = window_counts.get(val, 0) + 1
                
                # 3. Perform deletion
                del word[i]
                L -= 1
                improved = True
                # Restart scan to catch further deletions
                break 
            else:
                i += 1
                
    return "".join(map(str, word))
