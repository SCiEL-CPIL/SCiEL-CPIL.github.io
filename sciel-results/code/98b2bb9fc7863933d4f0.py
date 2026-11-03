import time
import random
from itertools import permutations

def construct(n):
    if n <= 1:
        return "1"
    
    # 1. Construct Classic Superpermutation
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Initial length is sum(1! to n!)
    # For n=7, length is 5913
    
    if n <= 6:
        return "".join(map(str, word))

    # 2. Local Search / Greedy Deletion with Incremental Counts
    start_time = time.time()
    deadline = start_time + 140 # Safety margin for 150s limit
    
    # Precompute permutation IDs
    all_perms = set(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts
    counts = [0] * num_perms
    L = len(word)
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        if p in perm_to_id:
            counts[perm_to_id[p]] += 1
            
    # Check initial validity
    if any(c == 0 for c in counts):
        # Should not happen, but return if it does
        return "".join(map(str, word))

    def get_pid(w, start):
        # w is list, start is index
        if start < 0 or start + n > len(w):
            return None
        p = tuple(w[start:start+n])
        return perm_to_id.get(p)

    # Local Search Loop
    improved = True
    while time.time() < deadline and improved:
        improved = False
        L = len(word)
        if L <= n:
            break
            
        # Randomize order to escape specific local minima
        indices = list(range(L))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            L = len(word)
            if i >= L:
                continue
                
            # Try to delete word[i]
            # Affected old windows: start k in [max(0, i-n+1), min(i, L-n)]
            old_starts = []
            old_pids = []
            for k in range(max(0, i - n + 1), min(i + 1, L - n + 1)):
                pid = get_pid(word, k)
                if pid is not None:
                    old_starts.append(k)
                    old_pids.append(pid)
                    counts[pid] -= 1
            
            # Check if any count became 0
            valid = True
            for pid in old_pids:
                if counts[pid] == 0:
                    valid = False
                    break
            
            if not valid:
                # Revert
                for pid in old_pids:
                    counts[pid] += 1
                continue
            
            # If valid, simulate deletion to find new windows
            # New word length is L-1
            # New affected windows:
            # 1. Windows starting at k < i in new word that cross the gap?
            #    In new word, index i is gone.
            #    Old index j > i becomes new index j-1.
            #    New window starting at k (k < i):
            #       If k+n <= i: same as old word[k:k+n]. No change in content, but index shift?
            #       Actually, if k+n <= i, the window is entirely before the deletion. 
            #       Its content is unchanged. Its start index in new word is k.
            #       In old word, it was at k. We already handled old windows starting at k.
            #       We need to check if this window is *newly* valid or was already counted?
            #       The window content is the same. So if it was a permutation before, it is now.
            #       We decremented it in old_starts. We must increment it back?
            #       No, the window still exists in the new word.
            #       Let's re-evaluate the logic.
            
            # Standard approach:
            # 1. Identify set of windows in OLD word that will be REMOVED or CHANGED.
            # 2. Identify set of windows in NEW word that will be ADDED or CHANGED.
            
            # Removed/Changed Old Windows:
            # Any window in OLD word that includes index i.
            # Starts k in [i-n+1, i].
            
            # Added/Changed New Windows:
            # Any window in NEW word that includes the "gap" at position i (between i-1 and i).
            # In NEW word, the gap is between i-1 and i.
            # Windows covering the gap start at k in [i-n+1, i-1].
            # Note: New word indices are 0..L-2.
            
            # Let's just compute the new windows for starts k in [max(0, i-n+1), min(i-1, L-n)]
            # And also, windows that were shifted?
            # Windows starting at k >= i in NEW word correspond to OLD windows starting at k+1.
            # These are unchanged in content.
            # Windows starting at k < i in NEW word:
            #   If k+n <= i: Content is OLD[k:k+n]. Unchanged.
            #   If k+n > i: Content is OLD[k:i] + OLD[i+1:k+n+1]. Changed.
            
            # So, we need to:
            # 1. Decrement counts for ALL old windows that include index i. (Done above in old_pids)
            # 2. Increment counts for ALL new windows that include the gap (start k in new word in [i-n+1, i-1]).
            #    BUT, we must be careful: some new windows might be identical to old windows that did NOT include i?
            #    No, if a window in NEW word starts at k < i and k+n <= i, it is identical to OLD[k:k+n].
            #    Did we decrement OLD[k:k+n]?
            #    OLD[k:k+n] includes i only if k+n > i.
            #    So if k+n <= i, we did NOT decrement it.
            #    So we should NOT increment it again.
            
            # So, for new windows:
            # Only increment those where the content actually changed (i.e., crossed the gap).
            # These are starts k in NEW word such that k < i and k+n > i.
            # i.e., k in [i-n+1, i-1].
            
            new_pids = []
            for k in range(max(0, i - n + 1), min(i, L - n + 1)): # k is start in NEW word
                # Check if this window crosses the gap
                # Gap is between i-1 and i in NEW word.
                # Window k..k+n-1 crosses if k <= i-1 and k+n-1 >= i.
                # i.e., k <= i-1 and k+n > i.
                # This is always true for k in [i-n+1, i-1].
                
                # Construct new window
                # New word is word[:i] + word[i+1:]
                # We don't want to create the full string every time.
                # Window is word[k:i] + word[i+1:k+n+1]
                # Note: k is index in NEW word.
                # For k < i, new_word[k] = word[k].
                # new_word[i] = word[i+1].
                # So window is word[k:i] + word[i+1:k+n+1]
                
                # Optimization: Just check if this specific tuple is in perm_to_id
                # and increment.
                
                # We need to ensure we don't double count if this window existed in OLD word 
                # and was not decremented?
                # If k+n <= i, it didn't cross gap, so we skip.
                # If k+n > i, it crossed gap.
                # Did we decrement the corresponding OLD window?
                # The OLD window at start k was word[k:k+n].
                # This OLD window included i (since k+n > i and k < i).
                # So yes, we decremented it.
                # So we must increment the NEW window.
                
                # Construct tuple
                # word[k:i] is list slice
                # word[i+1:k+n+1] is list slice
                # Combining them creates a new list.
                # For n=7, this is small.
                
                new_win = word[k:i] + word[i+1:k+n+1]
                pid = perm_to_id.get(tuple(new_win))
                if pid is not None:
                    new_pids.append(pid)
                    counts[pid] += 1
            
            # Check validity
            valid = True
            for pid in new_pids:
                if counts[pid] < 0: # Should not happen, but just in case
                    valid = False
                    break
            # We already checked old_pids for 0.
            # But wait, what if a new window is the same as an old window that was NOT affected?
            # Impossible, new windows only differ in the gap region.
            
            # One more check: Did we accidentally make a count 0?
            # We decremented old_pids. We checked if any became 0.
            # We incremented new_pids. This can only increase counts.
            # So validity holds if no old_pid count hit 0.
            
            if valid:
                # Commit deletion
                word = word[:i] + word[i+1:]
                improved = True
                break # Restart pass with new length
            else:
                # Revert
                for pid in old_pids:
                    counts[pid] += 1
                for pid in new_pids:
                    counts[pid] -= 1

    return "".join(map(str, word))
