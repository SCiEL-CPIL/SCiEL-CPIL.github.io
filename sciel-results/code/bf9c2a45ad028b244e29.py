import time
import itertools
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # 1. Generate classic superpermutation efficiently
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
            # Overlap logic: find max overlap between end of new and start of block
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new)-o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    if n <= 4:
        return "".join(map(str, word))

    # Optimization: Use integer IDs for permutations for faster hashing and counting
    all_perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Convert word to list of ints (already is, but ensure types)
    word = [int(x) for x in word]
    
    # Initialize window IDs and counts
    L = len(word)
    window_ids = [0] * (L - n + 1)
    counts = [0] * num_perms
    
    for i in range(L - n + 1):
        w = tuple(word[i:i+n])
        wid = perm_to_id[w]
        window_ids[i] = wid
        counts[wid] += 1
    
    # Greedy deletion with efficient count updates and "best" candidate selection
    # We will iterate, finding all safe deletions, and picking the one that maximizes
    # the minimum count of the affected windows (most "safe" deletion).
    
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        best_index = -1
        best_min_count = -1
        
        # Find all safe indices
        for i in range(len(word)):
            if time.time() > deadline:
                break
            
            s_min = max(0, i - n + 1)
            s_max = min(i, len(word) - n)
            
            min_count = float('inf')
            safe = True
            for s in range(s_min, s_max + 1):
                wid = window_ids[s]
                if counts[wid] == 1:
                    safe = False
                    break
                if counts[wid] < min_count:
                    min_count = counts[wid]
            
            if safe:
                # Heuristic: Prefer deletion that leaves windows with higher counts (more robust)
                if min_count > best_min_count:
                    best_min_count = min_count
                    best_index = i
        
        if best_index != -1:
            i = best_index
            # Perform deletion
            s_min = max(0, i - n + 1)
            s_max = min(i, len(word) - n)
            
            # Decrement counts of removed windows
            for s in range(s_min, s_max + 1):
                wid = window_ids[s]
                counts[wid] -= 1
            
            # Remove element
            word.pop(i)
            L = len(word)
            
            # Update window_ids and counts for new windows
            # New windows start at i.
            # Old window at s+1 becomes new window at s.
            # So for s in [i, L-n]:
            #   new_window_id[s] = old_window_id[s+1]
            #   counts[new_window_id[s]] += 1 (because it's a new occurrence in the word structure? 
            #   Wait. The window CONTENT is the same as old window at s+1.
            #   The old window at s+1 is still in the word?
            #   Yes, if s+1 > i.
            #   The windows affected by deletion at i are those starting at [i-n+1, i].
            #   The windows starting at i+1 and beyond are shifted to i, i+1...
            #   Their CONTENTS are unchanged.
            #   However, in the `window_ids` array, the entry for index `s` now holds the window that was at `s+1`.
            #   Do we need to increment counts?
            #   No! The counts represent the number of times each permutation appears in the word.
            #   The set of windows in the new word is:
            #   {Windows starting at 0..i-n+1 (unchanged content, except those overlapping i)}
            #   {Windows starting at i..L-n (these are the old windows starting at i+1..L-n+1)}
            #   The old windows starting at i-n+1..i are REMOVED.
            #   The old windows starting at i+1..L-n+1 are SHIFTED to i..L-n.
            #   So the multiset of window contents is:
            #   (Old multiset) - (Removed windows) + (No new windows added, just shifted)
            #   Wait, this logic is slightly off.
            #   Let's trace:
            #   Word: A B C D E
            #   n=3. Windows: ABC, BCD, CDE.
            #   Delete B (i=1).
            #   New Word: A C D E.
            #   New Windows: ACD, CDE.
            #   Removed: ABC, BCD.
            #   Shifted: CDE (was at index 2, now at index 1).
            #   New window ACD did not exist before.
            #   So we MUST update counts for the "shifted" positions?
            #   No, the count of a specific permutation P is the number of indices j such that Word[j:j+n] == P.
            #   In the new word, the window at index 1 is CDE. This permutation was already counted in the old word (at index 2).
            #   So the count of CDE remains the same?
            #   No! The count is a property of the WORD.
            #   Old Word: ABC, BCD, CDE. Count(CDE)=1.
            #   New Word: ACD, CDE. Count(CDE)=1.
            #   So the count of CDE didn't change.
            #   The count of ABC and BCD became 0.
            #   The count of ACD became 1.
            #   
            #   So the algorithm should be:
            #   1. Decrement counts for windows removed (s_min to s_max).
            #   2. For the shifted windows (new indices i to L-n), their CONTENT is the same as old indices i+1 to L-n+1.
            #      So their counts should NOT change?
            #      Wait. The window at new index `s` has the same content as old index `s+1`.
            #      The old index `s+1` is still in the range of valid windows in the old word?
            #      Yes, provided `s+1 <= L_old - n`.
            #      Since `s <= L_new - n = L_old - 1 - n`, `s+1 <= L_old - n`.
            #      So the window at new index `s` corresponds exactly to the window at old index `s+1`.
            #      Therefore, the counts for these permutations do NOT change due to the shift.
            #      The ONLY changes are:
            #      - Windows overlapping the deleted position are removed.
            #      - A NEW window is formed?
            #      Let's check the "New window" ACD.
            #      In the example, ACD is at index 0.
            #      Index 0 is in `s_min=0` to `s_max=1`?
            #      s_min = max(0, 1-3+1) = -1 -> 0.
            #      s_max = min(1, 5-3) = 2.
            #      Range(0, 3) -> 0, 1, 2.
            #      Windows 0 (ABC), 1 (BCD), 2 (CDE) are "affected".
            #      We decremented counts for ABC, BCD, CDE.
            #      But CDE is still in the new word!
            #      So we over-decremented CDE.
            #      
            #   Correct Logic:
            #   A window is REMOVED if it contains the deleted character.
            #   A window is SHIFTED if it is to the right of the deleted character.
            #   A window is UNCHANGED if it is to the left.
            #   
            #   Windows containing `word[i]`:
            #   Start index `s` such that `s <= i < s+n` => `i-n+1 <= s <= i`.
            #   These windows are GONE. Decrement their counts.
            #   
            #   Windows NOT containing `word[i]`:
            #   `s < i-n+1` (Left) or `s > i` (Right).
            #   Left windows: Unchanged.
            #   Right windows: Shifted left by 1. Content unchanged.
            #   
            #   Are there NEW windows?
            #   The window at new index `i` is `word[i:i+n]` (new).
            #   This corresponds to `word_old[i+1 : i+1+n]`.
            #   This window was at old index `i+1`.
            #   Did it contain `word[i]`? No, it starts at `i+1`.
            #   So it was a "Right" window. It was counted in the old word.
            #   So its count should remain the same.
            #   
            #   So the ONLY update needed is:
            #   Decrement counts for windows `s` in `[i-n+1, i]`.
            #   Update `window_ids` array to reflect the shift for `s >= i`.
            #   `window_ids[s] = window_ids[s+1]` for `s` in `[i, L_new - n]`.
            #   And the last entry is removed.
            
            # Shift window_ids
            for s in range(i, L - n + 1):
                window_ids[s] = window_ids[s+1]
            window_ids.pop()
            
            improved = True
            # Restart scan as indices changed
    
    return "".join(map(str, word))
