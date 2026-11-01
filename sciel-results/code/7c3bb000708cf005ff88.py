"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import Counter

def construct(n):
    if n == 1:
        return "1"
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Classic recursive construction of superpermutation
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # For n <= 4, the classic construction is optimal, return immediately
    if n <= 4:
        return "".join(map(str, word))
    
    # For n >= 5, attempt greedy deletion to shorten the word
    # Precompute all permutations of 1..n
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Count how many times each permutation appears as a window
    perm_counts = Counter()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if window in all_perms:
            perm_counts[window] += 1
    
    # Incremental deletion logic
    # We maintain perm_counts.
    # To check if index i can be deleted:
    # 1. Identify all windows that contain index i.
    #    These are windows starting at s where i-n+1 <= s <= i.
    #    Also s must be valid: 0 <= s <= len(word)-n.
    # 2. For each such window W_s, if perm_counts[W_s] == 1, then deleting i removes W_s.
    #    If W_s is a permutation, we lose it. So deletion is unsafe.
    # 3. If all such windows have count > 1, deletion is SAFE (locally).
    #    Note: This check is sufficient for validity of the resulting word?
    #    Yes, because:
    #    - Windows not containing i are unchanged (or shifted, but content same).
    #    - Windows containing i are removed/changed.
    #    - If a window W_s containing i is removed, we need another window to cover that permutation.
    #    - If perm_counts[W_s] > 1, there is at least one other window covering it.
    #    - Does the "other" window survive?
    #      The other window W_t (t != s) does not contain i (by definition of "other" in the count? No, count is global).
    #      If W_t also contains i, then W_t is also removed.
    #      So we need to check if the *set* of windows containing i covers the permutation uniquely.
    #      My previous check: "if any window containing i has count 1".
    #      This is CORRECT.
    #      If W_s has count 1, and it contains i, then W_s is the ONLY window for that perm.
    #      Deleting i removes W_s. So the perm is lost.
    #      If ALL windows containing i have count > 1, then for any perm P covered by a window containing i,
    #      there is at least one other window covering P.
    #      Does that other window survive?
    #      The other window might ALSO contain i.
    #      Example: P is covered by W_s and W_t. Both contain i.
    #      If we delete i, both W_s and W_t are removed/changed.
    #      Then P is lost.
    #      So the check "count > 1" is NOT sufficient if multiple windows containing i cover the same perm.
    #      
    #      We need to check: For each perm P covered by any window containing i,
    #      is there a window covering P that does NOT contain i?
    #      
    #      This is more complex.
    #      Let's refine the check.
    
    def is_deletable(word, perm_counts, all_perms, n, i):
        # Get all windows containing i
        s_min = max(0, i - n + 1)
        s_max = min(i, len(word) - n)
        
        # Collect permutations covered by these windows
        perms_to_check = set()
        for s in range(s_min, s_max + 1):
            w = tuple(word[s:s + n])
            if w in all_perms:
                perms_to_check.add(w)
        
        # For each such perm, check if it has support outside the "affected zone"
        # Affected zone windows are those containing i.
        # We need to know if there is a window NOT containing i that covers the perm.
        # Total count is perm_counts[perm].
        # Count of windows containing i that cover this perm is:
        #   sum(1 for s in range(s_min, s_max+1) if tuple(word[s:s+n]) == perm)
        # If Total Count > Count_In_Zone, then there is support outside.
        
        for p in perms_to_check:
            count_in_zone = 0
            for s in range(s_min, s_max + 1):
                if tuple(word[s:s + n]) == p:
                    count_in_zone += 1
            if perm_counts[p] == count_in_zone:
                # All supports are in the zone (which will be destroyed/changed)
                return False
        return True
    
    def perform_deletion(word, perm_counts, all_perms, n, i):
        # Update counts
        s_min = max(0, i - n + 1)
        s_max = min(i, len(word) - n)
        
        # 1. Decrement counts for old windows in zone
        for s in range(s_min, s_max + 1):
            w = tuple(word[s:s + n])
            if w in all_perms:
                perm_counts[w] -= 1
                if perm_counts[w] == 0:
                    del perm_counts[w]
        
        # 2. Remove element
        word.pop(i)
        
        # 3. Add new window at i (if valid)
        # New window starts at i, length n
        # It uses word[i-n+1 ... i-1] and word[i+1 ... i+n-1] from OLD word?
        # No, word is already updated.
        # The new window is word[i:i+n] in the NEW word.
        # This corresponds to old word[i-n+1:i] + old word[i+1:i+n].
        # Just take tuple(word[i:i+n]) from new word.
        if 0 <= i <= len(word) - n:
            new_w = tuple(word[i:i + n])
            if new_w in all_perms:
                perm_counts[new_w] += 1
        
        # Note: Windows starting > i in new word are same as windows starting > i in old word (shifted).
        # Their counts are already correct because we didn't change the multiset of tail windows.
        # Wait, did we?
        # Old tail: W[i+1], W[i+2], ...
        # New tail: W'[i+1], W'[i+2], ... where W'[k] = W[k+1].
        # So the multiset of windows in the tail is identical.
        # So we don't need to update counts for s > i.
        
        # However, we decremented counts for s in [s_min, s_max].
        # s_max is i.
        # So we decremented W[i].
        # W[i+1] was NOT decremented.
        # But W[i+1] (old) becomes W'[i+1] (new)?
        # No, W'[i+1] = W[i+2] (old).
        # So W[i+1] (old) is now at W'[i] (new)?
        # No, W'[i] is the NEW window.
        # So W[i+1] (old) is LOST from the "tail" set?
        # Let's trace:
        # Old indices: 0, 1, ..., i, i+1, ..., L-n
        # New indices: 0, 1, ..., i, ..., L-n-1
        # W'[k] = W[k] for k < i.
        # W'[i] = New Window.
        # W'[k] = W[k+1] for k >= i.
        
        # So the set of windows in the new word is:
        # { W[0]...W[i-1] } U { New_Window } U { W[i+1]...W[L-n] }
        
        # The set of windows in the old word was:
        # { W[0]...W[i-1] } U { W[i] } U { W[i+1]...W[L-n] }
        
        # So we removed W[i] and added New_Window.
        # We did NOT remove W[i+1]. It is still in the set (as W'[i+1]).
        
        # So my update logic:
        # Decrement W[s] for s in [s_min, s_max].
        # This includes W[i], W[i-1], ..., W[i-n+1].
        # But W[i-1] is NOT removed! It is still in the new word (as W'[i-1]).
        # So we should NOT decrement W[i-1].
        
        # Correction:
        # Only windows that are REMOVED or CHANGED should have counts updated.
        # Windows W[s] for s < i-n+1 are unchanged.
        # Windows W[s] for s in [i-n+1, i-1] are CHANGED (because they contain w_i).
        # Window W[i] is REMOVED.
        # Windows W[s] for s > i are UNCHANGED (content-wise, just shifted).
        
        # So we must decrement counts for W[s] where s in [i-n+1, i].
        # And increment count for New_Window.
        # This is what I did.
        # But wait, W[s] for s < i is CHANGED, not removed.
        # The old W[s] is gone. The new W'[s] is different.
        # So yes, we must decrement old W[s] and increment new W'[s].
        
        # I only incremented New_Window (which is W'[i]).
        # I did NOT increment W'[s] for s < i.
        
        # So I need to update W'[s] for s in [i-n+1, i-1] as well.
        
        # Let's redo the update logic in perform_deletion.
        
        # Actually, it's safer to just recount the affected region.
        # Affected region: indices [i-n+1, i+n-1] in new word?
        # No, just [i-n+1, i] for starts.
        
        # Let's stick to the simple but slightly slower local recount for correctness.
        # Or fix the incremental logic.
        
        # Given the complexity, let's use a local recount for the affected windows.
        # This is O(n^2) per deletion.
        
        # Revert word? No, we already popped.
        # Let's just recount all windows in the range [max(0, i-n), min(len(word)-n, i+n)].
        
        # For simplicity and safety, let's just do a full recount if n is small,
        # or a very wide local recount.
        
        # For n=7, full recount is ~40k ops.
        # If we do 100 deletions, 4M ops. Fast.
        
        # So, in perform_deletion, after pop:
        perm_counts = Counter()
        for s in range(len(word) - n + 1):
            w = tuple(word[s:s + n])
            if w in all_perms:
                perm_counts[w] += 1
        
    changed = True
    while changed and (time.time() - start_time) < deadline - 5:
        changed = False
        i = 0
        while i < len(word):
            if time.time() - start_time > deadline - 5:
                break
            
            if is_deletable(word, perm_counts, all_perms, n, i):
                perform_deletion(word, perm_counts, all_perms, n, i)
                changed = True
                # i stays same, check new word[i]
            else:
                i += 1
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
