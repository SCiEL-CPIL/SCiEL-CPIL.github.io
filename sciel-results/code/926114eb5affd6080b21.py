import math
import time
import sys

def construct(n):
    # Classic construction
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
            overlap = 0
            # Find max overlap
            max_o = min(len(new), len(block))
            for o in range(max_o, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Fast local search: try to delete chars
    if n <= 7:
        start_time = time.time()
        deadline = 140.0  # 10s margin before 150s kill
        target_count = int(math.factorial(n))
        
        # Build initial state
        N = len(word)
        wset = [set(word[i:i+n]) for i in range(N - n + 1)]
        perms = [tuple(word[i:i+n]) for i in range(N - n + 1) if len(wset[i]) == n]
        perm_set = set(perms)
        perm_count = len(perm_set)
        
        # Verify initial validity
        assert perm_count == target_count, f"Initial word invalid: {perm_count}/{target_count}"
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            N = len(word)
            num_windows = N - n + 1
            
            # Try deleting from end to start
            for i in range(N - 1, -1, -1):
                if time.time() > deadline:
                    break
                
                # Quick check: if removing position i destroys a unique permutation, skip
                # A window starting at j covers positions j..j+n-1
                # Position i is in windows starting at max(0, i-n+1) to min(i, num_windows-1)
                win_lo = max(0, i - n + 1)
                win_hi = min(i, num_windows - 1)
                
                # Check if any window containing position i is the unique occurrence of its perm
                skip = False
                for j in range(win_lo, win_hi + 1):
                    if len(wset[j]) == n:
                        p = tuple(word[j:j+n])
                        # Count occurrences of this perm
                        # We need to know if this is the only occurrence
                        # Use a count dict for efficiency
                        pass
                
                # For efficiency, maintain a count of each permutation
                # Let's use a different approach: just try the deletion and check affected windows
                
                # Try deletion
                new_word = word[:i] + word[i+1:]
                new_N = N - 1
                new_num_windows = new_N - n + 1
                
                # Check if new word is still valid
                # Only windows that changed are those starting at positions [i-n+1, i] (clamped)
                # A window starting at j in new_word corresponds to:
                #   if j < i: same as old window j
                #   if j >= i: corresponds to old window j+1 (shifted left)
                # Windows that are new/different: those starting at j where the window crosses position i
                # In new_word, window starting at j covers new_word[j:j+n]
                # If j < i and j+n > i: this window includes position i in old word -> changed
                # If j >= i: this window is old_word[j+1:j+1+n] -> shifted, might be new
                
                # Actually, let's just recompute the affected windows
                # Affected windows in new_word: those where j <= i and j+n-1 >= i (crossing the gap)
                # i.e., j in [max(0, i-n+1), min(i, new_num_windows-1)]
                
                # Check all new windows in the affected range
                new_perm_set = set()
                valid = True
                for j in range(max(0, i - n + 1), min(i, new_num_windows - 1) + 1):
                    w = new_word[j:j+n]
                    if len(set(w)) == n:
                        new_perm_set.add(tuple(w))
                
                # Also need to check that we didn't lose any permutations
                # The only way to lose a perm is if it was in a window that got destroyed
                # Windows in old word that contain position i: j in [win_lo, win_hi]
                # In new word, these windows are either:
                #   - shifted (if j >= i): become window j-1 in new word
                #   - crossing (if j < i and j+n > i): become a different window
                #   - destroyed if the window had a duplicate and now has a different char
                
                # Simpler: just check if new_word is a superpermutation by checking
                # that all target perms are still present
                # We can do this by checking the affected windows and seeing if any perm is missing
                
                # For efficiency, let's just do a full check only if the quick check passes
                # Quick check: if any window containing i is a valid perm, and it's the only occurrence, skip
                
                # Let's just try it - the affected window check is O(n^2) which is fine
                # But we need to verify ALL perms are still present
                # The only perms that could be lost are those in windows containing position i
                # So we check: for each valid perm in old windows containing i, is it still in new word?
                
                lost = False
                for j in range(win_lo, win_hi + 1):
                    if len(wset[j]) == n:
                        p = tuple(word[j:j+n])
                        # Check if p is still in new_word
                        # p could be in any window of new_word
                        # But since only windows near i changed, p is either:
                        # 1. In an unchanged window (j' < win_lo or j' > win_hi in old, same in new)
                        # 2. In a changed window near i
                        # So we just need to check if p appears in new_word's affected windows
                        # OR if p appears in an unaffected window
                        
                        # This is getting complex. Let's just do a targeted check.
                        # Check if p is in new_perm_set (affected windows) or in some unaffected window.
                        # For simplicity, just check affected windows + count.
                        
                        # Actually, the simplest correct approach:
                        # Build the full set of perms in new_word, but only recompute affected windows
                        pass
                
                # Let me take a cleaner approach: maintain a count of each permutation
                # and only recompute affected windows
                
                # Recompute affected windows in new_word
                affected = {}
                for j in range(max(0, i - n + 1), min(i, new_num_windows - 1) + 1):
                    w = new_word[j:j+n]
                    if len(set(w)) == n:
                        p = tuple(w)
                        affected[p] = affected.get(p, 0) + 1
                
                # Check if any perm that was in old affected windows is now missing
                # Old affected windows: j in [win_lo, win_hi]
                # For each valid perm in old affected windows, check if it's still present
                for j in range(win_lo, win_hi + 1):
                    if len(wset[j]) == n:
                        p = tuple(word[j:j+n])
                        # Count how many times p appears in new_word
                        # It appears in:
                        # 1. Unaffected windows: same count as before (minus any that shifted into affected)
                        # 2. Affected windows in new_word
                        # This is tricky. Let's just check if p is in affected or was in unaffected.
                        
                        # Simpler: just check if p appears in new_word at all
                        # Since only O(n) windows changed, if p was in many windows, it's still there
                        # If p was in exactly one window and that window was affected, check affected
                        pass
                
                # I'm overcomplicating this. Let's just do the full check but with early exit.
                # The full check is O(N*n) which is ~5905*7 = 41K operations per deletion attempt.
                # With 5905 positions, that's 242M operations per pass, too slow in Python.
                
                # Let's use a smarter approach: maintain a dict of perm -> count
                # and only update the affected windows
                
                # Actually, let me just use the simple approach but with a better is_superpermutation
                # that uses early termination and is optimized.
                
                # For now, let's just try the deletion and do a fast validity check
                # that only checks the affected windows and maintains a global count.
                
                # I'll restructure to maintain perm_counts
                break  # Break out and use a better approach below
        
        # Better approach: maintain perm_counts dict
        # Rebuild from scratch with proper tracking
        N = len(word)
        perm_counts = {}
        for i in range(N - n + 1):
            w = word[i:i+n]
            if len(set(w)) == n:
                p = tuple(w)
                perm_counts[p] = perm_counts.get(p, 0) + 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            N = len(word)
            num_windows = N - n + 1
            
            for i in range(N - 1, -1, -1):
                if time.time() > deadline:
                    break
                
                # Windows containing position i in old word: j in [max(0,i-n+1), min(i, num_windows-1)]
                win_lo = max(0, i - n + 1)
                win_hi = min(i, num_windows - 1)
                
                # Check if any perm with count 1 is in a window containing i
                # If so, we can't delete i without losing that perm
                can_delete = True
                for j in range(win_lo, win_hi + 1):
                    w = word[j:j+n]
                    if len(set(w)) == n:
                        p = tuple(w)
                        if perm_counts[p] == 1:
                            can_delete = False
                            break
                
                if not can_delete:
                    continue
                
                # Try deletion
                new_word = word[:i] + word[i+1:]
                new_N = N - 1
                new_num_windows = new_N - n + 1
                
                # Update perm_counts: remove old affected windows, add new affected windows
                # Old affected windows: j in [win_lo, win_hi]
                for j in range(win_lo, win_hi + 1):
                    w = word[j:j+n]
                    if len(set(w)) == n:
                        p = tuple(w)
                        perm_counts[p] -= 1
                        if perm_counts[p] == 0:
                            del perm_counts[p]
                
                # New affected windows in new_word: j in [max(0, i-n+1), min(i-1, new_num_windows-1)]
                # Wait, in new_word, position i is gone. Windows that cross the gap:
                # new_word[j:j+n] where j <= i-1 and j+n-1 >= i (in new_word coords)
                # In new_word, the gap is between positions i-1 and i (0-indexed)
                # A window starting at j covers j..j+n-1
                # It crosses the gap if j <= i-1 and j+n-1 >= i, i.e., j >= i-n+1
                # So j in [max(0, i-n+1), min(i-1, new_num_windows-1)]
                
                new_win_lo = max(0, i - n + 1)
                new_win_hi = min(i - 1, new_num_windows - 1)
                
                for j in range(new_win_lo, new_win_hi + 1):
                    w = new_word[j:j+n]
                    if len(set(w)) == n:
                        p = tuple(w)
                        perm_counts[p] = perm_counts.get(p, 0) + 1
                
                # Check if all target perms are still present
                if len(perm_counts) == target_count:
                    word = new_word
                    improved = True
                    break  # Restart scan
                else:
                    # Revert: we already updated perm_counts, need to undo
                    # This is getting complex. Let's just not commit and revert.
                    # Undo the changes
                    for j in range(new_win_lo, new_win_hi + 1):
                        w = new_word[j:j+n]
                        if len(set(w)) == n:
                            p = tuple(w)
                            perm_counts[p] -= 1
                            if perm_counts[p] == 0:
                                del perm_counts[p]
                    
                    for j in range(win_lo, win_hi + 1):
                        w = word[j:j+n]
                        if len(set(w)) == n:
                            p = tuple(w)
                            perm_counts[p] = perm_counts.get(p, 0) + 1
    
    return "".join(map(str, word))
