import time
import random
import itertools
from collections import Counter

def construct(n):
    # Step 1: Generate the standard recursive superpermutation
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
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Step 2: Optimization via incremental deletion
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    def get_windows(w):
        return [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    
    def is_valid(w):
        if len(w) < n:
            return False
        windows = set(get_windows(w))
        return all_perms.issubset(windows)
    
    # Initial check
    if not is_valid(word):
        return "".join(map(str, word))
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s safety margin
    
    # Initialize window counts
    window_counts = Counter(get_windows(word))
    
    # Helper: check if all required permutations are present
    def all_present(counts):
        for p in all_perms:
            if counts.get(p, 0) == 0:
                return False
        return True
    
    if not all_present(window_counts):
        return "".join(map(str, word))
    
    # Incremental deletion: try to delete each position
    # We'll do multiple passes, each time trying to delete one character
    # at each position if valid
    
    L = len(word)
    pass_count = 0
    max_passes = 50
    
    while time.time() < deadline and pass_count < max_passes:
        pass_count += 1
        deleted_any = False
        L = len(word)
        
        # Try deleting from position 1 to L-2 (not first or last)
        for i in range(1, L - 1):
            if time.time() > deadline:
                break
            
            # Windows that will be removed (contain position i)
            # Old windows: start j where j <= i <= j+n-1
            # i.e., j in [max(0, i-n+1), min(i, L-n)]
            old_starts = []
            for j in range(max(0, i - n + 1), min(i, L - n) + 1):
                old_starts.append(j)
            
            # Windows that will be added (new windows formed after deletion)
            # After deleting position i, positions i+1..end shift left by 1
            # New windows: start j where the window uses the shifted positions
            # A new window starting at j (in new word) corresponds to:
            #   If j < i: uses old positions j..j+n-1, but position i is removed
            #     so it uses old positions j..i-1, i+1..j+n
            #   If j >= i: uses old positions j+1..j+n
            # 
            # Actually, let's think more carefully.
            # Old word: W[0], W[1], ..., W[L-1]
            # Delete W[i]. New word: W[0], ..., W[i-1], W[i+1], ..., W[L-1]
            # New length: L-1
            # 
            # New windows start at positions 0 to L-1-n = L-n-1
            # 
            # For a new window starting at position j in the new word:
            #   If j < i: the window is W[j], W[j+1], ..., W[i-1], W[i+1], ..., W[j+n]
            #     (note: skips W[i])
            #   If j >= i: the window is W[j+1], W[j+2], ..., W[j+n]
            #
            # So new windows that differ from old:
            #   - New windows starting at j where j < i and j+n >= i (i.e., j >= i-n+1)
            #     These are new windows that "span" the deletion point
            #   - Old windows starting at j where j >= i are gone (shifted)
            #   - Old windows starting at j where j+n-1 >= i and j < i are also affected
            #
            # Let me identify which windows change:
            # Old windows that contain position i: j in [i-n+1, i] (clamped to [0, L-n])
            # These old windows are removed.
            # 
            # New windows that are formed:
            #   For j in [max(0, i-n+1), i-1]: new window at j spans the gap
            #   For j in [i, L-n-1]: new window at j = old window at j+1
            #
            # So the net change:
            #   Remove old windows at starts: max(0,i-n+1) to min(i, L-n)
            #   Add new windows:
            #     For j in [max(0,i-n+1), i-1]: new window spanning the gap
            #     For j in [i, L-n-1]: this is old window at j+1, so no net change if we think of it as a shift
            #
            # Actually, it's easier to just compute the affected windows directly.
            
            # Remove old windows that contain position i
            to_remove = []
            for j in range(max(0, i - n + 1), min(i, L - n) + 1):
                w = tuple(word[j:j + n])
                to_remove.append(w)
            
            # Add new windows that span the deletion
            to_add = []
            for j in range(max(0, i - n + 1), i):
                # New window at position j in new word
                # It consists of: word[j:i] + word[i+1:j+n+1]
                if j + n <= L - 1:  # Need enough characters
                    new_w = word[j:i] + word[i + 1:j + n + 1]
                    to_add.append(tuple(new_w))
            
            # Check if removal is valid: no permutation count drops to 0
            valid = True
            for w in to_remove:
                if window_counts.get(w, 0) == 1:
                    # This permutation would be lost
                    valid = False
                    break
            
            if valid:
                # Apply the deletion
                # Update counts
                for w in to_remove:
                    window_counts[w] -= 1
                    if window_counts[w] == 0:
                        del window_counts[w]
                for w in to_add:
                    window_counts[w] = window_counts.get(w, 0) + 1
                
                # Actually delete the character
                del word[i]
                deleted_any = True
                L = len(word)
                # After deletion, subsequent indices shift, so break and restart
                break
        
        if not deleted_any:
            break
    
    # Final verification
    if is_valid(word):
        return "".join(map(str, word))
    else:
        # Fallback: return the original standard construction
        word_orig = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word_orig) - m + 2):
                window = tuple(word_orig[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                max_overlap = min(len(new), len(block))
                for o in range(max_overlap, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word_orig = new
        return "".join(map(str, word_orig))
