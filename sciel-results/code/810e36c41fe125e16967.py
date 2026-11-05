import itertools
import time
import random
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    if n == 1:
        return "1"
    if n == 2:
        return "121"

    # Step 1: Build standard superpermutation
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
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if time.time() > deadline - 5:
        return "".join(map(str, word))

    # Helper: Check if word is a valid superpermutation
    def is_valid(w):
        if len(w) < n:
            return False
        perms = set()
        for i in range(len(w) - n + 1):
            perms.add(tuple(w[i:i+n]))
        return len(perms) == len(list(itertools.permutations(range(1, n+1))))

    # Helper: Get frequency of all n-windows
    def get_freq(w):
        freq = defaultdict(int)
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            freq[p] += 1
        return freq

    # Initial freq
    freq = get_freq(word)
    
    # Helper: Check if deletion at pos is safe
    def is_deletion_safe(w, pos, freq, L):
        # Windows affected: [pos-n+1, pos]
        q_lo = max(0, pos - n + 1)
        q_hi = min(pos, L - n)
        
        # Store perms that are unique and affected
        unique_affected = []
        for q in range(q_lo, q_hi + 1):
            p = tuple(w[q:q+n])
            if freq[p] == 1:
                unique_affected.append(p)
        
        if not unique_affected:
            return True
            
        # For each unique affected perm, check if it reappears in the "gap"
        # New windows at w in [pos-n+1, pos-1] are w_gap = w[w:pos] + w[pos+1:w+n+1]
        # We need to see if any unique_affected perm is in the set of new gap windows
        # OR if it appears elsewhere (but freq==1 means it doesn't)
        
        # Actually, if freq[p]==1, it ONLY appears at q.
        # After deletion, it must appear in a new gap window.
        # Let's generate the set of new gap windows
        new_gap_perms = set()
        w_lo = max(0, pos - n + 1)
        w_hi = min(pos - 1, L - n - 1)
        for w in range(w_lo, w_hi + 1):
            # Construct window
            # w[w:pos] + w[pos+1:w+n+1]
            # Since n is small (7), we can just slice
            new_p = tuple(w[w:pos] + w[pos+1:w+n+1])
            new_gap_perms.add(new_p)
            
        for p in unique_affected:
            if p not in new_gap_perms:
                return False
        return True

    # Helper: Check if swap at i, i+1 is safe
    def is_swap_safe(w, i, freq, L):
        # Swap w[i] and w[i+1]
        # Affected windows: [i-n+1, i+1]
        q_lo = max(0, i - n + 1)
        q_hi = min(i + 1, L - n)
        
        # Create new word locally? No, just simulate the windows
        # We need to check if any perm drops to 0
        # And if any new perm is created that was 0? No, we just need all perms present.
        # Since we are swapping, the set of symbols is the same.
        # We just need to ensure no unique perm is destroyed.
        
        # Perms affected in original:
        affected_originals = []
        for q in range(q_lo, q_hi + 1):
            p = tuple(w[q:q+n])
            if freq[p] == 1:
                affected_originals.append(p)
                
        if not affected_originals:
            return True
            
        # Check if these perms appear in the new word
        # New windows at q in [q_lo, q_hi]
        # We can construct the new word segment or just the windows
        # Let's construct the new word for the affected region
        # Actually, easier: just do the swap in a copy and check freq?
        # L is ~6000. Copying is fast.
        new_w = w[:]
        new_w[i], new_w[i+1] = new_w[i+1], new_w[i]
        
        # Check if any perm in affected_originals is missing in new_w
        # We only need to check if they appear in the new windows at [q_lo, q_hi]
        # because they were unique, so they don't appear elsewhere.
        
        new_window_set = set()
        for q in range(q_lo, q_hi + 1):
            new_window_set.add(tuple(new_w[q:q+n]))
            
        for p in affected_originals:
            if p not in new_window_set:
                return False
        return True

    best_len = len(word)
    best_word = word[:]
    
    # Local Search Loop
    while time.time() < deadline - 10:
        L = len(word)
        if L < best_len:
            best_len = L
            best_word = word[:]
            
        # 1. Try Deletions
        safe_deletions = []
        for pos in range(1, L - 1):
            if is_deletion_safe(word, pos, freq, L):
                safe_deletions.append(pos)
                
        if safe_deletions:
            # Pick random one to avoid bias
            pos = random.choice(safe_deletions)
            word = word[:pos] + word[pos+1:]
            freq = get_freq(word)
            continue

        # 2. Try Swaps
        # Only try a few random swaps to save time
        swaps_tried = 0
        max_swaps = 50
        found_swap = False
        while swaps_tried < max_swaps and time.time() < deadline - 10:
            i = random.randint(0, L - 2)
            if is_swap_safe(word, i, freq, L):
                word[i], word[i+1] = word[i+1], word[i]
                freq = get_freq(word)
                found_swap = True
                break
            swaps_tried += 1
            
        if not found_swap:
            # Stuck. Try a "Jump": Randomly modify word and hope to fix?
            # Or just break.
            # Given the structure, we might be at a local minimum.
            # We can try to restart from best_word with a perturbation?
            # For simplicity, break.
            break

    # Final check
    if is_valid(best_word):
        return "".join(map(str, best_word))
    else:
        # Fallback
        return "".join(map(str, word))

