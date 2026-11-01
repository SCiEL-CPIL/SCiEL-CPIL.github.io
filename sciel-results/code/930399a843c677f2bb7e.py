import time
import random
import math
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0

    def time_remaining():
        return deadline - time.time()

    # 1. Standard recursive construction
    # For n=7, this gives 5913
    # For small n this is optimal or near-optimal
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
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n < 5:
        return "".join(map(str, word))

    # Precompute factorials and mapping functions
    factorials = [1] * (n + 1)
    for i in range(1, n + 1):
        factorials[i] = factorials[i-1] * i
    
    target_count = factorials[n]
    full_mask = (1 << target_count) - 1

    def perm_to_index(p):
        index = 0
        used = [False] * (n + 1)
        for i in range(n):
            count = 0
            for j in range(1, n + 1):
                if not used[j] and j < p[i]:
                    count += 1
            index += count * factorials[n - 1 - i]
            used[p[i]] = True
        return index

    def is_valid(w):
        if len(w) < n:
            return False
        mask = 0
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                idx = perm_to_index(window)
                mask |= (1 << idx)
                if mask == full_mask:
                    return True
        return mask == full_mask

    # Optimization Phase: Remove redundant symbols
    # We use a greedy deletion approach with time-limited randomization.
    # We attempt to remove characters one by one, restarting the scan if successful.
    
    current_word = word
    current_len = len(current_word)
    
    # 2. Aggressive Deletion Loop
    # We try to remove symbols. If successful, we continue.
    # We use randomization to escape local minima (though for superpermutations,
    # the structure is quite rigid, so deterministic scan order might be better,
    # but random order helps find different valid configurations).
    
    max_attempts = 1000
    attempts = 0
    no_progress_count = 0
    
    while time_remaining() > 5.0 and attempts < max_attempts:
        improved = False
        # Shuffle indices to try different positions
        indices = list(range(len(current_word)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 5.0:
                break
            
            # Check if we can remove index i
            # Optimization: Only check validity if we might have made progress?
            # Actually, we just try removal and check validity.
            
            # A simple heuristic:
            # A symbol at position i is in windows starting at i-n+1 ... i.
            # If any of these windows corresponds to a unique permutation that appears
            # ONLY here, removal is impossible.
            # Checking uniqueness is expensive.
            
            new_word = current_word[:i] + current_word[i+1:]
            
            if is_valid(new_word):
                current_word = new_word
                improved = True
                no_progress_count = 0
                break # Restart scan with new indices
            else:
                no_progress_count += 1
                # If we've checked many indices without success, break inner loop?
                # Or continue?
                # Let's continue checking other indices in this pass.
                # But if no_progress_count is high, we might be stuck.
        
        if not improved:
            # If we went through all indices without improvement, we are likely done
            # or stuck. We can try one more random pass or stop.
            # Given the rigidity, if a full random pass fails, it's likely optimal
            # for this local structure.
            if no_progress_count > len(current_word):
                break
            attempts += 1
        else:
            attempts += 1

    return "".join(map(str, current_word))
