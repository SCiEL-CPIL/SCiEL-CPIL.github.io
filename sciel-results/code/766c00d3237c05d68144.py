import time
import sys
import math
import random
from collections import defaultdict

def construct(n):
    # 1. Build Patterson-Stockmeyer Superpermutation
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
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    if n <= 4:
        return "".join(map(str, word))
        
    # 2. Local Search
    start_time = time.time()
    time_limit = 110.0
    
    def get_window_counts(w):
        counts = defaultdict(int)
        for i in range(len(w) - n + 1):
            counts[tuple(w[i:i+n])] += 1
        return counts

    def try_remove_greedy(w):
        """
        Performs greedy deletion until no more removals are possible.
        Uses a single backward pass per iteration to avoid index issues.
        """
        counts = get_window_counts(w)
        changed = True
        while changed:
            changed = False
            L = len(w)
            # Iterate backwards
            for i in range(L - 1, -1, -1):
                # Check if removal is safe
                # Windows affected: start indices j in [i-n+1, i]
                # Valid j: 0 <= j <= L-n
                safe = True
                j_start = max(0, i - n + 1)
                j_end = min(i, L - n)
                
                # Quick check: if any window containing i has count 1
                for j in range(j_start, j_end + 1):
                    w_tuple = tuple(w[j:j+n])
                    if counts[w_tuple] == 1:
                        safe = False
                        break
                        
                if safe:
                    # Perform removal
                    # Identify old windows to decrement
                    old_windows = [tuple(w[j:j+n]) for j in range(j_start, j_end + 1)]
                    
                    # Identify new windows to increment
                    # New windows are formed at positions j in new indexing
                    # that span the removed index i.
                    # In old indexing, these correspond to windows starting at j
                    # where j < i and j+n-1 >= i.
                    # The new window content is w[j:i] + w[i+1:j+n]
                    new_windows = []
                    # j in new indexing: max(0, i-n+1) to min(i-1, L-1-n)
                    # Note: L-1-n is the max start index in the NEW word (length L-1)
                    # But we are using old indices to construct the window.
                    # The start index j in the new word corresponds to the same j in the old word
                    # for j < i.
                    # The condition for spanning is j < i and j+n-1 >= i.
                    # Also j must be a valid start index in the new word: j <= L-1-n.
                    # And j >= 0.
                    
                    new_j_end = min(i - 1, L - 1 - n)
                    for j in range(max(0, i - n + 1), new_j_end + 1):
                        # Check if it spans
                        if j < i and j + n - 1 >= i:
                            # Construct new window
                            # w[j:i] + w[i+1:j+n]
                            # Note: j+n in old indexing is the end of the window
                            # The window in old indexing would have been w[j:j+n]
                            # But we removed w[i].
                            # So we take w[j:i] and w[i+1:j+n]
                            nw = tuple(w[j:i] + w[i+1:j+n])
                            new_windows.append(nw)
                            
                    # Update counts
                    for ow in old_windows:
                        counts[ow] -= 1
                    for nw in new_windows:
                        counts[nw] += 1
                        
                    w.pop(i)
                    changed = True
                    # Continue loop, i will decrement
        return w

    # Initial Greedy Pass
    current_word = try_remove_greedy(word[:])
    best_len = len(current_word)
    best_word = current_word
    
    # If we are already at 5905 or better, we might stop or try to improve
    # But 5905 is the known best.
    
    # Perturbation Loop
    # If we didn't reach 5905, try to improve.
    # Even if we did, we can try to see if we can go lower (unlikely).
    
    # To ensure we finish in time, we limit the number of perturbations.
    max_perturbations = 50
    perturb_count = 0
    
    while time.time() - start_time < time_limit and perturb_count < max_perturbations:
        # Perturb
        w = best_word[:]
        
        # Randomly insert a character
        # We want to try to "unlock" deletions.
        # Inserting a character that matches a neighbor might help.
        # Let's just pick a random symbol and position.
        
        if len(w) == 0:
            break
            
        sym = random.randint(1, n)
        pos = random.randint(0, len(w))
        w.insert(pos, sym)
        
        # Try to shorten
        new_w = try_remove_greedy(w)
        
        if len(new_w) < best_len:
            best_len = len(new_w)
            best_word = new_w
            
        perturb_count += 1
        
    return "".join(map(str, best_word))
