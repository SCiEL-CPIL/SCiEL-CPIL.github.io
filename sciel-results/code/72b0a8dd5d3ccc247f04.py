"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import math
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s limit
    
    if n <= 3:
        # Standard construction is optimal for n<=3
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
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # Build the standard recursive superpermutation
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
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    base_len = len(word)
    total_perms = math.factorial(n)
    
    # Verify standard word is valid
    all_perms = set(permutations(range(1, n+1)))
    covered = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if w in all_perms:
            covered.add(w)
    
    if len(covered) != total_perms:
        # Shouldn't happen, but return what we have
        return "".join(map(str, word))
    
    # Greedy reconstruction: build word symbol by symbol
    # Key insight: at each step, we want to choose a symbol that:
    # 1) Creates a new uncovered permutation (highest priority)
    # 2) Keeps the last (n-1) symbols as a valid (n-1)-permutation (for extendability)
    # 3) Among ties, prefer symbols that appear less recently (diversity)
    
    symbols = list(range(1, n+1))
    uncovered = set(all_perms)
    
    # Start with first n symbols from standard word
    new_word = list(word[:n])
    first_window = tuple(new_word)
    if first_window in uncovered:
        uncovered.discard(first_window)
    
    t0 = time.time()
    timeout = 100  # seconds for greedy
    
    while uncovered and time.time() - t0 < timeout:
        best_sym = None
        best_score = -1
        best_new_perm = None
        
        for sym in symbols:
            new_window = tuple(new_word[-(n-1):]) + (sym,)
            
            # Score components
            score = 0.0
            
            # Is this a valid permutation window?
            if new_window in all_perms:
                if new_window in uncovered:
                    score += 10.0  # High priority: new permutation
                    best_new_perm = new_window
                else:
                    score += 1.0  # Already covered, but valid
            else:
                score -= 5.0  # Invalid window, penalize
            
            # Extendability: last (n-1) of candidate should be a valid (n-1)-perm
            # This ensures we can continue building
            if len(set(new_window)) == n:
                # All distinct, good for future extensions
                score += 2.0
            
            # Diversity bonus: prefer symbols not in the last n-1 positions
            last_n1 = set(new_word[-(n-1):])
            if sym not in last_n1:
                score += 1.0
            
            if score > best_score:
                best_score = score
                best_sym = sym
                if new_window in uncovered:
                    best_new_perm = new_window
        
        if best_sym is None:
            break
        
        new_word.append(best_sym)
        new_window = tuple(new_word[-n:])
        if new_window in uncovered:
            uncovered.discard(new_window)
        
        # Early termination if we've covered all
        if not uncovered:
            break
    
    if not uncovered and len(new_word) < base_len:
        word = new_word
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
