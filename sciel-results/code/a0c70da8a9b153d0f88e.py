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
    
    # Greedy reconstruction with symbol rotation heuristic
    symbols = list(range(1, n+1))
    uncovered = set(all_perms)
    
    # Start with first n symbols from standard word
    new_word = list(word[:n])
    first_window = tuple(new_word)
    if first_window in uncovered:
        uncovered.discard(first_window)
    
    # Track symbol usage frequency for balanced rotation
    sym_freq = {s: 0 for s in symbols}
    for s in new_word:
        sym_freq[s] += 1
    
    t0 = time.time()
    timeout = 100  # seconds for greedy
    
    while uncovered and time.time() - t0 < timeout:
        best_sym = None
        best_score = -1e18
        
        last_n1 = tuple(new_word[-(n-1):])
        last_n1_set = set(last_n1)
        
        for sym in symbols:
            new_window = last_n1 + (sym,)
            
            # Score components
            score = 0.0
            
            # Is this a valid permutation window?
            if new_window in all_perms:
                if new_window in uncovered:
                    score += 1000.0  # High priority: new permutation
                else:
                    score += 10.0  # Already covered, but valid
            else:
                score -= 100.0  # Invalid window, strong penalty
            
            # Extendability: all distinct symbols in window
            if len(set(new_window)) == n:
                score += 50.0
            
            # Symbol rotation: prefer symbols with lower frequency
            # This helps avoid getting stuck in local patterns
            score -= sym_freq[sym] * 0.1
            
            # Diversity bonus: prefer symbols not in the last n-1 positions
            if sym not in last_n1_set:
                score += 20.0
            
            if score > best_score:
                best_score = score
                best_sym = sym
        
        if best_sym is None:
            break
        
        new_word.append(best_sym)
        sym_freq[best_sym] += 1
        
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
