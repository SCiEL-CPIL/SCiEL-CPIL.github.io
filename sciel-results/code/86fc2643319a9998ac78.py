"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
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
    
    # For n >= 4, use improved construction
    # Strategy: Build using the standard recursive structure but optimize overlaps
    # by trying alternative orderings of permutations at each level
    
    # The standard construction gives length sum(k! for k=1..n) = 5911 for n=7
    # We need to find a shorter valid superpermutation
    
    # Approach: Use a greedy algorithm that builds the word symbol by symbol,
    # always choosing the next symbol that maximizes new permutations covered
    # while keeping the partial word "extendable"
    
    if n == 4:
        # Known shortest is 33
        # Use a precomputed or search-based approach
        # For n=4, we can do a small search
        best_word = None
        best_len = 34  # standard gives 34
        
        # Try greedy with different starting permutations
        symbols = list(range(1, n+1))
        all_perms = list(permutations(symbols))
        perm_set = set(all_perms)
        
        # Greedy: start with a permutation, then greedily extend
        def greedy_construct(start_perm, time_limit):
            word = list(start_perm)
            covered = {tuple(word)}
            t0 = time.time()
            
            while len(covered) < n ** (n) and time.time() - t0 < time_limit:
                # Try each possible next symbol
                best_sym = None
                best_new_count = 0
                best_new_word = None
                for sym in symbols:
                    new_word = word + [sym]
                    new_window = tuple(new_word[-n:])
                    new_count = 0
                    if new_window not in covered:
                        new_count = 1
                    # Check if this extension is "good" - the last n-1 symbols
                    # should form a valid (n-1)-permutation that can be extended
                    last_n1 = tuple(new_word[-(n-1):])
                    if len(set(last_n1)) == n - 1:
                        # This is extendable
                        if new_count > best_new_count or (new_count == best_new_count and best_sym is None):
                            best_new_count = new_count
                            best_sym = sym
                            best_new_word = new_word
                    elif new_count > best_new_count:
                        # Even if not extendable in the standard way, count it
                        best_new_count = new_count
                        best_sym = sym
                        best_new_word = new_word
                if best_sym is None:
                    break
                word = best_new_word
                new_window = tuple(word[-n:])
                if new_window not in covered:
                    covered.add(new_window)
            
            if len(covered) == n ** n:
                return word
            return None
        
        # Try several starting permutations
        for start in all_perms[:50]:
            if time.time() > deadline:
                break
            result = greedy_construct(start, 5)
            if result and len(result) < best_len:
                best_len = len(result)
                best_word = result
                if best_len <= 33:
                    break
        
        if best_word:
            return "".join(map(str, best_word))
    
    # For n >= 5, use the standard construction as baseline
    # then try to improve with local search
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
    
    # Try to improve with a greedy symbol-by-symbol reconstruction
    # using the standard word as a guide but allowing deviations
    if time.time() < deadline - 30:
        # Verify the standard word is valid
        all_perms = set(permutations(range(1, n+1)))
        covered = set()
        for i in range(len(word) - n + 1):
            w = tuple(word[i:i+n])
            if w in all_perms:
                covered.add(w)
        
        if len(covered) == len(all_perms):
            # Standard word is valid, try to shorten
            # Use a greedy approach: rebuild the word, at each step
            # choose the symbol that covers the most uncovered permutations
            # and maintains validity
            
            symbols = list(range(1, n+1))
            uncovered = set(all_perms)
            
            # Start with the first n symbols of the standard word
            new_word = list(word[:n])
            first_window = tuple(new_word)
            if first_window in uncovered:
                uncovered.discard(first_window)
            
            t0 = time.time()
            while uncovered and time.time() - t0 < 80:
                best_sym = None
                best_score = -1
                best_new_uncovered = None
                
                for sym in symbols:
                    candidate = new_word + [sym]
                    new_window = tuple(candidate[-n:])
                    # Check if the new window is a valid permutation
                    if new_window not in all_perms:
                        # Still might be useful, but lower priority
                        score = 0
                    elif new_window in uncovered:
                        score = 2  # covers a new permutation
                    else:
                        score = 1  # covers already-covered
                    
                    # Bonus for keeping the last n-1 as a valid (n-1)-perm
                    last_n1 = tuple(candidate[-(n-1):])
                    if len(set(last_n1)) == n - 1:
                        score += 0.5
                    
                    if score > best_score:
                        best_score = score
                        best_sym = sym
                        if new_window in all_perms and new_window in uncovered:
                            best_new_uncovered = new_window
                
                if best_sym is None:
                    break
                
                new_word.append(best_sym)
                new_window = tuple(new_word[-n:])
                if new_window in uncovered:
                    uncovered.discard(new_window)
            
            if not uncovered and len(new_word) < base_len:
                word = new_word
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
