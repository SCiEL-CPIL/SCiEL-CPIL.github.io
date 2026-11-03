"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
import random
from math import factorial
from itertools import permutations

def _classic_superperm(n):
    """Optimized recursive construction with better overlap handling."""
    if n <= 1:
        return [1]
    
    # Start with all permutations of 1..(n-1) in a specific order
    word = _classic_superperm(n - 1)
    
    # Extract all (n-1)-length windows that are permutations of 1..(n-1)
    perms, seen = [], set()
    for i in range(len(word) - n + 2):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    # Greedy merge: for each permutation, create block p + [n] + p
    # and append with maximum overlap
    new = list(word)
    for p in perms:
        block = list(p) + [n] + list(p)
        # Find maximum overlap between end of new and start of block
        max_overlap = min(len(new), len(block))
        overlap = 0
        for o in range(max_overlap, -1, -1):
            if o == 0 or new[len(new) - o:] == block[:o]:
                overlap = o
                break
        new += block[overlap:]
    
    return new

def _is_valid_superperm(word, n):
    """Check if word contains all permutations of 1..n as substrings."""
    if len(word) < n:
        return False
    
    # Use a set for O(1) lookups
    seen = set()
    n_perms = factorial(n)
    
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        # Check if it's a valid permutation (all distinct symbols)
        if len(set(w)) == n:
            seen.add(w)
            # Early exit if we've found all permutations
            if len(seen) == n_perms:
                return True
    
    return len(seen) == n_perms

def _greedy_extend(current, remaining, n, deadline):
    """
    Greedily extend current word to cover all remaining permutations.
    At each step, choose the next symbol that maximizes the number of new
    permutations covered, with tie-breaking by lookahead.
    """
    if time.time() > deadline:
        return None
    
    word = list(current)
    remaining = set(remaining)
    
    # Precompute: for each (last n-1 symbols) -> set of permutations starting with that
    # This helps us quickly know what's reachable
    
    while remaining:
        if time.time() > deadline:
            return None
        
        if len(word) < n - 1:
            # Not enough symbols yet, just append something
            # Try to start with a permutation
            best_next = 1
            for sym in range(1, n + 1):
                if sym not in word[-(n-1):]:
                    best_next = sym
                    break
            word.append(best_next)
            continue
        
        last = tuple(word[-(n - 1):])
        
        # Find all possible next symbols and score them
        best_next = None
        best_score = -1
        best_tiebreak = -1
        
        for sym in range(1, n + 1):
            # Check if adding sym creates a valid window
            if len(word) >= n - 1:
                new_window = last[1:] + (sym,)
                if len(set(new_window)) != n:
                    # Invalid window (duplicate symbols), skip unless we must
                    continue
            
            # Score: how many remaining permutations would be completed or progressed
            score = 0
            
            if len(word) >= n - 1:
                new_window = last[1:] + (sym,)
                
                # Does this complete a permutation?
                if len(set(new_window)) == n:
                    if new_window in remaining:
                        score += 100  # High reward for completing a permutation
                        remaining_copy = remaining - {new_window}
                    else:
                        score -= 50  # Penalty for duplicate window
                        remaining_copy = remaining
                else:
                    remaining_copy = remaining
                
                # Lookahead: how many remaining perms have this as prefix
                # (i.e., perms p where p[:n-1] == new_window)
                # This is expensive, so use a heuristic
                # Count remaining perms that start with the first few symbols of new_window
                prefix_len = min(3, n - 1)
                prefix = new_window[:prefix_len]
                cnt = 0
                for p in remaining_copy:
                    if p[:prefix_len] == prefix:
                        cnt += 1
                score += cnt * 0.1
            
            # Tie-break: prefer symbols that don't create immediate duplicates
            if len(word) >= 1 and sym == word[-1]:
                score -= 10  # Avoid immediate repetition
            
            if score > best_score or (score == best_score and sym < (best_next or n+1)):
                best_score = score
                best_next = sym
                best_tiebreak = sym
        
        if best_next is None:
            # Fallback: pick any symbol
            best_next = random.randint(1, n)
        
        word.append(best_next)
        
        # Remove any completed permutations
        if len(word) >= n:
            win = tuple(word[-n:])
            if len(set(win)) == n and win in remaining:
                remaining.discard(win)
    
    return word

def _local_search(word, n, deadline, max_iters=1000):
    """
    Try to shorten the word by removing redundant symbols.
    A symbol can be removed if no unique permutation depends on it.
    """
    if time.time() > deadline:
        return word
    
    word = list(word)
    n_perms = factorial(n)
    
    # Precompute which permutations are covered at which positions
    # perm_to_positions: dict from permutation tuple to set of start positions
    perm_to_positions = {}
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if len(set(w)) == n:
            if w not in perm_to_positions:
                perm_to_positions[w] = set()
            perm_to_positions[w].add(i)
    
    # A symbol at position j can be removed if for every permutation that uses it,
    # there's another position where that permutation appears
    # This is complex, so use a simpler heuristic:
    # Try removing each symbol and check validity
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing symbols from the middle (not the first n-1 or last n-1)
        for j in range(len(word) - n + 1, n - 1, -1):
            if time.time() > deadline:
                break
            
            # Try removing position j
            new_word = word[:j] + word[j+1:]
            if len(new_word) < n:
                continue
            
            # Quick check: does new_word still contain all perms?
            # This is expensive, so do a partial check
            # Check a sample of permutations
            sample_size = min(100, n_perms)
            if random.random() < 0.1:  # Only do full check sometimes
                if _is_valid_superperm(new_word, n):
                    word = new_word
                    improved = True
                    break
            else:
                # Partial check: verify a random sample
                all_perms = list(perm_to_positions.keys())
                sample = random.sample(all_perms, sample_size) if len(all_perms) > sample_size else all_perms
                valid = True
                for p in sample:
                    # Check if p is in new_word
                    p_str = tuple(p)
                    found = False
                    for i in range(len(new_word) - n + 1):
                        if tuple(new_word[i:i+n]) == p_str:
                            found = True
                            break
                    if not found:
                        valid = False
                        break
                if valid:
                    word = new_word
                    improved = True
                    break
    
    return word

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    if n <= 1:
        return "1" if n == 1 else ""
    
    # For small n, use exact or near-exact methods
    if n <= 4:
        word = _classic_superperm(n)
        return "".join(map(str, word))
    
    # Start with the classic construction
    word = _classic_superperm(n)
    best_word = list(word)
    best_len = len(best_word)
    
    # Try local search to shorten
    if time.time() < deadline - 10:
        shortened = _local_search(best_word, n, deadline - 5)
        if len(shortened) < best_len:
            # Verify
            if _is_valid_superperm(shortened, n):
                best_word = shortened
                best_len = len(best_word)
    
    # Try greedy construction with different starting permutations
    if n <= 6 and time.time() < deadline - 20:
        all_perms = list(permutations(range(1, n + 1)))
        random.shuffle(all_perms)
        
        # Try a few starting permutations with greedy extension
        for start_perm in all_perms[:50]:
            if time.time() > deadline - 5:
                break
            
            remaining = set(all_perms)
            remaining.discard(start_perm)
            
            result = _greedy_extend(list(start_perm), remaining, n, deadline - 5)
            if result is not None and len(result) < best_len:
                if _is_valid_superperm(result, n):
                    best_word = result
                    best_len = len(result)
    
    # Final verification
    if not _is_valid_superperm(best_word, n):
        # Fallback to classic
        best_word = _classic_superperm(n)
    
    return "".join(map(str, best_word))
# EVOLVE-BLOCK-END
