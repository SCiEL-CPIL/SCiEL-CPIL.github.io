import time
import math
from itertools import permutations

def _classic_superperm(n):
    """Build the classic superpermutation of length sum(k! for k=1..n)."""
    if n <= 1:
        return [1]
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_ov = min(len(new_word), len(block) - 1)
            for o in range(max_ov, -1, -1):
                if o == 0 or new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    return word

def _greedy_superperm(n, deadline):
    """
    Greedy least-recently-used superpermutation construction.
    Maintains a set of seen permutations (windows of size n).
    At each step, appends the symbol that:
      1. Is not in the current window of size n-1 (preferred), OR
      2. Is the least recently used symbol overall.
    Stops when all n! permutations have been seen.
    """
    if n <= 1:
        return [1]
    
    all_perms = set(permutations(range(1, n + 1)))
    total_perms = n ** 1 if n == 1 else math.factorial(n)
    
    word = []
    seen_perms = set()
    
    # Start with a seed window
    for s in range(1, n + 1):
        word.append(s)
    
    # Record the initial permutation
    if len(word) >= n:
        seen_perms.add(tuple(word[:n]))
    
    # Track last used time for each symbol
    last_used = {s: -1 for s in range(1, n + 1)}
    for i, s in enumerate(word):
        last_used[s] = i
    
    step = len(word)
    
    while len(seen_perms) < total_perms:
        if time.time() > deadline:
            break
        
        # Current window of size n-1 (the last n-1 symbols)
        if len(word) < n - 1:
            window = tuple(word)
        else:
            window = tuple(word[-(n-1):])
        
        # Determine which symbols to consider
        window_set = set(window)
        
        # Priority 1: symbols NOT in the current window
        # Priority 2: least recently used among those in the window
        
        candidates_not_in_window = [s for s in range(1, n + 1) if s not in window_set]
        candidates_in_window = [s for s in range(1, n + 1) if s in window_set]
        
        if candidates_not_in_window:
            # Among symbols not in window, pick the one that creates the most new permutations
            # or simply the least recently used
            best_sym = min(candidates_not_in_window, key=lambda s: last_used[s])
        else:
            # All symbols are in the window; pick least recently used
            best_sym = min(candidates_in_window, key=lambda s: last_used[s])
        
        # Check if appending best_sym creates a valid new window
        new_window = tuple(word[-(n-1):] + [best_sym])
        
        # Append the symbol
        word.append(best_sym)
        last_used[best_sym] = step
        step += 1
        
        # Check if we have a new permutation
        if len(word) >= n:
            perm = tuple(word[-n:])
            if perm not in seen_perms:
                seen_perms.add(perm)
        
        # Safety: if we're going in circles, break
        if step > total_perms * n * 2:
            break
    
    return word

def _verify_superperm(word, n):
    """Check if word is a valid superpermutation for n symbols."""
    if len(word) < n:
        return False
    all_perms = set(permutations(range(1, n + 1)))
    seen = set()
    for i in range(len(word) - n + 1):
        seen.add(tuple(word[i:i+n]))
    return seen == all_perms

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if n <= 1:
        return "1"
    
    # Try greedy first
    greedy_word = _greedy_superperm(n, deadline)
    
    # Verify greedy result
    if _verify_superperm(greedy_word, n):
        return "".join(map(str, greedy_word))
    
    # Fall back to classic
    classic_word = _classic_superperm(n)
    return "".join(map(str, classic_word))
