<<<<<<< SEARCH
"""Shortest superpermutation construction with optimization."""
import time
from itertools import permutations

def construct(n):
    """Return the shortest known superpermutation word for 1..n."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    if n <= 5:
        # Classic construction is optimal for n <= 5
        return _classic_superperm(n)
    
    # For n >= 6, start with classic and try to optimize
    word = _classic_superperm(n)
    
    # Try to shorten by removing redundant characters
    word = _optimize_superperm(word, n, deadline)
    
    return word

def _classic_superperm(n):
    """Generate classic recursive superpermutation."""
    word = [1]
    for m in range(2, n + 1):
        # Get all permutations of current word
        perms = []
        seen = set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build new word
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap with current end
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[-o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    return "".join(map(str, word))

def _get_all_perms_set(word, n):
    """Get set of all n-permutations in word."""
    perms = set()
    for i in range(len(word) - n + 1):
        window = word[i:i+n]
        # Check if it's a valid permutation of 1..n
        if len(set(window)) == n and all(c in '1234567'[:n] for c in window):
            perms.add(window)
    return perms

def _optimize_superperm(word, n, deadline):
    """Try to shorten word by removing redundant characters."""
    if len(word) < n:
        return word
    
    # Get initial set of permutations
    initial_perms = _get_all_perms_set(word, n)
    
    # Try removing each character and check if all perms are still present
    # Use a greedy approach: try to remove from left to right
    result = list(word)
    
    # For efficiency, we'll track which permutations are present
    # and try to remove characters that don't affect coverage
    
    i = 0
    while i < len(result) and time.time() < deadline:
        # Try removing character at position i
        if i < len(result):
            # Check if removing this char still preserves all permutations
            test_word = result[:i] + result[i+1:]
            
            # Quick check: if the char is part of any unique permutation, skip
            # More efficient: just check if all perms are still present
            if len(test_word) >= n:
                test_perms = _get_all_perms_set("".join(test_word), n)
                if test_perms == initial_perms:
                    # Safe to remove
                    result.pop(i)
                    # Don't increment i, check same position again
                    continue
            
            i += 1
    
    return "".join(result)
=======
"""Shortest superpermutation construction with probabilistic greedy optimization."""
import time
import random
from itertools import permutations
from collections import defaultdict

def construct(n):
    """Return the shortest known superpermutation word for 1..n."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    if n <= 5:
        # Classic construction is optimal for n <= 5
        return _classic_superperm(n)
    
    # For n >= 6, use probabilistic greedy construction
    best_word = _probabilistic_greedy_superperm(n, deadline)
    
    # Try to further optimize by removing redundant characters
    best_word = _optimize_superperm(best_word, n, deadline)
    
    return best_word

def _classic_superperm(n):
    """Generate classic recursive superpermutation."""
    word = [1]
    for m in range(2, n + 1):
        # Get all permutations of current word
        perms = []
        seen = set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build new word
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap with current end
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[-o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    return "".join(map(str, word))

def _get_all_perms_set(word, n):
    """Get set of all n-permutations in word."""
    perms = set()
    for i in range(len(word) - n + 1):
        window = word[i:i+n]
        # Check if it's a valid permutation of 1..n
        if len(set(window)) == n and all(c in '1234567'[:n] for c in window):
            perms.add(window)
    return perms

def _probabilistic_greedy_superperm(n, deadline):
    """
    Probabilistic greedy construction with urgency-based symbol selection.
    
    At each step, we append the symbol that maximizes:
    - Number of newly completed permutations
    - Minus an urgency penalty for symbols whose remaining completions are scarce
    
    This balances immediate coverage with future feasibility.
    """
    all_perms = set(permutations(range(1, n+1)))
    total_perms = len(all_perms)
    
    # Start with a random permutation
    initial_perm = list(random.sample(range(1, n+1), n))
    word = list(initial_perm)
    covered = {tuple(initial_perm)}
    
    # Track which permutations are "pending" - those that share a prefix/suffix with current end
    # and could be completed soon
    
    while len(covered) < total_perms and time.time() < deadline:
        # Get current suffix of length n-1
        if len(word) < n - 1:
            # Should not happen, but handle gracefully
            break
        
        suffix = tuple(word[-(n-1):])
        
        # For each possible next symbol, compute:
        # 1. How many new permutations would be completed
        # 2. Urgency penalty based on scarcity
        
        best_score = -1
        best_symbol = None
        best_new_perms = 0
        
        for sym in range(1, n+1):
            if sym in suffix:
                continue  # Can't repeat a symbol in a permutation
            
            # What new permutation would be formed?
            new_perm = suffix + (sym,)
            
            # Count new permutations (just this one for now)
            new_count = 1 if new_perm not in covered else 0
            
            # Compute urgency: how many permutations have this suffix as prefix?
            # i.e., how many permutations start with suffix + sym?
            urgency_penalty = 0
            remaining_with_suffix = 0
            for perm in all_perms:
                if perm not in covered and perm[:n-1] == suffix:
                    remaining_with_suffix += 1
            
            # If there are many remaining permutations with this suffix,
            # we're not in a hurry - lower penalty
            # If few remain, higher penalty to prioritize completing them
            if remaining_with_suffix > 0:
                urgency_penalty = 1.0 / remaining_with_suffix
            else:
                urgency_penalty = 0.0
            
            score = new_count - urgency_penalty
            
            if score > best_score or (score == best_score and new_count > best_new_perms):
                best_score = score
                best_symbol = sym
                best_new_perms = new_count
        
        if best_symbol is None:
            # No valid symbol - this shouldn't happen in a valid superpermutation
            # but handle gracefully by picking a random unused symbol
            used = set(suffix)
            available = [s for s in range(1, n+1) if s not in used]
            if available:
                best_symbol = random.choice(available)
            else:
                break
        
        word.append(best_symbol)
        
        # Update covered permutations
        if len(word) >= n:
            new_perm = tuple(word[-n:])
            if new_perm in all_perms:
                covered.add(new_perm)
    
    return "".join(map(str, word))

def _optimize_superperm(word, n, deadline):
    """Try to shorten word by removing redundant characters."""
    if len(word) < n:
        return word
    
    # Get initial set of permutations
    initial_perms = _get_all_perms_set(word, n)
    
    # Try removing each character and check if all perms are still present
    # Use a greedy approach: try to remove from left to right
    result = list(word)
    
    # For efficiency, we'll track which permutations are present
    # and try to remove characters that don't affect coverage
    
    i = 0
    while i < len(result) and time.time() < deadline:
        # Try removing character at position i
        if i < len(result):
            # Check if removing this char still preserves all permutations
            test_word = result[:i] + result[i+1:]
            
            # Quick check: if the char is part of any unique permutation, skip
            # More efficient: just check if all perms are still present
            if len(test_word) >= n:
                test_perms = _get_all_perms_set("".join(test_word), n)
                if test_perms == initial_perms:
                    # Safe to remove
                    result.pop(i)
                    # Don't increment i, check same position again
                    continue
            
            i += 1
    
    return "".join(result)
>>>>>>> REPLACE
