"""Superpermutation construction with deadline guard."""
import time
import sys

# EVOLVE-BLOCK-START

def construct(n):
    """
    Construct a superpermutation of length n.
    Returns a string of digits 1..n.
    """
    start_time = time.time()
    DEADLINE = 150.0
    MARGIN = 2.0  # safety margin in seconds

    # For n <= 4, use the classic recursive construction which is optimal
    # For n = 5, 6, 7, we need a more efficient approach
    # The classic construction gives length sum(i! for i in 1..n)
    # For n=7: 1!+2!+3!+4!+5!+6!+7! = 1+2+6+24+120+720+5040 = 5913
    # But the shortest known is 5905, so we need to do better.
    #
    # The classic construction is already close. Let's implement it first
    # and see if we can optimize within the time budget.
    
    if n <= 1:
        return "1"
    
    # Classic recursive construction
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
            # Find maximum overlap
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    result = "".join(map(str, word))
    
    # Check if we're within time budget
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - MARGIN:
        return result
    
    # For n >= 5, try to optimize the result
    # The classic construction has some redundancy that can potentially be removed
    # However, for n=7, the improvement from 5913 to 5905 is small (8 symbols)
    # and finding it requires sophisticated search.
    
    # Let's try a local optimization: look for redundant patterns
    # This is a heuristic and may not always improve the result
    
    if n >= 5 and len(result) > 0:
        # Try to find and remove redundant substrings
        # This is a simple heuristic
        optimized = _try_optimize(result, n, start_time, DEADLINE, MARGIN)
        if optimized is not None:
            return optimized
    
    return result


def _try_optimize(word, n, start_time, deadline, margin):
    """
    Attempt to optimize the superpermutation by removing redundant patterns.
    This is a heuristic approach.
    """
    elapsed = time.time() - start_time
    if elapsed > deadline - margin:
        return None
    
    # Convert to list for easier manipulation
    w = [int(c) for c in word]
    improved = True
    max_iterations = 1000
    
    for _ in range(max_iterations):
        elapsed = time.time() - start_time
        if elapsed > deadline - margin:
            break
        
        # Try to find patterns that can be shortened
        # One approach: look for sequences where a symbol appears unnecessarily
        # This is complex, so let's use a simpler heuristic
        
        # Check if removing any single character (not at ends) still gives a valid superpermutation
        # This is expensive, so we limit the search
        
        found_improvement = False
        for i in range(1, len(w) - 1):
            elapsed = time.time() - start_time
            if elapsed > deadline - margin:
                break
            
            # Try removing character at position i
            w_temp = w[:i] + w[i+1:]
            
            # Quick check: does this still contain all permutations?
            # This is expensive for n=7, so we do a partial check
            if _is_valid_partial(w_temp, n, start_time, deadline, margin):
                w = w_temp
                found_improvement = True
                break
        
        if not found_improvement:
            break
    
    if len(w) < len(word):
        return "".join(map(str, w))
    return None


def _is_valid_partial(w, n, start_time, deadline, margin):
    """
    Check if w is a valid superpermutation (all n! permutations of 1..n appear).
    Returns True if valid, False otherwise, or if time budget is exceeded.
    """
    elapsed = time.time() - start_time
    if elapsed > deadline - margin:
        return False
    
    if len(w) < n:
        return False
    
    # Check all windows of length n
    seen = set()
    total_perms = _factorial(n)
    
    for i in range(len(w) - n + 1):
        window = tuple(w[i:i+n])
        # Check if it's a valid permutation (all distinct, all in 1..n)
        if len(set(window)) != n:
            continue
        if any(x < 1 or x > n for x in window):
            continue
        seen.add(window)
        if len(seen) == total_perms:
            return True
        
        elapsed = time.time() - start_time
        if elapsed > deadline - margin:
            return False
    
    return len(seen) == total_perms


def _factorial(n):
    """Compute n! efficiently."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# EVOLVE-BLOCK-END
