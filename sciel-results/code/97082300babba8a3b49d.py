import time
import sys
from itertools import permutations

def construct(n):
    """
    Constructs a superpermutation using the standard factorial construction,
    then shortens it by iteratively removing redundant symbols.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Step 1: Build the standard construction
    word = _standard_construction(n, deadline)
    
    # Step 2: Try to shorten by removing redundant symbols
    if time.time() < deadline:
        word = _attempt_shortening(word, n, deadline)
    
    return "".join(map(str, word))

def _standard_construction(n, deadline):
    """Build the standard 1!+2!+...+n! superpermutation."""
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
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
    return word

def _verify(word, n):
    """Check if word contains all n! permutations of 1..n as consecutive windows."""
    needed = set(permutations(range(1, n + 1)))
    found = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            found.add(window)
    return found == needed

def _attempt_shortening(word, n, deadline):
    """
    Iteratively remove redundant symbols from the superpermutation.
    A symbol at index i is redundant if removing it does not eliminate
    the last occurrence of any permutation of length n.
    """
    word = list(word)
    
    # Pre-compute counts of all permutations in the current word
    # Only count windows that are valid permutations (distinct symbols)
    perm_count = {}
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            perm_count[window] = perm_count.get(window, 0) + 1
            
    # Total number of required permutations
    total_perms = 1
    for i in range(1, n + 1):
        total_perms *= i
        
    # Iterate until no more removals are possible or deadline reached
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Scan from left to right
        for i in range(n - 1, len(word) - n + 1):
            if time.time() > deadline:
                break
            
            # Check if removal of word[i] is safe
            # A window contains index i if its start j satisfies i-n+1 <= j <= i
            # And j must be a valid window start: 0 <= j <= len(word)-n
            safe = True
            for j in range(max(0, i - n + 1), min(i, len(word) - n) + 1):
                window = tuple(word[j:j + n])
                if len(set(window)) == n:
                    if perm_count.get(window, 0) <= 1:
                        safe = False
                        break
            
            if safe:
                # Remove word[i]
                # Update perm_count
                # 1. Decrement counts for windows that included i
                for j in range(max(0, i - n + 1), min(i, len(word) - n) + 1):
                    window = tuple(word[j:j + n])
                    if len(set(window)) == n:
                        perm_count[window] -= 1
                        if perm_count[window] == 0:
                            del perm_count[window]
                
                # 2. Remove the element
                del word[i]
                
                # 3. Increment counts for new windows formed by the shift
                # The new windows are at indices j such that the window content
                # is different from the old ones.
                # Specifically, windows starting at j in the new word correspond to
                # word[j:i] + word[i+1:j+n] in the old word.
                # These new windows exist for j in range(max(0, i-n+1), min(i-1, len(word)-n) + 1)
                # Note: len(word) is now reduced by 1.
                new_len = len(word)
                for j in range(max(0, i - n + 1), min(i - 1, new_len - n) + 1):
                    window = tuple(word[j:j + n])
                    if len(set(window)) == n:
                        perm_count[window] = perm_count.get(window, 0) + 1
                
                changed = True
                break # Restart scan as indices have shifted
    
    return word

# EVOLVE-BLOCK-END
