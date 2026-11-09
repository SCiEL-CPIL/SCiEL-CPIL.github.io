"""Improved superpermutation construction with better overlap optimization."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import permutations

def construct(n):
    """
    Constructs a superpermutation using the standard factorial construction,
    then attempts to shorten it by finding and removing redundant symbols.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Step 1: Build the standard construction
    word = _standard_construction(n, deadline)
    
    # Step 2: Try to shorten by removing redundant middle symbols
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
    Try to remove redundant symbols from the superpermutation.
    The standard construction has known redundancies that can be eliminated.
    """
    word = list(word)
    original_len = len(word)
    
    # Strategy: Try removing each position and check if still valid
    # But that's O(n^2 * n!) which is too slow. Instead, use known structure.
    
    # The standard construction for n=7 has specific patterns where
    # consecutive permutation blocks share more than just the last/first symbol.
    # We can try to find and exploit longer overlaps.
    
    # Alternative: Try removing positions from the end of each (m-1)-permutation block
    # where the next block starts with the same symbol
    
    improved = list(word)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(improved) - 1):
            if time.time() > deadline:
                break
            # Try removing position i
            candidate = improved[:i] + improved[i+1:]
            if len(candidate) < n:
                continue
            # Quick check: does it still contain all permutations?
            # This is expensive, so only do it if we think it might work
            if _quick_validity_check(candidate, n):
                improved = candidate
                changed = True
                break
        if time.time() > deadline:
            break
    
    if len(improved) < len(word):
        return improved
    return word

def _quick_validity_check(word, n):
    """Fast check if word still contains all n! permutations."""
    needed = set(permutations(range(1, n + 1)))
    found = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            found.add(window)
            if len(found) == len(needed):
                return True
    return len(found) == len(needed)
# EVOLVE-BLOCK-END
