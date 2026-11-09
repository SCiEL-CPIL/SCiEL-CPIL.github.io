import time
import sys
import math
from itertools import permutations

def _classic_construct(n):
    """Classical recursive construction."""
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word

def _local_search_shorten(word, n, time_limit=10):
    """Try to shorten the word by finding and removing redundant segments.
    
    A position i is removable if removing word[i] doesn't destroy any permutation.
    We check all positions and remove those that are safe.
    """
    start_time = time.time()
    word = list(word)
    
    # Build set of all required permutations
    required_perms = set(permutations(range(1, n+1)))
    
    # Helper to check if a word contains all required permutations
    def contains_all(w):
        if len(w) < n:
            return False
        found = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if window in required_perms:
                found.add(window)
            if len(found) == len(required_perms):
                return True
        return len(found) == len(required_perms)
    
    # Try removing single characters
    changed = True
    while changed and (time.time() - start_time) < time_limit:
        changed = False
        for i in range(1, len(word) - 1):  # Don't remove first or last
            # Quick check: if removing position i, check if all perms still present
            # We can be smarter: only check perms that include position i
            # But for simplicity, let's try and verify
            test_word = word[:i] + word[i+1:]
            if contains_all(test_word):
                word = test_word
                changed = True
                break  # Restart from beginning after a change
    
    return word

def construct(n):
    """Construct the shortest known superpermutation for n symbols.
    
    Uses the classical recursive construction with local search optimization.
    """
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with classical construction
    word = _classic_construct(n)
    
    # Try local search to shorten
    if n <= 7 and (time.time() < deadline - 5):
        word = _local_search_shorten(word, n, time_limit=min(10, deadline - time.time() - 5))
    
    return "".join(map(str, word))
