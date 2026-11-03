"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations

def _classic_superperm(n):
    """Classic recursive construction of length 1! + 2! + ... + n!"""
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

def _is_valid_superperm(word, n):
    """Check if word contains all permutations of 1..n as consecutive substrings."""
    target_perms = set(permutations(range(1, n + 1)))
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        seen.add(window)
    return seen == target_perms

def _try_remove_chars(word, n, time_limit):
    """Try to remove redundant characters from word while maintaining validity."""
    start_time = time.time()
    word = list(word)
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(len(word) - 1, -1, -1):
            if (time.time() - start_time) >= time_limit:
                break
            # Try removing character at position i
            test_word = word[:i] + word[i + 1:]
            if _is_valid_superperm(test_word, n):
                word = test_word
                improved = True
                break  # Restart scan from beginning after a removal
    return word

def construct(n):
    # Use deadline guard
    start_time = time.time()
    time_limit = 100  # Leave 20s safety margin for 120s limit
    
    # Get classic superpermutation
    word = _classic_superperm(n)
    
    # Try to remove redundant characters
    word = _try_remove_chars(word, n, time_limit - (time.time() - start_time))
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
