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
