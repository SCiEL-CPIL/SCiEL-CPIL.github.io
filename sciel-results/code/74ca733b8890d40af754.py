"""Improved superpermutation construction using better overlap strategies."""
# EVOLVE-BLOCK-START
import time
import itertools
import math

def construct(n):
    deadline = time.time() + 110  # safety margin
    
    if n <= 1:
        return "1"
    
    # 1. Construct the classic superpermutation
    word = [1]
    
    for m in range(2, n + 1):
        # Extract all (m-1)-permutations from current word in order of first appearance
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build new word
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            
            new += block[overlap:]
        
        word = new
    
    # Convert to string for faster manipulation
    w_str = "".join(map(str, word))
    
    # 2. Greedy Deletion Search
    # Target: reduce length while maintaining validity
    target_len = math.factorial(n)
    
    def is_valid(s):
        if len(s) < n:
            return False
        seen = set()
        # Optimization: early exit if we find all permutations
        for i in range(len(s) - n + 1):
            seen.add(s[i:i+n])
            if len(seen) == target_len:
                return True
        return len(seen) == target_len

    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try to delete each character
        # Iterate from start to end
        for i in range(len(w_str)):
            # Quick check: if time is running out, break
            if time.time() > deadline:
                break
            
            candidate = w_str[:i] + w_str[i+1:]
            if is_valid(candidate):
                w_str = candidate
                changed = True
                break # Restart scan from beginning
    
    return w_str
# EVOLVE-BLOCK-END
