"""Shortest superpermutation on n symbols using the classic recursive construction."""
import time
import sys

# EVOLVE-BLOCK-START
def construct(n):
    """
    Construct a superpermutation for n symbols using the classic recursive method.
    Returns a string of digits 1..n.
    """
    if n <= 0:
        return ""
    if n == 1:
        return "1"
    
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Start with the base case
    word = [1]
    
    for m in range(2, n + 1):
        # Find all (m-1)-permutations in order of first appearance
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build the new word
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        
        word = new
        
        # Check time
        if time.time() > deadline:
            break
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
