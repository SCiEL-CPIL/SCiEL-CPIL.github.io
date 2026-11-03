"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct(n):
    # Classic recursive construction of length 1! + 2! + ... + n!
    # This is the standard superpermutation construction.
    # For n=7, length is 5913.
    
    if n == 1:
        return "1"
        
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all unique (m-1)-permutations in order of appearance
        # A window of length m-1 is a permutation if it has m-1 distinct elements.
        # In the context of superpermutations, these windows are the "anchors" for the next layer.
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            # Check if it's a permutation of 1..m-1
            # Since we are building up, the elements are 1..m-1
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new = []
        for p in perms:
            # Block is p, then m, then p
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap with the end of 'new'
            # The overlap must be a suffix of 'new' and prefix of 'block'
            # Max possible overlap is len(p) = m-1
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            
            new += block[overlap:]
        word = new
        
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
