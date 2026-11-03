import time
import sys

def construct(n):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for Python overhead/exit
    
    # Standard recursive construction (Egan's algorithm)
    # This produces a superpermutation of length sum_{i=1}^n i!
    # For n=7, this is 5913.
    
    word = [1]
    for m in range(2, n + 1):
        # Check time
        if time.time() > deadline:
            # If we are here, we've likely already built most of it.
            # For n=7, the loop is fast. 
            # If n is large, this might be an issue, but task is n=7.
            pass
        
        perms, seen = [], set()
        # Extract all unique (m-1)-permutations from the current word
        # The current word contains all permutations of 1..m-1
        # We need to find them in order of appearance to maintain the recursive structure
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            # A valid window must have distinct elements
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        # For each permutation p of 1..m-1, we insert m to form p+m+p
        # and append it to the new word with maximum overlap
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap between end of 'new' and start of 'block'
            # Overlap length o: new[-o:] == block[:o]
            # Max possible overlap is min(len(new), len(block))
            # Since block length is 2*(m-1)+1 = 2m-1
            # And new ends with a permutation of 1..m-1 (length m-1)
            # The overlap is at most m-1 (the length of the permutation part)
            # Actually, the standard construction ensures that the overlap is exactly m-1 
            # if we look at the structure, but let's calculate it generally.
            
            max_overlap = min(len(new), len(block))
            overlap = 0
            # Optimization: The overlap must be a valid prefix of block and suffix of new.
            # Since new is built from permutations of 1..m-1, its suffix is a permutation.
            # Block starts with p, which is a permutation.
            # So we check if new ends with a prefix of block.
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            
            new += block[overlap:]
        word = new
        
    return "".join(map(str, word))
