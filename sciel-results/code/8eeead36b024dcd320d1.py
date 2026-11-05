import time
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Standard recursive construction
    # This generates the word of length sum(k! for k in 1..n)
    # which is the known upper bound for the standard family.
    
    word = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            # Return what we have? No, it must be valid.
            # If we break early, it's invalid.
            # So we should not break early unless we have a valid fallback.
            # The standard construction is deterministic and fast.
            pass
            
        perms, seen = [], set()
        # Extract permutations of 1..m-1 from word
        # The window size is m-1
        # We need to find all distinct (m-1)-permutations in order of appearance
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            # Check if it's a valid permutation of 1..m-1
            # Since the previous word was a superpermutation, all windows of size m-1 
            # that contain distinct elements are valid permutations.
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
                
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap
            max_overlap = 0
            # The overlap can be at most min(len(new), len(block))
            # But practically, it's small.
            # We check from largest possible overlap down to 0
            # Optimization: The overlap must be a suffix of new and prefix of block.
            # Since block starts with p (perm of 1..m-1) and new ends with a permutation of 1..m-1,
            # the overlap is determined by how much the end of new matches the start of block.
            
            # Efficient overlap check
            # new ends with some sequence. block starts with p.
            # We can just check suffixes of new against prefixes of block.
            min_len = min(len(new), len(block))
            for o in range(min_len, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    max_overlap = o
                    break
                    
            new += block[max_overlap:]
            
        word = new
        
    # Convert to string
    return "".join(map(str, word))
