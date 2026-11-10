"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct(n):
    """
    Constructs the shortest known superpermutation for n symbols using the classic recursive method.
    For n=7, this yields a length of 5913.
    """
    # Deadline guard: 150s limit, use 120s for safety
    start_time = time.time()
    deadline = start_time + 120.0

    # Classic recursive construction of length 1! + 2! + ... + n!
    # This is optimal for n <= 4, and the standard construction for n >= 5.
    # For n=7, length is 5913.
    
    if n == 1:
        return "1"
    
    word = [1]
    for m in range(2, n + 1):
        # Check time
        if time.time() > deadline:
            # Return current best (which is the full construction if it finished, 
            # but the loop is fast so this shouldn't trigger for n=7)
            break
            
        # Extract unique permutations of length m-1 from the current word
        # These are the "skeletons" for the next step
        perms, seen = [], set()
        # The windows of length m-1 in `word` that contain all distinct symbols
        # are the permutations of 1..m-1 in the order they appear.
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            # Construct the block: p + [m] + p
            # Overlap with the end of `new` as much as possible
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap
            max_overlap = min(len(new), len(block))
            overlap = 0
            # Check from largest possible overlap down to 0
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
