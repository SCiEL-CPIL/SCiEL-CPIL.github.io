import time
import sys

# EVOLVE-BLOCK-START
def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    if n <= 1:
        return "1"
    
    word = [1]
    
    for m in range(2, n + 1):
        # Check deadline
        if time.time() > deadline:
            break
            
        perms, seen = [], set()
        # Extract all (m-1)-permutations from the current word in order of appearance
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            # Check if it's a permutation of 1..m-1
            # For efficiency, we can just check length and unique elements
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximal overlap
            # We want the largest o such that new[-o:] == block[:o]
            # Max possible overlap is min(len(new), len(block))
            max_overlap = min(len(new), len(block))
            overlap = 0
            # Iterate from largest possible overlap down to 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        
        word = new
        
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
