import time
import sys

def construct(n):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    if n <= 1:
        return "1"
    
    word = [1]
    
    for m in range(2, n + 1):
        # Check time
        if time.time() > deadline:
            break
            
        perms = []
        seen = set()
        
        # Extract unique permutations of length m-1 from the current word
        # The current word contains all permutations of 1..m-1
        # We need to find them in order of appearance
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            # A valid permutation of 1..m-1 must contain all distinct symbols
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
                
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap with the end of new_word
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            # We can optimize this search, but for n=7 it's small enough
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word += block[overlap:]
        
        word = new_word
        
    return "".join(map(str, word))
