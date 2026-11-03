import time
import sys
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Recursive construction
    word = [1]
    for m in range(2, n + 1):
        # Extract unique permutations of length m-1 from word
        # in order of appearance
        perms = []
        seen = set()
        # word length is L. windows of size m-1
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            # Check if it's a permutation (distinct elements)
            # Since word is built from permutations, this should hold for most,
            # but the recursive construction ensures we only pick valid perms?
            # Actually, the seed checks len(set(window)) == m-1.
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            # Block: p[0]...p[m-2], m, p[0]...p[m-2]
            block = list(p) + [m] + list(p)
            
            # Find max overlap with new_word
            max_overlap = 0
            # The overlap must be a suffix of new_word and prefix of block
            # Max possible overlap is min(len(new_word), len(block))
            # But usually small.
            # We can check from largest possible down.
            # However, block ends with p, and new_word ends with previous block's p.
            # Overlap is limited by the structure.
            # Efficient check:
            limit = min(len(new_word), len(block))
            for o in range(limit, -1, -1):
                if o == 0:
                    break
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            
            new_word.extend(block[max_overlap:])
        
        word = new_word
        
        # Time check
        if time.time() > deadline:
            break
    
    # If we finished the loop, we have a valid superpermutation.
    # If we broke early, we might not have all n symbols.
    # But for n=7, the loop is fast (4 iterations of heavy work).
    # The heavy work is for m=7.
    
    # Optimization: Try to shorten the word by removing redundant elements
    # This is a heuristic that may not always work, but can help.
    # We'll try a simple local search: for each position, if removing the element
    # still leaves all permutations present, remove it.
    
    # First, check if we have a valid superpermutation for n
    if len(word) >= n:
        # Check if all permutations are present
        all_perms = set(itertools.permutations(range(1, n+1)))
        present_perms = set()
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i+n])
            if len(set(window)) == n:
                present_perms.add(window)
        
        if present_perms == all_perms:
            # Try to shorten the word
            # We'll do a few passes of local search
            for _ in range(3):
                changed = False
                for i in range(1, len(word) - 1):
                    # Try removing word[i]
                    new_word = word[:i] + word[i+1:]
                    # Check if all permutations are still present
                    present_perms_new = set()
                    for j in range(len(new_word) - n + 1):
                        window = tuple(new_word[j:j+n])
                        if len(set(window)) == n:
                            present_perms_new.add(window)
                    
                    if present_perms_new == all_perms:
                        word = new_word
                        changed = True
                        break
                if not changed:
                    break
    
    return "".join(map(str, word))
