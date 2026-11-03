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
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # If no perms found (shouldn't happen for m>=2 if word is valid superperm of m-1),
        # fallback to all perms?
        if not perms:
             perms = list(itertools.permutations(range(1, m)))
        
        # Order the permutations to maximize overlap
        # Greedy: Start with first perm, then pick the next one that maximizes overlap with current suffix
        # Current suffix is the last (m-1) elements of the word being built?
        # No, the overlap is between the *block* being added and the *previous* block.
        # We are building a sequence of blocks.
        # Let ordered_perms be the list.
        # We want to maximize sum of overlaps between consecutive blocks.
        
        # Overlap(p1, p2):
        # Block 1 ends with p1.
        # Block 2 starts with p2.
        # Overlap is longest suffix of p1 that is a prefix of p2.
        
        def get_overlap(p1, p2):
            max_ov = 0
            # p1 and p2 are tuples of length m-1
            # Check suffix of p1 vs prefix of p2
            for k in range(len(p1), 0, -1):
                if p1[-k:] == p2[:k]:
                    return k
            return 0

        # Greedy Ordering
        remaining = set(perms)
        # Start with the first perm (or any)
        current = perms[0]
        remaining.remove(current)
        ordered_perms = [current]
        
        while remaining:
            best_next = None
            best_ov = -1
            # To speed up, we might just check all
            for p in remaining:
                ov = get_overlap(current, p)
                if ov > best_ov:
                    best_ov = ov
                    best_next = p
            
            ordered_perms.append(best_next)
            remaining.remove(best_next)
            current = best_next
            
        # Construct new_word using ordered_perms
        new_word = []
        for p in ordered_perms:
            block = list(p) + [m] + list(p)
            
            # Find max overlap with new_word
            max_overlap = 0
            limit = min(len(new_word), len(block))
            # Optimization: We know the theoretical max overlap is m-1
            # But we can just check from min(limit, m-1) down to 0
            # Actually, the overlap can be larger?
            # Block ends with p. New word ends with previous block's p.
            # So overlap is between prev_p and curr_p.
            # Max overlap is m-1.
            check_limit = min(limit, m - 1)
            for o in range(check_limit, -1, -1):
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
    
    # Return the word
    return "".join(map(str, word))
