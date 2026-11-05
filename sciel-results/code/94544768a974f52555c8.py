import itertools
import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Generate standard superpermutation
    word = [1]
    # Store the list of permutations at each step to allow reordering in the next step
    # Actually, we only need the list of perms for the *previous* step to build the next.
    # But for randomization, we need to be able to reorder the blocks of S_{n-1}.
    # So we should store the list of perms extracted from S_{n-1}.
    
    perms_list = []
    for m in range(2, n + 1):
        # Extract perms from current word (which is S_{m-1})
        # These are the blocks we will use to build S_m
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # We have the list of perms.
        # If m == n, we might want to try random orders.
        # But we need to build S_m now.
        
        # For m < n, just use standard order to build S_m.
        # For m == n, we can try multiple orders.
        
        best_word_for_m = None
        best_len = float('inf')
        
        if m < n:
            order = perms
            new = []
            for p in order:
                block = list(p) + [m] + list(p)
                overlap = 0
                max_overlap = min(len(new), len(block))
                # Optimize overlap check
                # block is short (2*m - 1)
                # new is long
                # We only need to check up to min(len(new), len(block)-1)
                # Actually, overlap is suffix of new matching prefix of block.
                # Max possible overlap is len(block) - 1?
                # No, if new ends with entire block, overlap is len(block).
                # But new is built from similar blocks.
                # Standard max overlap is m-2.
                # But let's be safe.
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            best_word_for_m = new
            best_len = len(new)
            
        else:
            # m == n
            # Try standard order first
            order = perms
            new = []
            for p in order:
                block = list(p) + [m] + list(p)
                overlap = 0
                max_overlap = min(len(new), len(block))
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            best_word_for_m = new
            best_len = len(new)
            
            # Try random orders if time permits
            # Number of perms is (m-1)!
            # For n=7, 720 perms.
            # Construction is O(N * M) where N is num perms, M is word len.
            # 720 * 6000 ~ 4.3M ops.
            # In Python, ~0.1s per construction?
            # We have 110s.
            # We can try ~500 random orders.
            
            # Pre-calculate blocks to avoid re-creating lists
            blocks = [list(p) + [m] + list(p) for p in perms]
            
            # Helper to build word from blocks
            def build_from_blocks(blocks_list):
                res = []
                for b in blocks_list:
                    overlap = 0
                    max_overlap = min(len(res), len(b))
                    # Optimization: only check up to m-2?
                    # Actually, for safety, check up to min(len(res), len(b))
                    # But we know structure.
                    for o in range(min(len(res), len(b)), -1, -1):
                        if o == 0 or res[len(res) - o:] == b[:o]:
                            overlap = o
                            break
                    res += b[overlap:]
                return res

            # Random search
            # Use a fixed seed for reproducibility? No, random is fine.
            # But we want to find the best.
            # Let's try a few.
            
            max_tries = 500
            for _ in range(max_tries):
                if time.time() > deadline - 5:
                    break
                
                random.shuffle(perms)
                # Note: we are shuffling the list of tuples.
                # We need to use the corresponding blocks.
                # Since blocks are in same order as perms, we can just shuffle blocks?
                # No, we shuffled perms. We should shuffle blocks in same way.
                # Easier: create list of (perm, block) and shuffle that.
                # Or just shuffle the blocks list directly?
                # Yes, the blocks are independent.
                
                # Wait, I created `blocks` from `perms`.
                # If I shuffle `perms`, I must shuffle `blocks` identically.
                # Better to shuffle `blocks` directly.
                
                shuffled_blocks = blocks[:]
                random.shuffle(shuffled_blocks)
                
                candidate = build_from_blocks(shuffled_blocks)
                if len(candidate) < best_len:
                    best_len = len(candidate)
                    best_word_for_m = candidate
                    # If we find the theoretical minimum (5905), we can stop?
                    # 5905 is best known.
                    if best_len <= 5905:
                        break

            word = best_word_for_m
        # End if m == n
        
        # For m < n, word is already set
        if m < n:
            word = best_word_for_m

    # After loop, word is S_n
    if time.time() > deadline:
        return "".join(map(str, word))
        
    # 2. Calculate frequency
    freq = {}
    L = len(word)
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        freq[p] = freq.get(p, 0) + 1
        
    # 3. Trim
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        
        if len(word) > n:
            first_perm = tuple(word[:n])
            if freq.get(first_perm, 0) > 1:
                freq[first_perm] -= 1
                word = word[1:]
                changed = True
                
        if len(word) > n:
            last_perm = tuple(word[-n:])
            if freq.get(last_perm, 0) > 1:
                freq[last_perm] -= 1
                word = word[:-1]
                changed = True
                
    return "".join(map(str, word))
