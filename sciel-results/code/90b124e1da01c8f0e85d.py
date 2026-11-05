import itertools
import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Generate standard superpermutation
    word = [1]
    
    for m in range(2, n + 1):
        # Extract perms from current word (which is S_{m-1})
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Precompute blocks as tuples for faster comparison
        blocks = [tuple(list(p) + [m] + list(p)) for p in perms]
        
        def build_word_fast(blocks_seq, max_overlap_limit):
            """Build word from sequence of blocks with limited overlap search."""
            if not blocks_seq:
                return []
            res = list(blocks_seq[0])
            for b in blocks_seq[1:]:
                # Only check overlaps up to max_overlap_limit
                # Structural knowledge: max useful overlap is m-1
                limit = min(max_overlap_limit, len(res), len(b))
                overlap = 0
                for o in range(limit, -1, -1):
                    if o == 0 or tuple(res[len(res)-o:]) == b[:o]:
                        overlap = o
                        break
                res.extend(b[overlap:])
            return res
        
        if m < n:
            # For intermediate steps, use standard order with optimized overlap
            best_word_for_m = build_word_fast(blocks, m - 1)
            word = best_word_for_m
        else:
            # m == n: try standard order first
            best_word_for_m = build_word_fast(blocks, m - 1)
            best_len = len(best_word_for_m)
            
            # Try random orders if time permits
            n_blocks = len(blocks)
            
            # Convert to list of lists for easier shuffling
            blocks_list = [list(b) for b in blocks]
            
            # Precompute prefix tuples for faster comparison
            # For each block, precompute all prefixes as tuples
            block_prefixes = []
            for b in blocks_list:
                prefixes = [tuple(b[:i]) for i in range(len(b) + 1)]
                block_prefixes.append(prefixes)
            
            def build_word_fast_prefix(blocks_seq, max_overlap_limit):
                """Build word using precomputed prefixes."""
                if not blocks_seq:
                    return []
                res = list(blocks_seq[0])
                for idx in range(1, len(blocks_seq)):
                    b = blocks_seq[idx]
                    limit = min(max_overlap_limit, len(res), len(b))
                    overlap = 0
                    # Check from largest to smallest
                    for o in range(limit, -1, -1):
                        if o == 0:
                            overlap = 0
                            break
                        # Compare suffix of res with prefix of b
                        if tuple(res[len(res)-o:]) == b[:o]:
                            overlap = o
                            break
                    res.extend(b[overlap:])
                return res
            
            # Try more random shuffles with time budget
            max_tries = 2000
            for trial in range(max_tries):
                if time.time() > deadline - 5:
                    break
                
                shuffled = blocks_list[:]
                random.shuffle(shuffled)
                
                candidate = build_word_fast_prefix(shuffled, m - 1)
                if len(candidate) < best_len:
                    best_len = len(candidate)
                    best_word_for_m = candidate
                    if best_len <= 5905:
                        break

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
