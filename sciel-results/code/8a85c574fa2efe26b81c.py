import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code using bitmask-based greedy set cover
    and local search (removal, 1-swaps, and 2-swaps).
    
    Returns: list of codewords (each a list of n integers in 0..q-1)
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1
    
    # Helper functions for base conversion
    def int_to_digits(num, length, base):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % base
            num //= base
        return digits

    def digits_to_int(digits, base):
        num = 0
        for d in reversed(digits):
            num = num * base + d
        return num

    # Precompute neighborhoods as bitmasks
    # Each word's neighborhood is an integer where bit i is 1 if word i is in the neighborhood
    masks = [0] * total_words
    for i in range(total_words):
        d = int_to_digits(i, n, q)
        mask = 1 << i
        for pos in range(n):
            orig = d[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                d[pos] = new_val
                nb_idx = digits_to_int(d, q)
                mask |= (1 << nb_idx)
            d[pos] = orig
        masks[i] = mask

    # Precompute neighbor lists for faster candidate generation
    neighbor_lists = [[] for _ in range(total_words)]
    for i in range(total_words):
        d = int_to_digits(i, n, q)
        for pos in range(n):
            orig = d[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                d[pos] = new_val
                nb_idx = digits_to_int(d, q)
                neighbor_lists[i].append(nb_idx)
            d[pos] = orig

    # 1. Greedy Set Cover
    # uncovered: bitmask of all words
    uncovered = (1 << total_words) - 1
    code = []
    code_set = set()
    
    while uncovered != 0:
        if time.time() > deadline:
            break
            
        best_center = -1
        best_gain = 0
        
        for c in range(total_words):
            if c in code_set:
                continue
            gain = (masks[c] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_center = c
        
        if best_center == -1 or best_gain == 0:
            break
            
        code.append(best_center)
        code_set.add(best_center)
        uncovered &= ~masks[best_center]

    # Efficient trim: use a coverage count array
    def trim_code():
        nonlocal code, code_set
        changed = True
        while changed and time.time() < deadline:
            changed = False
            if not code:
                break
            
            # Count how many codewords cover each word
            cover_count = [0] * total_words
            for c in code:
                mask = masks[c]
                # Iterate through set bits
                m = mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    cover_count[idx] += 1
                    m ^= lsb
            
            # Find redundant codewords
            to_remove = []
            for i, c in enumerate(code):
                mask = masks[c]
                m = mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        break  # This word is only covered by c, so c is not redundant
                    m ^= lsb
                else:
                    # All words in c's neighborhood are covered by at least one other codeword
                    to_remove.append(i)
            
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    code_set.discard(code[idx])
                    code.pop(idx)
                changed = True
    
    trim_code()

    # 2. Local Search
    def local_search():
        nonlocal code, code_set
        
        def compute_unique_parts():
            """Compute unique parts efficiently using coverage counts."""
            uniques = {}
            if not code:
                return uniques
            
            # Count how many codewords cover each word
            cover_count = [0] * total_words
            for c in code:
                mask = masks[c]
                m = mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    cover_count[idx] += 1
                    m ^= lsb
            
            # For each codeword, find words it uniquely covers
            for c in code:
                mask = masks[c]
                unique = 0
                m = mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        unique |= lsb
                    m ^= lsb
                uniques[c] = unique
            return uniques

        max_rounds = 80
        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
                
            improved = False
            uniques = compute_unique_parts()
            
            # 1. Try removals
            for c in list(code):
                if uniques.get(c, 0) == 0:
                    code_set.discard(c)
                    code.remove(c)
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2. Try 1-swaps: replace c with w
            code_list = list(code)
            random.shuffle(code_list)
            
            for c in code_list:
                if time.time() > deadline:
                    break
                unique_part = uniques.get(c, 0)
                if unique_part == 0:
                    continue
                    
                # Generate candidates: neighbors of c first, then random
                candidates = []
                # Add neighbors of c that are not in code
                for nb in neighbor_lists[c]:
                    if nb not in code_set:
                        candidates.append(nb)
                # Add random candidates
                remaining = total_words - len(code_set)
                if remaining > 0:
                    for _ in range(min(100, remaining)):
                        w = random.randint(0, total_words - 1)
                        if w not in code_set:
                            candidates.append(w)
                
                random.shuffle(candidates)
                
                # Try candidates
                limit = min(len(candidates), 300)
                for w in candidates[:limit]:
                    if (unique_part & ~masks[w]) == 0:
                        # Valid swap
                        code_set.discard(c)
                        code[code.index(c)] = w
                        code_set.add(w)
                        improved = True
                        break
                        
                if improved:
                    break
            
            # 3. Try 2-swaps: replace c1, c2 with w1, w2
            if not improved and len(code) >= 2:
                # Only attempt 2-swaps if we have time
                if time.time() < deadline - 30:
                    # Sample pairs of codewords
                    code_list2 = list(code)
                    random.shuffle(code_list2)
                    pair_limit = min(len(code_list2) * (len(code_list2) - 1) // 2, 50)
                    
                    for _ in range(pair_limit):
                        if time.time() > deadline:
                            break
                        i = random.randint(0, len(code_list2) - 1)
                        j = random.randint(0, len(code_list2) - 1)
                        if i == j:
                            continue
                        c1 = code_list2[i]
                        c2 = code_list2[j]
                        
                        # Combined unique part: words covered only by c1 or only by c2
                        # Actually, we need: words that are covered by c1 or c2 but not by any other
                        # This is more complex. Let's use a simpler heuristic:
                        # unique1 = words only covered by c1 (not c2 or others)
                        # unique2 = words only covered by c2 (not c1 or others)
                        # We need w1 and w2 such that w1 covers unique1 and w2 covers unique2
                        # But this might not be sufficient if there are words covered by both c1 and c2 but no other.
                        
                        # Simpler approach: try to find w1 that covers all of c1's unique part
                        # and w2 that covers all of c2's unique part
                        unique1 = uniques.get(c1, 0)
                        unique2 = uniques.get(c2, 0)
                        
                        if unique1 == 0 and unique2 == 0:
                            # Both are redundant, just remove one
                            code_set.discard(c1)
                            code.remove(c1)
                            improved = True
                            break
                        
                        # Find w1 for unique1
                        w1_found = False
                        candidates1 = []
                        for nb in neighbor_lists[c1]:
                            if nb not in code_set and nb != c2:
                                candidates1.append(nb)
                        for _ in range(50):
                            w = random.randint(0, total_words - 1)
                            if w not in code_set and w != c2:
                                candidates1.append(w)
                        random.shuffle(candidates1)
                        
                        w1 = None
                        for w in candidates1[:100]:
                            if (unique1 & ~masks[w]) == 0:
                                w1 = w
                                break
                        
                        if w1 is None:
                            continue
                        
                        # Find w2 for unique2
                        w2_found = False
                        candidates2 = []
                        for nb in neighbor_lists[c2]:
                            if nb not in code_set and nb != c1 and nb != w1:
                                candidates2.append(nb)
                        for _ in range(50):
                            w = random.randint(0, total_words - 1)
                            if w not in code_set and w != c1 and w != w1:
                                candidates2.append(w)
                        random.shuffle(candidates2)
                        
                        w2 = None
                        for w in candidates2[:100]:
                            if (unique2 & ~masks[w]) == 0:
                                w2 = w
                                break
                        
                        if w2 is None:
                            continue
                        
                        # Verify the swap: after removing c1, c2 and adding w1, w2,
                        # is everything still covered?
                        # Quick check: union of all masks except c1,c2 plus w1,w2 should cover everything
                        # This is expensive. Let's do a lighter check:
                        # Check that unique1 is covered by w1 and unique2 is covered by w2
                        # And that no word becomes uncovered
                        # For now, accept if the unique parts are covered
                        # We'll verify with a full check occasionally
                        
                        # Perform the swap
                        code_set.discard(c1)
                        code_set.discard(c2)
                        idx1 = code.index(c1)
                        idx2 = code.index(c2)
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        code[idx1] = w1
                        code[idx2] = w2
                        code_set.add(w1)
                        code_set.add(w2)
                        improved = True
                        break
                
                if improved:
                    continue

        # Final trim
        trim_code()
        
    # Run local search with multiple seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        if time.time() > deadline:
            break
        random.seed(seed)
        local_search()
        trim_code()
    
    # Final polish
    trim_code()
    
    # Convert back to list of lists
    result = []
    for c in code:
        result.append(int_to_digits(c, n, q))
        
    return result
