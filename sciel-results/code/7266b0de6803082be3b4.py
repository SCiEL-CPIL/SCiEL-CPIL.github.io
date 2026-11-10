# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    total_words = q ** n
    
    # Map words to integers
    # 000000 -> 0
    # 000001 -> 1
    # ...
    # 222222 -> 728
    
    def word_to_int(w):
        v = 0
        for i in range(n):
            v = v * q + w[i]
        return v

    def int_to_word(v):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = v % q
            v //= q
        return w

    # Precompute neighbors as bitmasks
    # neighbors[i] is an integer with bits set for all words within Hamming distance <= radius
    neighbors = [0] * total_words
    
    # Generate all words as integers
    all_words_int = list(range(total_words))
    
    # Precompute neighbors
    for i in all_words_int:
        w = int_to_word(i)
        mask = 1 << i
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    w[pos] = new_val
                    j = word_to_int(w)
                    mask |= (1 << j)
                w[pos] = orig
        neighbors[i] = mask

    # Initialize with Direct Sum Code
    # Tetracode for first 4 coords: { (a, b, a+b, a+2b) | a,b in {0,1,2} }
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code = []
    if n >= 4:
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                init_code.append(word_to_int(h + list(tail)))
    else:
        init_code = all_words_int[:]
        
    init_code = list(set(init_code))

    # State
    code_set = set(init_code)
    coverage = np.zeros(total_words, dtype=np.int32)
    
    def add_codeword(c):
        code_set.add(c)
        # Update coverage
        mask = neighbors[c]
        # Iterate set bits in mask
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            coverage[idx] += 1
            m ^= lsb

    def remove_codeword(c):
        code_set.remove(c)
        mask = neighbors[c]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            coverage[idx] -= 1
            m ^= lsb

    # Initialize coverage
    for c in init_code:
        mask = neighbors[c]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            coverage[idx] += 1
            m ^= lsb

    def try_remove():
        """Remove any non-essential codeword. Returns True if removed."""
        removed = False
        candidates = list(code_set)
        random.shuffle(candidates)
        for c in candidates:
            if c not in code_set:
                continue
            # Check if c is essential
            mask = neighbors[c]
            m = mask
            essential = False
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if coverage[idx] == 1:
                    essential = True
                    break
                m ^= lsb
            if not essential:
                remove_codeword(c)
                removed = True
        return removed

    def try_swap_1_for_1():
        """Swap one codeword for another to enable reduction."""
        if not code_set:
            return False
        
        candidates = list(code_set)
        random.shuffle(candidates)
        
        for c_out in candidates:
            if c_out not in code_set:
                continue
            
            # Find words uniquely covered by c_out
            mask = neighbors[c_out]
            m = mask
            unique_words = []
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if coverage[idx] == 1:
                    unique_words.append(idx)
                m ^= lsb
            
            if not unique_words:
                continue
            
            # Find a candidate c_in that covers all unique_words
            # To speed up, we can pre-filter candidates that cover at least one unique word?
            # Or just iterate all non-code words.
            # 729 is small.
            
            # Optimization: Check if there is ANY word that covers all unique_words.
            # We can intersect the neighbor masks of the unique words? No, we need a codeword that IS a neighbor to all.
            # So we need c_in such that for all u in unique_words, c_in is in neighbors[u].
            # This is equivalent to c_in being in the intersection of neighbors[u] for all u in unique_words.
            
            intersection_mask = (1 << total_words) - 1
            for u in unique_words:
                intersection_mask &= neighbors[u]
                
            # Find a word in intersection_mask that is not in code_set
            # We can iterate through set bits of intersection_mask
            m = intersection_mask
            found = False
            while m:
                lsb = m & -m
                c_in = lsb.bit_length() - 1
                if c_in not in code_set:
                    # Perform swap
                    remove_codeword(c_out)
                    add_codeword(c_in)
                    return True
                m ^= lsb
            # If no candidate found, try next c_out
            
        return False

    def try_swap_2_for_1():
        """Replace 2 codewords with 1."""
        if len(code_set) < 2:
            return False
        
        candidates = list(code_set)
        random.shuffle(candidates)
        
        # Limit pairs to check
        max_pairs = 50
        for i in range(min(max_pairs, len(candidates))):
            for j in range(i + 1, min(max_pairs, len(candidates))):
                c1 = candidates[i]
                c2 = candidates[j]
                
                if c1 not in code_set or c2 not in code_set:
                    continue
                
                # Temporarily remove c1 and c2 to see what becomes uncovered
                # But we don't want to modify state unless we find a valid c_new
                
                # A word u is "critical" if it is covered by c1 or c2, and by NO other codeword.
                # coverage[u] == (1 if u in N(c1) else 0) + (1 if u in N(c2) else 0)
                
                critical_mask = 0
                # Check neighbors of c1
                m1 = neighbors[c1]
                while m1:
                    lsb = m1 & -m1
                    u = lsb.bit_length() - 1
                    # Check if u is covered by c2?
                    is_in_c2 = (neighbors[c2] >> u) & 1
                    # Expected coverage if c1, c2 are the only ones
                    expected = (1 if (neighbors[c1] >> u) & 1 else 0) + is_in_c2
                    if coverage[u] == expected:
                        critical_mask |= lsb
                    m1 ^= lsb
                    
                # Check neighbors of c2 (only those not in c1 to avoid double counting, but mask handles it)
                # Actually, just iterate all neighbors of c2 and check condition
                m2 = neighbors[c2]
                while m2:
                    lsb = m2 & -m2
                    u = lsb.bit_length() - 1
                    is_in_c1 = (neighbors[c1] >> u) & 1
                    expected = is_in_c1 + (1 if (neighbors[c2] >> u) & 1 else 0)
                    if coverage[u] == expected:
                        critical_mask |= lsb
                    m2 ^= lsb

                if critical_mask == 0:
                    continue
                
                # We need a c_new that covers all words in critical_mask
                # So c_new must be in intersection of neighbors[u] for all u in critical_mask
                intersection_mask = (1 << total_words) - 1
                m_crit = critical_mask
                while m_crit:
                    lsb = m_crit & -m_crit
                    u = lsb.bit_length() - 1
                    intersection_mask &= neighbors[u]
                    m_crit ^= lsb
                    
                # Find c_new in intersection_mask not in code_set
                m_new = intersection_mask
                found = False
                while m_new:
                    lsb = m_new & -m_new
                    c_new = lsb.bit_length() - 1
                    if c_new not in code_set:
                        remove_codeword(c1)
                        remove_codeword(c2)
                        add_codeword(c_new)
                        return True
                    m_new ^= lsb

        return False

    # Main Loop
    best_size = len(code_set)
    
    while time.time() < deadline:
        # Phase 1: Greedy Removal
        while try_remove():
            pass
        
        current_size = len(code_set)
        if current_size < best_size:
            best_size = current_size
        
        # Phase 2: Swaps
        improved = False
        
        # Try 1-for-1 swaps
        for _ in range(100):
            if try_swap_1_for_1():
                improved = True
                break
            if time.time() > deadline - 5:
                break
        
        # Try 2-for-1 swaps
        for _ in range(50):
            if try_swap_2_for_1():
                improved = True
                break
            if time.time() > deadline - 5:
                break
        
        if not improved:
            # If no swap worked, we are in a local minimum.
            # Try a random restart? Or just break?
            # Given the time limit, breaking is safer, but we can try a few more random 1-for-1 swaps
            # with a different random seed or just break.
            break

    # Final Code
    final_code = [int_to_word(c) for c in code_set]
    
    return final_code
# EVOLVE-BLOCK-END
