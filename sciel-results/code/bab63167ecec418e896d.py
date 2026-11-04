import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    total_words = q ** n
    
    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute balls (neighbors) as bitmasks
    balls = [0] * total_words
    
    # Precompute powers of q for fast index calculation
    powers = [1] * (n + 1)
    for i in range(1, n + 1):
        powers[i] = powers[i-1] * q

    for i in range(total_words):
        d = digits_of[i]
        mask = 1 << i
        
        if radius == 1:
            for pos in range(n):
                orig_digit = d[pos]
                for new_digit in range(q):
                    if new_digit == orig_digit:
                        continue
                    delta = (new_digit - orig_digit) * powers[n - 1 - pos]
                    idx = i + delta
                    mask |= (1 << idx)
        else:
            # General radius
            for r in range(1, radius + 1):
                for pos_comb in itertools.combinations(range(n), r):
                    orig_digits = [d[p] for p in pos_comb]
                    for variation in itertools.product(range(q), repeat=r):
                        is_diff = False
                        delta = 0
                        for k, pos in enumerate(pos_comb):
                            new_val = variation[k]
                            if new_val != d[pos]:
                                is_diff = True
                                delta += (new_val - d[pos]) * powers[n - 1 - pos]
                        if not is_diff:
                            continue
                        idx = i + delta
                        mask |= (1 << idx)
                        
        balls[i] = mask

    all_mask = (1 << total_words) - 1
    
    # Helper to get cover mask of a set
    def get_cover_mask(code_indices):
        mask = 0
        for c in code_indices:
            mask |= balls[c]
        return mask

    # Coverage count: how many codewords cover each word
    def compute_coverage_counts(code_indices):
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code_indices:
            # Add 1 to all positions in balls[c]
            mask = balls[c]
            # Efficient way: iterate over set bits? Or use numpy?
            # Since total_words is 6561, we can just loop
            # But better: use a precomputed list of neighbors?
            # Actually, let's just use the mask and count via bit operations is hard in numpy
            # Let's use a different approach: for each word, count how many codewords cover it
            # This is O(len(code) * total_words / word_size) which is fine for ~500 codewords
            pass
        # Alternative: for each word, count coverage
        # This is O(total_words * len(code))
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code_indices:
            m = balls[c]
            # Find set bits
            x = m
            while x:
                # Get lowest set bit
                lsb = x & (-x)
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                x ^= lsb
        return counts

    # Initial Solution: Linear Algebraic Construction
    # Use a small covering code and direct sum
    # For q=3, n=8, radius=1:
    # Use a (4,9,1) ternary code (all 9 words of length 4 is trivial, but we can do better)
    # Actually, the optimal (4,1,1) ternary code has size 3^4/3^0... wait.
    # K_3(4,1) = 9 (since 3^4 = 81, ball size 1 + 4*2 = 9, 81/9 = 9). So all 9 words.
    # That's not helpful.
    # Let's use the known optimal code for n=4? No, it's trivial.
    # Let's use a (3, 3, 1) code? K_3(3,1) = 3^3/7 = 27/7 ~ 4. So 5 words?
    # Actually, let's just use a greedy construction for the initial code, but with a twist:
    # We'll use a "structured" greedy that prefers codewords that are far apart.
    
    # Better idea: Use the van Laarhoven et al. construction? No, we don't have it.
    # Let's use a simple direct sum of a (4, 9, 1) code and a (4, 9, 1) code?
    # That gives 81*81 = 6561 words, which is the whole space. Useless.
    
    # Let's use a (4, 5, 1) code if it exists? K_3(4,1) lower bound is 9, so 5 is impossible.
    
    # Okay, let's stick with greedy but improve the local search.
    # The key insight from the inspiration is that the current local search is inefficient.
    # We'll use a "coverage count" based removal.
    
    # Greedy Construction
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = all_mask
        code = []
        
        while uncovered:
            if time.time() > deadline - 20.0:
                break
                
            best_idx = -1
            max_cover = 0
            candidates = []
            
            for i in range(total_words):
                gain = (balls[i] & uncovered).bit_count()
                if gain > max_cover:
                    max_cover = gain
                    best_idx = i
                    candidates = [i]
                elif gain == max_cover and gain > 0:
                    candidates.append(i)
                    
            if max_cover == 0:
                break
                
            if len(candidates) > 1:
                best_idx = random.choice(candidates)
            else:
                best_idx = candidates[0]
                
            code.append(best_idx)
            uncovered &= ~balls[best_idx]
            
        return code

    # Local Search with Coverage Counts
    def local_search(initial_code, max_time=120.0):
        if not initial_code:
            return initial_code
            
        current_code = initial_code[:]
        current_cover = get_cover_mask(current_code)
        
        # Track best found
        best_code = current_code[:]
        best_size = current_size = len(current_code)
        
        # Precompute set for fast membership check
        current_set = set(current_code)
        
        start_ls = time.time()
        
        while time.time() < deadline - 10.0:
            # Compute coverage counts
            coverage_counts = compute_coverage_counts(current_code)
            
            # Try to remove a codeword that is redundant
            # A codeword is redundant if removing it doesn't uncover any words
            # i.e., all words in its ball are covered by at least one other codeword
            # This is equivalent to: for all w in ball(c), coverage_counts[w] > 1
            
            removable = []
            for c in current_code:
                # Check if all neighbors of c are covered by others
                # i.e., min coverage in ball(c) > 1
                # We can check this by iterating over the ball
                m = balls[c]
                x = m
                is_redundant = True
                while x:
                    lsb = x & (-x)
                    idx = lsb.bit_length() - 1
                    if coverage_counts[idx] <= 1:
                        is_redundant = False
                        break
                    x ^= lsb
                if is_redundant:
                    removable.append(c)
            
            if removable:
                # Remove one
                c_remove = random.choice(removable)
                current_code.remove(c_remove)
                current_set.remove(c_remove)
                current_cover = get_cover_mask(current_code)
                current_size = len(current_code)
                
                if current_cover == all_mask and current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
                    continue
            
            # Try a swap
            # Pick a random codeword to remove
            c_remove = random.choice(current_code)
            
            # Find a codeword to add that improves coverage
            # We want to add a codeword that covers many currently "weakly covered" words
            # i.e., words with coverage count == 1
            weakly_covered = all_mask & ~0  # All words
            # Actually, we want to find a codeword that covers many words with count==1
            # and doesn't uncover any words (which is impossible if we just add)
            # But we are swapping, so we remove c_remove first, then add c_add
            
            # Simulate removal of c_remove
            cover_without_remove = current_cover & ~balls[c_remove]
            
            # Find best c_add
            best_c_add = -1
            best_gain = 0
            candidates = []
            
            for i in range(total_words):
                if i in current_set and i != c_remove:
                    continue
                # Gain from adding i to cover_without_remove
                gain = (balls[i] & ~cover_without_remove).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c_add = i
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if best_gain > 0:
                if len(candidates) > 1:
                    best_c_add = random.choice(candidates)
                
                # Apply swap
                current_code.remove(c_remove)
                current_set.remove(c_remove)
                current_code.append(best_c_add)
                current_set.add(best_c_add)
                current_cover = get_cover_mask(current_code)
                current_size = len(current_code)
                
                if current_cover == all_mask and current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
                    
        return best_code

    # Run Greedy to get a starting point
    # Then run Local Search
    best_candidate = None
    best_size = float('inf')
    
    # Run multiple greedy + local search iterations
    seeds = [42, 123, 456, 789, 1011, 1213]
    
    for seed in seeds:
        if time.time() > deadline - 30.0:
            break
            
        code = greedy_construct(seed)
        
        if code:
            # Check validity of greedy result
            cover = get_cover_mask(code)
            if cover == all_mask:
                # Run Local Search
                ls_code = local_search(code)
                if ls_code:
                    ls_cover = get_cover_mask(ls_code)
                    if ls_cover == all_mask and len(ls_code) < best_size:
                        best_size = len(ls_code)
                        best_candidate = ls_code
    
    if best_candidate is None:
        # Fallback to tetracode direct sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(digits_of[i]) for i in best_candidate]
