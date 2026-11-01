import itertools
import time
import random

def construct(q, n, radius):
    # This solution is specialized for q=3, n=6, radius=1
    if q != 3 or n != 6 or radius != 1:
        # Fallback to tetracode direct sum for other cases (though task is fixed)
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(itertools.product(range(3), repeat=n))[0]] # Dummy

    start_time = time.time()
    deadline = start_time + 118.0 # 2s safety margin
    
    # Precompute neighbors for all 729 words
    neighbors = []
    for c in range(729):
        digs = [(c // (3**i)) % 3 for i in range(6)]
        ns = [c]
        for i in range(6):
            for delta in [-1, 1]:
                new_dig = (digs[i] + delta) % 3
                weight = 3**i
                old_val = digs[i] * weight
                new_val = new_dig * weight
                ns.append(c - old_val + new_val)
        neighbors.append(ns)
    
    # Precompute neighbor bitmasks (using Python ints as arbitrary-precision bitsets)
    # bit w set in neighbor_mask[c] means word w is within distance 1 of codeword c
    neighbor_mask = [0] * 729
    for c in range(729):
        mask = 0
        for w in neighbors[c]:
            mask |= (1 << w)
        neighbor_mask[c] = mask
    
    # Initial set: Tetracode direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codes = []
    for h in tetracode:
        for t in itertools.product(range(3), repeat=2):
            code_digits = h + list(t)
            val = 0
            for i, d in enumerate(code_digits):
                val += d * (3**i)
            initial_codes.append(val)
    
    current_codes = list(set(initial_codes))
    code_set = set(current_codes)
    
    # Coverage count per word
    cov = [0] * 729
    # For each word, track which codewords cover it (as a set for intersection)
    coverers = [set() for _ in range(729)]
    
    for c in current_codes:
        for w in neighbors[c]:
            cov[w] += 1
            coverers[w].add(c)
            
    best_size = len(current_codes)
    best_codes = current_codes[:]
    
    def remove_code(c):
        for w in neighbors[c]:
            cov[w] -= 1
            coverers[w].remove(c)
        code_set.remove(c)
        current_codes.remove(c)
        
    def add_code(c):
        for w in neighbors[c]:
            cov[w] += 1
            coverers[w].add(c)
        code_set.add(c)
        current_codes.append(c)
        
    def is_removable(c):
        for w in neighbors[c]:
            if cov[w] == 1:
                return False
        return True
        
    def try_removals():
        removed = False
        while True:
            any_removed = False
            for c in current_codes[:]:
                if is_removable(c):
                    remove_code(c)
                    any_removed = True
                    removed = True
            if not any_removed:
                break
        return removed

    def try_swaps():
        """Try to find a beneficial swap: remove one codeword, add another that covers its exclusive words."""
        swapped = False
        # Find codewords with exclusive words (words only they cover)
        candidates_for_swap = []
        for c in current_codes:
            exclusive = []
            for w in neighbors[c]:
                if cov[w] == 1:
                    exclusive.append(w)
            if exclusive:
                candidates_for_swap.append((len(exclusive), c, exclusive))
                
        if not candidates_for_swap:
            return False
            
        # Sort by size of exclusive set (smallest first - easier to cover)
        candidates_for_swap.sort(key=lambda x: x[0])
        
        for size, c, exclusive in candidates_for_swap:
            # Find codewords that cover ALL exclusive words
            # Start with coverers of first exclusive word, intersect with rest
            common = set(coverers[exclusive[0]])
            for w in exclusive[1:]:
                common &= coverers[w]
                if not common:
                    break
            
            if not common:
                continue
            
            # Remove codewords already in our set
            common -= code_set
            if not common:
                continue
            
            # Try each candidate new codeword; pick one that maximizes additional coverage
            # (i.e., covers the most currently-uncovered or low-coverage words)
            best_new = None
            best_gain = -1
            for new_c in list(common)[:40]:  # Limit to 40 candidates for speed
                # Estimate gain: number of words where cov would increase
                gain = 0
                for w in neighbors[new_c]:
                    if cov[w] < 2:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_new = new_c
            
            if best_new is not None:
                remove_code(c)
                add_code(best_new)
                swapped = True
                break
        
        return swapped

    def try_group_perturbation():
        """Move a codeword to c+v for a small Z3 vector v (group-structured perturbation)."""
        # Precompute all vectors with Hamming weight 1 or 2 (mod 3, non-zero)
        # weight 1: 6*2 = 12 vectors
        # weight 2: C(6,2)*4 = 60 vectors
        if not hasattr(try_group_perturbation, '_vectors'):
            vecs = []
            for i in range(6):
                for d in (1, 2):
                    v = [0]*6
                    v[i] = d
                    vecs.append(tuple(v))
            for i in range(6):
                for j in range(i+1, 6):
                    for di in (1, 2):
                        for dj in (1, 2):
                            v = [0]*6
                            v[i] = di
                            v[j] = dj
                            vecs.append(tuple(v))
            try_group_perturbation._vectors = vecs
        vecs = try_group_perturbation._vectors
        
        # Pick a random codeword to move
        if not current_codes:
            return False
        c = random.choice(current_codes)
        # Try all vectors, pick the best move (maximizes coverage gain without breaking coverage)
        # Compute current "deficit" profile for words in neighbors[c]
        best_v = None
        best_score = 0
        c_digits = [(c // (3**i)) % 3 for i in range(6)]
        for v in vecs:
            new_digits = [(c_digits[i] + v[i]) % 3 for i in range(6)]
            new_c = 0
            for i, d in enumerate(new_digits):
                new_c += d * (3**i)
            if new_c == c:
                continue
            if new_c in code_set:
                continue
            # Score: how many words in neighbors[new_c] would gain coverage (cov < 2)
            # minus how many words in neighbors[c] would lose their only cover
            score = 0
            for w in neighbors[new_c]:
                if cov[w] < 2:
                    score += 1
            for w in neighbors[c]:
                if cov[w] == 1:
                    score -= 3  # penalty for breaking coverage
            if score > best_score:
                best_score = score
                best_v = (c, new_c)
        
        if best_v is None or best_score <= 0:
            return False
        
        old_c, new_c = best_v
        remove_code(old_c)
        add_code(new_c)
        return True

    def try_multi_swaps():
        """Try removing up to 2 codewords and adding 1-2 new ones."""
        # Try a 2-for-1 swap: remove two codewords, add one that covers all their exclusive words
        # First find pairs of codewords whose combined exclusive set is small
        codes_list = list(current_codes)
        best_pair = None
        best_pair_size = 10  # threshold
        for i in range(len(codes_list)):
            for j in range(i+1, len(codes_list)):
                c1, c2 = codes_list[i], codes_list[j]
                # Combined exclusive words: words covered by c1 or c2 with cov==1
                exclusive = set()
                for w in neighbors[c1]:
                    if cov[w] == 1:
                        exclusive.add(w)
                for w in neighbors[c2]:
                    if cov[w] == 1:
                        exclusive.add(w)
                if len(exclusive) < best_pair_size:
                    best_pair_size = len(exclusive)
                    best_pair = (c1, c2, exclusive)
                if best_pair_size <= 2:
                    break
            if best_pair_size <= 2:
                break
        
        if best_pair is None:
            return False
        
        c1, c2, exclusive = best_pair
        if not exclusive:
            # Both are removable; just remove both
            remove_code(c1)
            remove_code(c2)
            return True
        
        # Find a codeword that covers all exclusive words
        common = set(coverers[list(exclusive)[0]])
        for w in list(exclusive)[1:]:
            common &= coverers[w]
            if not common:
                break
        if not common:
            return False
        common -= code_set
        if not common:
            return False
        
        # Pick the best new codeword
        best_new = None
        best_gain = -1
        for new_c in list(common)[:30]:
            gain = 0
            for w in neighbors[new_c]:
                if cov[w] < 2:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_new = new_c
        
        if best_new is not None:
            remove_code(c1)
            remove_code(c2)
            add_code(best_new)
            return True
        
        return False

    # Main Loop
    no_improve_count = 0
    max_no_improve = 30
    
    while time.time() < deadline:
        # 1. Try removing
        if try_removals():
            if len(current_codes) < best_size:
                best_size = len(current_codes)
                best_codes = current_codes[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
            continue
            
        # 2. Try swapping
        if try_swaps():
            # After a swap, coverage changed; loop back to try removals
            no_improve_count = 0
            continue
        
        # 3. Try multi-swap (2-for-1)
        if try_multi_swaps():
            no_improve_count = 0
            # After multi-swap, try removals
            try_removals()
            if len(current_codes) < best_size:
                best_size = len(current_codes)
                best_codes = current_codes[:]
                no_improve_count = 0
            continue
        
        # 4. Try group perturbation (move a codeword along a Z3 vector)
        if no_improve_count > 5 and try_group_perturbation():
            no_improve_count = 0
            try_removals()
            if len(current_codes) < best_size:
                best_size = len(current_codes)
                best_codes = current_codes[:]
                no_improve_count = 0
            continue
        
        # 5. Stuck. Perturb.
        # If we've been stuck too long, do a stronger perturbation
        if no_improve_count > max_no_improve:
            # Revert to best and try a different random seed
            current_codes[:] = best_codes[:]
            code_set = set(current_codes)
            cov = [0] * 729
            coverers = [set() for _ in range(729)]
            for c in current_codes:
                for w in neighbors[c]:
                    cov[w] += 1
                    coverers[w].add(c)
            no_improve_count = 0
            
            # Add 2-3 random codewords for stronger perturbation
            for _ in range(random.randint(2, 3)):
                for attempt in range(20):
                    r = random.randint(0, 728)
                    if r not in code_set:
                        add_code(r)
                        break
        
        else:
            # Try group perturbation first, then random
            if no_improve_count > 3:
                if try_group_perturbation():
                    no_improve_count = 0
                    try_removals()
                    if len(current_codes) < best_size:
                        best_size = len(current_codes)
                        best_codes = current_codes[:]
                        no_improve_count = 0
                    continue
            # Single random perturbation
            for attempt in range(20):
                r = random.randint(0, 728)
                if r not in code_set:
                    add_code(r)
                    break
        
        no_improve_count += 1
        
        # After perturbation, run removals to get back down
        try_removals()
        
        if len(current_codes) < best_size:
            best_size = len(current_codes)
            best_codes = current_codes[:]
            no_improve_count = 0
        
    # Convert best_codes back to list of lists
    result = []
    for c in best_codes:
        digs = [(c // (3**i)) % 3 for i in range(6)]
        result.append(digs)
        
    return result
