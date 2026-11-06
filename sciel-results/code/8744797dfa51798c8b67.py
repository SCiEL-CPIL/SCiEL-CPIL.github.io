def construct():
    import time
    import numpy as np
    import random
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 85.0

    N = 17
    M = 1 << N
    min_dist = 6
    max_radius = min_dist - 1  # 5

    # Precompute offsets (masks) of weight 1 to max_radius
    offsets_list = []
    for r in range(1, max_radius + 1):
        for comb in combinations(range(N), r):
            mask = 0
            for pos in comb:
                mask |= (1 << pos)
            offsets_list.append(mask)
    offsets = np.array(offsets_list, dtype=np.uint32)
    num_offsets = len(offsets)

    # --- Step 1: Generate the [17,8,6] QR Code ---

    def poly_divides(p, q):
        deg_p = p.bit_length() - 1
        deg_q = q.bit_length() - 1
        if deg_p > deg_q:
            return False
        if p == 0:
            return False
        r = q
        while r.bit_length() - 1 >= deg_p and r != 0:
            shift = (r.bit_length() - 1) - deg_p
            r ^= (p << shift)
        return r == 0

    target_poly = (1 << 17) | 1
    gen_poly = None

    for mid_bits in range(1 << 6):
        p = (1 << 8) | (mid_bits << 1) | 1
        if poly_divides(p, target_poly):
            gen_poly = p
            break

    if gen_poly is None:
        return [0]

    # Generate basis vectors for the [17,9,5] code
    basis = []
    for k in range(9):
        val = gen_poly << k
        val &= (1 << N) - 1
        basis.append(val)

    # Generate all 512 codewords of the [17,9,5] code
    qr_code_words = []
    for mask in range(1 << 9):
        word = 0
        for i in range(9):
            if mask & (1 << i):
                word ^= basis[i]
        qr_code_words.append(word)

    # Extend to [18,9,6] and Shorten to [17,8,6]
    # Keep words with even parity
    initial_code = []
    for w in qr_code_words:
        if bin(w).count('1') % 2 == 0:
            initial_code.append(w)

    if time.time() > deadline - 5.0:
        return initial_code

    # Precompute popcount for all 17-bit words
    popcount = np.zeros(M, dtype=np.uint8)
    for i in range(1, M):
        popcount[i] = popcount[i >> 1] + (i & 1)

    # Helper: check if a word is valid (distance >= 6 from all words in code)
    def is_valid_addition(word, code_set):
        for off in offsets:
            if (word ^ off) in code_set:
                return False
        return word not in code_set

    # Helper: try to repair by flipping bits in one codeword to make room for target
    def try_repair(target, code_list, code_set, blocked):
        """
        Try to find a codeword c in code_list such that if we flip 1-2 bits of c,
        the resulting c' still has dist >= 6 from all other codewords, and
        target has dist >= 6 from c' and all other codewords.
        Returns (new_code_list, new_code_set, new_blocked) if successful, else None.
        """
        if time.time() > deadline - 3.0:
            return None
        
        # Get the conflicting codewords (those within dist < 6 of target)
        conflicts = []
        for c in code_list:
            if c == target:
                continue
            d = popcount[c ^ target]
            if d < min_dist:
                conflicts.append((c, d))
        
        if not conflicts:
            return None
        
        # For each conflict, try to move it
        for c, d in conflicts:
            if time.time() > deadline - 3.0:
                return None
            # Try flipping 1 bit in c
            for bit in range(N):
                c_new = c ^ (1 << bit)
                if c_new in code_set:
                    continue
                # Check c_new is valid against all other codewords
                valid = True
                for off in offsets:
                    if (c_new ^ off) in code_set and (c_new ^ off) != c:
                        valid = False
                        break
                if not valid:
                    continue
                # Check target is valid against c_new
                if popcount[c_new ^ target] < min_dist:
                    continue
                # Check target is valid against all other codewords
                target_valid = True
                for other in code_list:
                    if other == c:
                        continue
                    if popcount[other ^ target] < min_dist:
                        target_valid = False
                        break
                if not target_valid:
                    continue
                # Success! Replace c with c_new and add target
                new_code_set = set(code_set)
                new_code_set.remove(c)
                new_code_set.add(c_new)
                new_code_set.add(target)
                
                new_blocked = blocked.copy()
                # Unblock c's neighborhood
                new_blocked[c] = False
                for off in offsets:
                    new_blocked[c ^ off] = False
                # Block c_new's neighborhood
                new_blocked[c_new] = True
                for off in offsets:
                    new_blocked[c_new ^ off] = True
                # Block target's neighborhood
                new_blocked[target] = True
                for off in offsets:
                    new_blocked[target ^ off] = True
                
                new_code_list = list(new_code_set)
                return (new_code_list, new_code_set, new_blocked)
            
            # Try flipping 2 bits in c
            for bit1 in range(N):
                for bit2 in range(bit1 + 1, N):
                    c_new = c ^ (1 << bit1) ^ (1 << bit2)
                    if c_new in code_set:
                        continue
                    # Check c_new is valid against all other codewords
                    valid = True
                    for off in offsets:
                        if (c_new ^ off) in code_set and (c_new ^ off) != c:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Check target is valid against c_new
                    if popcount[c_new ^ target] < min_dist:
                        continue
                    # Check target is valid against all other codewords
                    target_valid = True
                    for other in code_list:
                        if other == c:
                            continue
                        if popcount[other ^ target] < min_dist:
                            target_valid = False
                            break
                    if not target_valid:
                        continue
                    # Success! Replace c with c_new and add target
                    new_code_set = set(code_set)
                    new_code_set.remove(c)
                    new_code_set.add(c_new)
                    new_code_set.add(target)
                    
                    new_blocked = blocked.copy()
                    # Unblock c's neighborhood
                    new_blocked[c] = False
                    for off in offsets:
                        new_blocked[c ^ off] = False
                    # Block c_new's neighborhood
                    new_blocked[c_new] = True
                    for off in offsets:
                        new_blocked[c_new ^ off] = True
                    # Block target's neighborhood
                    new_blocked[target] = True
                    for off in offsets:
                        new_blocked[target ^ off] = True
                    
                    new_code_list = list(new_code_set)
                    return (new_code_list, new_code_set, new_blocked)
        
        return None

    # Main local search
    best_code = initial_code[:]
    best_len = len(initial_code)

    # Build initial blocked array
    code_set = set(initial_code)
    blocked = np.zeros(M, dtype=bool)
    for w in initial_code:
        blocked[w] = True
        for off in offsets:
            blocked[w ^ off] = True

    # Candidate pool
    candidates = [x for x in range(M) if not blocked[x]]
    random.shuffle(candidates)

    # Simulated annealing-style local search
    temp = 10.0
    cooling = 0.95
    min_temp = 0.1
    max_iters = 5000
    iter_count = 0

    cur_code = initial_code[:]
    cur_set = set(initial_code)
    cur_blocked = blocked.copy()
    cur_len = len(cur_code)

    while iter_count < max_iters:
        if time.time() > deadline - 2.0:
            break

        # Pick a random candidate
        if not candidates:
            break
        target = candidates[0]
        candidates = candidates[1:]
        
        if cur_blocked[target]:
            continue

        # Try to add target directly
        if is_valid_addition(target, cur_set):
            cur_code.append(target)
            cur_set.add(target)
            cur_blocked[target] = True
            for off in offsets:
                cur_blocked[target ^ off] = True
            cur_len += 1
            if cur_len > best_len:
                best_code = cur_code[:]
                best_len = cur_len
            continue

        # Try repair
        result = try_repair(target, cur_code, cur_set, cur_blocked)
        if result is not None:
            new_code, new_set, new_blocked = result
            new_len = len(new_code)
            
            # Accept if improves or with some probability (annealing)
            delta_len = new_len - cur_len
            if delta_len > 0:
                cur_code = new_code
                cur_set = new_set
                cur_blocked = new_blocked
                cur_len = new_len
                if cur_len > best_len:
                    best_code = cur_code[:]
                    best_len = cur_len
            else:
                # delta_len == 0 (replaced one, added one) - always accept for diversity
                # or accept with probability based on temperature
                if random.random() < 0.5:
                    cur_code = new_code
                    cur_set = new_set
                    cur_blocked = new_blocked
                    cur_len = new_len

        # Cool down
        temp *= cooling
        if temp < min_temp:
            temp = min_temp

        iter_count += 1

    return best_code
