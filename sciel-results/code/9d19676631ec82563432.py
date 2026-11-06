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

    # --- Step 2: Greedy Improvement with better heuristics ---

    if time.time() > deadline - 10.0:
        return initial_code

    # Use a set for fast membership and a list for the code
    code_set = set(initial_code)
    code = list(initial_code)

    # Precompute popcount for all 17-bit words
    popcount = np.zeros(M, dtype=np.uint8)
    for i in range(1, M):
        popcount[i] = popcount[i >> 1] + (i & 1)

    # For each word, precompute its "forbidden set" (words within distance < 6)
    # We'll use a boolean array for blocked words
    blocked = np.zeros(M, dtype=bool)
    for w in initial_code:
        blocked[w] = True
        for off in offsets:
            blocked[w ^ off] = True

    # Candidate pool: all words not blocked
    candidates = []
    for x in range(M):
        if not blocked[x]:
            candidates.append(x)

    # Greedy: repeatedly add the word that blocks the fewest candidates (MRV-like)
    # But computing this exactly is expensive. Use a heuristic.

    # Strategy: try multiple random orderings and pick the best
    best_code = code[:]
    best_len = len(code)

    def greedy_lc(cur_blocked, cur_code, cur_candidates, deadline):
        """
        Greedy with least-constraining heuristic:
        At each step, choose the candidate that minimizes the number of
        other candidates it would block. Use sampling for efficiency.
        """
        cand_set = set(cur_candidates)
        
        while cand_set and time.time() < deadline - 2.0:
            # Find the candidate that blocks the fewest other candidates
            best_x = None
            best_score = float('inf')
            
            # Sample for efficiency
            if len(cand_set) > 80:
                sample = random.sample(list(cand_set), min(80, len(cand_set)))
            else:
                sample = list(cand_set)
            
            for x in sample:
                if cur_blocked[x]:
                    continue
                # Score: count how many other candidates in cand_set would be blocked
                score = 0
                for off in offsets:
                    y = x ^ off
                    if y in cand_set:
                        score += 1
                if score < best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                # Fallback: pick first unblocked
                found = False
                for x in cand_set:
                    if not cur_blocked[x]:
                        best_x = x
                        found = True
                        break
                if not found:
                    break
            
            # Add the chosen word
            cur_code.append(best_x)
            cur_blocked[best_x] = True
            cand_set.discard(best_x)
            
            # Remove newly blocked candidates
            for off in offsets:
                y = best_x ^ off
                cur_blocked[y] = True
                cand_set.discard(y)
        
        return cur_code

    def greedy_rand(cur_blocked, cur_code, cur_candidates, deadline):
        """Simple greedy with random ordering"""
        cand_list = list(cur_candidates)
        random.shuffle(cand_list)
        for x in cand_list:
            if time.time() > deadline - 2.0:
                break
            if cur_blocked[x]:
                continue
            cur_code.append(x)
            cur_blocked[x] = True
            for off in offsets:
                cur_blocked[x ^ off] = True
        return cur_code

    # Try multiple strategies
    for attempt in range(6):
        if time.time() > deadline - 3.0:
            break
        cur_blocked = blocked.copy()
        cur_code = list(initial_code)
        cur_code = greedy_lc(cur_blocked, cur_code, candidates, deadline)
        if len(cur_code) > best_len:
            best_code = cur_code
            best_len = len(cur_code)

    for attempt in range(4):
        if time.time() > deadline - 3.0:
            break
        cur_blocked = blocked.copy()
        cur_code = list(initial_code)
        cur_code = greedy_rand(cur_blocked, cur_code, candidates, deadline)
        if len(cur_code) > best_len:
            best_code = cur_code
            best_len = len(cur_code)

    return best_code
