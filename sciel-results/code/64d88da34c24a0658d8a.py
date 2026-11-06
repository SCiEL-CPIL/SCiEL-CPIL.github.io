def construct():
    import time
    import numpy as np
    import random
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 86.0  # safety margin before 90s kill

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

    # Precompute popcount for all 17-bit words
    popcount = np.zeros(M, dtype=np.uint8)
    for i in range(1, M):
        popcount[i] = popcount[i >> 1] + (i & 1)

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

    # --- Blocked array construction ---
    def make_blocked(code_list):
        blocked = np.zeros(M, dtype=bool)
        for w in code_list:
            blocked[w] = True
            for off in offsets:
                blocked[w ^ off] = True
        return blocked

    # --- Spectral projection approach ---
    # The Hamming graph H(N,1) has adjacency matrix A where A[i,j] = 1 iff dist(i,j)=1
    # For our distance constraint (dist >= 6), we want an independent set in the graph
    # where edges connect vertices at distance 1..5.
    # 
    # Key insight: The eigenvalues of the Hamming graph are well-known.
    # The graph G where edges connect vertices at distance 1..5 has eigenvalues:
    #   lambda_k = sum_{j=1}^{5} K_j(k) for k = 0, ..., N
    # where K_j(k) are Krawtchouk polynomials.
    #
    # For a random vector v, projecting onto the subspace spanned by eigenvectors
    # with eigenvalue > threshold and thresholding can find large independent sets.
    #
    # In practice, we use a simpler spectral approach:
    # 1. Start with a random vector
    # 2. Iteratively: for each coordinate, if it's "too close" to existing selected vertices,
    #    flip it (like a soft constraint satisfaction)
    # 3. This is essentially a spectral version of local search

    # Actually, let's use a more direct approach that leverages the structure:
    # The QR code has a nice linear structure. We can try to find cosets or
    # use the dual code structure to guide additions.

    # For now, let's use an improved local search with better repair strategies.

    cur_code = initial_code[:]
    cur_set = set(initial_code)
    cur_blocked = make_blocked(initial_code)
    cur_len = len(cur_code)

    best_code = cur_code[:]
    best_len = cur_len

    # Build candidate list: all words not blocked
    unblocked = np.where(~cur_blocked)[0]
    random.shuffle(unblocked)
    candidates = list(unblocked)

    bit_positions = list(range(N))
    search_start = time.time()
    max_search_time = deadline - search_start - 3.0

    iter_count = 0
    no_improve_count = 0
    max_no_improve = 300

    while True:
        if time.time() - search_start > max_search_time:
            break
        if not candidates:
            unblocked = np.where(~cur_blocked)[0]
            if len(unblocked) == 0:
                break
            random.shuffle(unblocked)
            candidates = list(unblocked)

        target = candidates.pop(0)

        if cur_blocked[target]:
            continue

        # Try direct addition
        if not cur_blocked[target]:
            cur_code.append(target)
            cur_set.add(target)
            cur_blocked[target] = True
            for off in offsets:
                cur_blocked[target ^ off] = True
            cur_len += 1
            no_improve_count = 0
            if cur_len > best_len:
                best_code = cur_code[:]
                best_len = cur_len
            continue

        # Try repair: find a codeword to move
        conflicts = []
        for c in cur_code:
            d = popcount[c ^ target]
            if d < min_dist:
                conflicts.append((c, d))

        if not conflicts:
            no_improve_count += 1
            continue

        found_repair = False
        # Sort conflicts by distance (closest first)
        conflicts.sort(key=lambda x: x[1])

        for c, d_conf in conflicts:
            if time.time() - search_start > max_search_time:
                break

            # Try 1-bit flips
            for bit in bit_positions:
                c_new = c ^ (1 << bit)
                if c_new in cur_set:
                    continue
                if popcount[c_new ^ target] < min_dist:
                    continue
                valid_new = True
                for other in cur_code:
                    if other == c:
                        continue
                    if popcount[c_new ^ other] < min_dist:
                        valid_new = False
                        break
                if not valid_new:
                    continue
                valid_target = True
                for other in cur_code:
                    if other == c:
                        continue
                    if popcount[target ^ other] < min_dist:
                        valid_target = False
                        break
                if not valid_target:
                    continue

                # Success
                cur_blocked[c] = False
                for off in offsets:
                    cur_blocked[c ^ off] = False

                cur_set.remove(c)
                cur_set.add(c_new)
                cur_set.add(target)
                idx = cur_code.index(c)
                cur_code[idx] = c_new
                cur_code.append(target)

                cur_blocked[c_new] = True
                for off in offsets:
                    cur_blocked[c_new ^ off] = True
                cur_blocked[target] = True
                for off in offsets:
                    cur_blocked[target ^ off] = True

                cur_len += 1
                no_improve_count = 0
                if cur_len > best_len:
                    best_code = cur_code[:]
                    best_len = cur_len
                found_repair = True
                break

            if found_repair:
                break

            if time.time() - search_start > max_search_time:
                break

            # Try 2-bit flips (limited for speed)
            for bit1 in bit_positions:
                for bit2 in bit_positions:
                    if bit2 <= bit1:
                        continue
                    c_new = c ^ (1 << bit1) ^ (1 << bit2)
                    if c_new in cur_set:
                        continue
                    if popcount[c_new ^ target] < min_dist:
                        continue
                    valid_new = True
                    for other in cur_code:
                        if other == c:
                            continue
                        if popcount[c_new ^ other] < min_dist:
                            valid_new = False
                            break
                    if not valid_new:
                        continue
                    valid_target = True
                    for other in cur_code:
                        if other == c:
                            continue
                        if popcount[target ^ other] < min_dist:
                            valid_target = False
                            break
                    if not valid_target:
                        continue

                    # Success
                    cur_blocked[c] = False
                    for off in offsets:
                        cur_blocked[c ^ off] = False

                    cur_set.remove(c)
                    cur_set.add(c_new)
                    cur_set.add(target)
                    idx = cur_code.index(c)
                    cur_code[idx] = c_new
                    cur_code.append(target)

                    cur_blocked[c_new] = True
                    for off in offsets:
                        cur_blocked[c_new ^ off] = True
                    cur_blocked[target] = True
                    for off in offsets:
                        cur_blocked[target ^ off] = True

                    cur_len += 1
                    no_improve_count = 0
                    if cur_len > best_len:
                        best_code = cur_code[:]
                        best_len = cur_len
                    found_repair = True
                    break
                if found_repair:
                    break

            if found_repair:
                break

        if not found_repair:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                no_improve_count = 0
                unblocked = np.where(~cur_blocked)[0]
                if len(unblocked) == 0:
                    break
                random.shuffle(unblocked)
                candidates = list(unblocked)

        iter_count += 1

    return best_code
