<<<<<<< SEARCH
    # 1. Fast Greedy Initialization with potential tracking (inspired by solution 1)
    start_time = time.time()
    time_limit = 115.0
    
    # Use potential-based greedy for better initial code
    is_covered = np.zeros(N, dtype=bool)
    potential = np.full(N, 17, dtype=np.int32)  # ball size is 17
    C = []
    rng = np.random.default_rng(42)
    order = rng.permutation(N)
    
    while True:
        if time.time() > start_time + 30.0:  # Limit greedy to 30s
            break
        maxp = potential.max()
        if maxp <= 0:
            break
        # Candidates with max potential
        cands = np.where(potential == maxp)[0]
        if len(cands) == 1:
            c = int(cands[0])
        else:
            c = int(cands[rng.integers(len(cands))])
        
        C.append(c)
        potential[c] = 0
        for x in neighbors[c]:
            if not is_covered[x]:
                is_covered[x] = True
                # For every ball covering x, reduce potential
                potential[neighbors[x]] -= 1
                np.maximum(potential, 0, out=potential)
        
        if is_covered.all():
            break
=======
    # 1. Tensor-product initialization: split 8D into two 4D subspaces
    start_time = time.time()
    time_limit = 115.0
    
    # Decompose: first 4 coords (A) and last 4 coords (B)
    # Each 4D subspace has 3^4 = 81 words
    # Ball in 8D: a word w=(a,b) is covered by c=(c_a,c_b) if d(a,c_a)+d(b,c_b)<=1
    # i.e., (d(a,c_a)<=1 and d(b,c_b)==0) or (d(a,c_a)==0 and d(b,c_b)<=1)
    # or (d(a,c_a)==0 and d(b,c_b)==0)
    
    # Precompute 4D neighbors
    words_4d = list(itertools.product(range(3), repeat=4))
    idx_4d = {w: i for i, w in enumerate(words_4d)}
    N4 = 81
    neighbors_4d = []
    for i, w in enumerate(words_4d):
        ns = set([i])
        for pos in range(4):
            for val in range(3):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                ns.add(idx_4d[tuple(nw)])
        neighbors_4d.append(np.array(list(ns), dtype=int))
    
    # Map 8D index to (a_idx, b_idx)
    # 8D index = a_idx * 81 + b_idx where a_idx is first 4 coords, b_idx is last 4
    # Actually: index = w[0]*3^7 + w[1]*3^6 + ... + w[7]
    # a_idx = w[0]*3^3 + w[1]*3^2 + w[2]*3 + w[3]
    # b_idx = w[4]*3^3 + w[5]*3^2 + w[6]*3 + w[7]
    # 8D index = a_idx * 81 + b_idx
    
    # Generate structured initial code using slice decomposition
    # For each pair (a_code, b_code) in product of 4D codes, add (a_code, b_code)
    # But we need a smarter approach: use a small 4D covering code for each subspace
    
    # 4D ternary covering code with radius 1: need to cover 81 words, ball size in 4D = 1 + 4*2 = 9
    # Lower bound: ceil(81/9) = 9, so ~9-12 words per subspace
    # Use greedy to find small 4D code
    def greedy_4d():
        uncovered = np.ones(N4, dtype=bool)
        code = []
        for _ in range(20):
            best = -1
            best_cnt = -1
            for i in range(N4):
                cnt = np.sum(uncovered[neighbors_4d[i]])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best = i
            if best_cnt <= 0:
                break
            code.append(best)
            uncovered[neighbors_4d[best]] = False
            if not np.any(uncovered):
                break
        return code
    
    code_4d = greedy_4d()
    
    # Build 8D code as product: for each a in code_4d, for each b in code_4d, add (a,b)
    # This gives |code_4d|^2 codewords
    # But this may be too large. Let's use a smarter approach:
    # For each a in code_4d, cover all b with a single "universal" b, plus targeted b's
    
    # Actually, let's use the product but then reduce heavily
    C = []
    for a in code_4d:
        for b in code_4d:
            C.append(a * N4 + b)
    
    # Reduce the product code
    def get_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors[c]] += 1
        return counts
    
    def reduce_code(code, counts):
        if not code:
            return code, counts
        current_code = list(code)
        current_counts = counts.copy()
        while True:
            removed_any = False
            next_code = []
            for c in current_code:
                if np.min(current_counts[neighbors[c]]) >= 2:
                    current_counts[neighbors[c]] -= 1
                    removed_any = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed_any:
                break
        return current_code, current_counts
    
    C, counts = reduce_code(C, get_counts(C))
>>>>>>> REPLACE

