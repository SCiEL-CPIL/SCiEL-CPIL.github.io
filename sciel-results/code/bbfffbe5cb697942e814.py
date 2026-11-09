import time
import random
import numpy as np

TOTAL_BUDGET = 60.0
SAFETY_MARGIN = 5.0
DEADLINE = TOTAL_BUDGET - SAFETY_MARGIN

def _hamming(a, b):
    return (a ^ b).bit_count()

def _check_min_distance(words, min_dist):
    n = len(words)
    for i in range(n):
        wi = words[i]
        for j in range(i + 1, n):
            if _hamming(wi, words[j]) < min_dist:
                return False
    return True

def _reed_muller_extended():
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def _generate_low_weight_core(N=18):
    """Generate all vectors of weight 0, 1, and 2. These are mutually at distance >= 2,
    but we need distance >= 6. So we can't include all of them.
    Instead, let's just use weight 0 and weight 1 vectors? No, weight 1 vectors are at
    distance 2 from each other. 
    
    Actually, for min distance 6, we can include:
    - All weight 0: just {0}
    - We cannot include two weight-1 vectors (distance 2)
    - We cannot include a weight-1 and weight-2 if they share a bit (distance 1 or 3)
    
    Let me think differently. The sphere packing bound: each codeword excludes a ball of radius 2.
    Volume of radius-2 ball in 18-dim: C(18,0)+C(18,1)+C(18,2) = 1+18+153 = 172.
    Upper bound: 2^18 / 172 ≈ 2849.
    
    For the core, let's just start with {0} and greedily add.
    """
    return [0]

def _vectorized_distance_check(candidate, existing_words, min_dist):
    """Check if candidate has distance >= min_dist to all existing words using numpy."""
    if len(existing_words) == 0:
        return True
    existing = np.array(existing_words, dtype=np.uint32)
    xor = existing ^ candidate
    # Count bits in each element
    # Use a lookup or loop - for small arrays, Python is fine
    # For larger arrays, use numpy trick
    # Actually, let's use a fast method
    dists = np.array([x.bit_count() for x in xor])
    return bool(np.all(dists >= min_dist))

def _fast_greedy_pack(start_words, N=18, min_dist=6, time_limit=50.0, seed=42):
    """
    Greedy packing: repeatedly try to add random or structured candidates.
    Use a candidate pool that's pre-generated for speed.
    """
    rng = random.Random(seed)
    words = list(start_words)
    start_time = time.time()
    
    # Pre-generate a large pool of candidates to try
    # Use structured candidates: all weight-3, weight-4, weight-5 vectors
    # This is more systematic than pure random
    
    # Actually, let's use a hybrid: try random candidates but with smart rejection
    # Key insight: after adding a word, many nearby words become forbidden.
    # We can track a "forbidden" set to skip quickly.
    
    # For speed, batch-check candidates against the current code
    batch_size = 10000
    added_count = 0
    stagnant_rounds = 0
    
    while time.time() - start_time < time_limit:
        # Generate a batch of random candidates
        candidates = [rng.randint(0, (1 << N) - 1) for _ in range(batch_size)]
        
        # Also add some structured candidates (specific weights)
        for _ in range(100):
            w = rng.choice([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
            bits = rng.sample(range(N), w)
            word = 0
            for b in bits:
                word |= (1 << b)
            candidates.append(word)
        
        # Try to add each candidate
        newly_added = []
        for cand in candidates:
            if time.time() - start_time >= time_limit:
                break
            # Check against all existing words
            ok = True
            for w in words:
                if _hamming(cand, w) < min_dist:
                    ok = False
                    break
            if ok:
                words.append(cand)
                newly_added.append(cand)
        
        if len(newly_added) == 0:
            stagnant_rounds += 1
            if stagnant_rounds > 50:
                break
        else:
            stagnant_rounds = 0
    
    return words

def _greedy_with_local_improvement(start_words, N=18, min_dist=6, time_limit=50.0, seed=42):
    """
    Greedy packing with occasional local search: try to replace a word with another
    that allows more future additions.
    """
    rng = random.Random(seed)
    words = list(start_words)
    start_time = time.time()
    
    # First, do pure greedy
    words = _fast_greedy_pack(words, N, min_dist, time_limit=time_limit * 0.7, seed=seed)
    
    # Then try local search: remove a word, try to add 2+ new words
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time < 5:
        return words
    
    ls_start = time.time()
    improved = True
    while improved and time.time() - ls_start < remaining_time:
        improved = False
        # Try removing each word and see if we can add more
        for idx in range(len(words)):
            if time.time() - ls_start >= remaining_time:
                break
            
            saved = words[idx]
            words.pop(idx)
            
            # Try to add new words that were previously blocked by 'saved'
            new_words = []
            attempts = 0
            while attempts < 200:
                cand = rng.randint(0, (1 << N) - 1)
                ok = True
                for w in words:
                    if _hamming(cand, w) < min_dist:
                        ok = False
                        break
                if ok and _hamming(cand, saved) < min_dist:  # Must have been blocked by saved
                    words.append(cand)
                    new_words.append(cand)
                attempts += 1
            
            if len(new_words) > 1:  # Net gain
                improved = True
            else:
                # Put saved back
                words.insert(idx, saved)
                for nw in new_words:
                    words.remove(nw)
    
    return words

def _structured_greedy(N=18, min_dist=6, time_limit=50.0, seed=42):
    """
    Structured greedy: try candidates in order of increasing weight,
    with randomization within each weight class. This tends to find
    a more balanced code.
    """
    rng = random.Random(seed)
    words = [0]  # Start with zero word
    start_time = time.time()
    
    # Pre-generate candidates for each weight
    candidates_by_weight = {}
    for w in range(1, N + 1):
        # Generate a large random sample of weight-w vectors
        n_samples = min(100000, 2 ** (N // 2))  # Cap for memory
        cands = []
        for _ in range(n_samples):
            bits = rng.sample(range(N), w)
            word = 0
            for b in bits:
                word |= (1 << b)
            cands.append(word)
        # Shuffle
        rng.shuffle(cands)
        candidates_by_weight[w] = cands
    
    # Try to add candidates from each weight class
    # Use a round-robin approach
    pointers = {w: 0 for w in range(1, N + 1)}
    
    while time.time() - start_time < time_limit:
        progress = False
        for w in range(1, N + 1):
            if time.time() - start_time >= time_limit:
                break
            
            cands = candidates_by_weight[w]
            ptr = pointers[w]
            
            # Try up to 100 candidates from this weight class
            tried = 0
            while tried < 100 and ptr < len(cands):
                cand = cands[ptr]
                ptr += 1
                tried += 1
                
                # Check against all existing words
                ok = True
                for word in words:
                    if _hamming(cand, word) < min_dist:
                        ok = False
                        break
                
                if ok:
                    words.append(cand)
                    progress = True
            
            pointers[w] = ptr
            
            # If we exhausted this weight class, regenerate
            if ptr >= len(cands):
                # Generate more
                new_cands = []
                for _ in range(10000):
                    bits = rng.sample(range(N), w)
                    word = 0
                    for b in bits:
                        word |= (1 << b)
                    new_cands.append(word)
                rng.shuffle(new_cands)
                candidates_by_weight[w] = new_cands
                pointers[w] = 0
        
        if not progress:
            # No progress in a full round, try random high-weight
            for _ in range(1000):
                if time.time() - start_time >= time_limit:
                    break
                w = rng.randint(5, 13)
                bits = rng.sample(range(N), w)
                word = 0
                for b in bits:
                    word |= (1 << b)
                
                ok = True
                for existing in words:
                    if _hamming(word, existing) < min_dist:
                        ok = False
                        break
                if ok:
                    words.append(word)
                    progress = True
            
            if not progress:
                break
    
    return words

def _get_golay_generator_matrix():
    G = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]
    return G

def _generate_golay_code(G):
    codewords = []
    n_rows = len(G)
    n_cols = len(G[0])
    for mask in range(1 << n_rows):
        word = 0
        for i in range(n_rows):
            if (mask >> i) & 1:
                for j in range(n_cols):
                    if G[i][j]:
                        word ^= (1 << j)
        codewords.append(word)
    return codewords

def _shorten_code(codewords, cols_to_zero):
    remaining_cols = [c for c in range(24) if c not in cols_to_zero]
    shortened = []
    for w in codewords:
        if all(((w >> c) & 1) == 0 for c in cols_to_zero):
            new_word = 0
            for i, c in enumerate(remaining_cols):
                if (w >> c) & 1:
                    new_word |= (1 << i)
            shortened.append(new_word)
    return shortened

def _find_good_shortening(G, golay_words, target_size=512):
    import itertools
    n_cols = 24
    n_rows = 12
    best_set = None
    best_count = 0

    for cols in itertools.combinations(range(n_cols), 6):
        sub = [[G[r][c] for c in cols] for r in range(n_rows)]

        matrix = [row[:] for row in sub]
        rank = 0
        for c in range(6):
            pivot = -1
            for r in range(rank, 12):
                if matrix[r][c] == 1:
                    pivot = r
                    break
            if pivot == -1:
                continue
            matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
            for r in range(12):
                if r != rank and matrix[r][c] == 1:
                    for cc in range(c, 6):
                        matrix[r][cc] ^= matrix[rank][cc]
            rank += 1
            if rank == 6:
                break

        if rank <= 3:
            count = 0
            for w in golay_words:
                ok = True
                for c in cols:
                    if (w >> c) & 1:
                        ok = False
                        break
                if ok:
                    count += 1
            if count > best_count:
                best_count = count
                best_set = cols

    return best_set, best_count

def construct():
    N = 18
    MIN_DIST = 6
    start = time.time()

    # Strategy 1: Get a good base code from Golay shortening
    G = _get_golay_generator_matrix()
    golay_words = _generate_golay_code(G)
    
    shortening_set, count = _find_good_shortening(G, golay_words, target_size=512)
    
    base_code = None
    if shortening_set is not None and count >= 512:
        code = _shorten_code(golay_words, shortening_set)
        if len(code) >= 512 and _check_min_distance(code, MIN_DIST):
            base_code = code
    
    # Strategy 2: Structured greedy packing from scratch
    # This is the main new approach
    elapsed = time.time() - start
    time_remaining = DEADLINE - elapsed
    
    best_code = base_code if base_code else [0]
    
    # Run structured greedy with the base code as starting point
    if time_remaining > 10:
        greedy_result = _structured_greedy(
            N=N, 
            min_dist=MIN_DIST, 
            time_limit=min(time_remaining - 5, 45.0), 
            seed=42
        )
        if len(greedy_result) > len(best_code) and _check_min_distance(greedy_result, MIN_DIST):
            best_code = greedy_result
    
    # Strategy 3: Try pure random greedy as backup
    elapsed = time.time() - start
    time_remaining = DEADLINE - elapsed
    if time_remaining > 10:
        random_greedy = _fast_greedy_pack(
            [0], 
            N=N, 
            min_dist=MIN_DIST, 
            time_limit=min(time_remaining - 5, 20.0), 
            seed=123
        )
        if len(random_greedy) > len(best_code) and _check_min_distance(random_greedy, MIN_DIST):
            best_code = random_greedy
    
    # Strategy 4: Reed-Muller as fallback
    rm_words = _reed_muller_extended()
    if len(rm_words) > len(best_code) and _check_min_distance(rm_words, MIN_DIST):
        best_code = rm_words
    
    # Final verification
    if not _check_min_distance(best_code, MIN_DIST):
        # Fallback to RM
        if _check_min_distance(rm_words, MIN_DIST):
            best_code = rm_words
        else:
            best_code = [0]
    
    return best_code
