"""Rewritten program with a stronger greedy algorithm for A_2(18,6)."""
import time
import numpy as np

def construct():
    """
    Construct a large binary code of length 18 with minimum distance 6.
    Uses a fast vectorized greedy algorithm with bit-parallel distance checks.
    """
    deadline = time.time() + 50.0  # Safety margin before 60s limit
    
    N = 18
    MIN_DIST = 6
    TOTAL = 1 << N  # 262144
    
    # Precompute bit matrices for all 2^18 words
    # Shape: (TOTAL, N), dtype uint8
    all_words = np.arange(TOTAL, dtype=np.int64)
    bit_mat_all = np.zeros((TOTAL, N), dtype=np.uint8)
    for j in range(N):
        bit_mat_all[:, j] = (all_words >> j) & 1
    
    def reed_muller_seed():
        """Generate the extended Reed-Muller code RM(1,4) extended to length 18? 
        Actually RM(1,4) has length 16, dimension 5, so 32 words, distance 8.
        We need length 18. Let's use a different seed.
        
        Actually, let's just use a simple seed: the all-zero word and its complement,
        plus a few others. Or better, use a known good code.
        
        For now, start with just {0} and let greedy build from there.
        """
        return [0]
    
    def greedy_pass(seed_words, rng, time_limit):
        """
        Greedy addition of words. We maintain a 'forbidden' boolean array of size TOTAL.
        A word is forbidden if it's within distance < MIN_DIST of any codeword.
        When we add a codeword c, we mark all words within distance < MIN_DIST of c as forbidden.
        """
        code = list(seed_words)
        forbidden = np.zeros(TOTAL, dtype=bool)
        
        # Mark seed words as forbidden
        for w in seed_words:
            forbidden[w] = True
        
        # Precompute: for each possible distance d, we can't easily enumerate neighbors.
        # Instead, for each new codeword, we need to find all words within distance < MIN_DIST.
        # That's expensive. Alternative: for each candidate, check distance to all codewords.
        
        # Better approach: maintain the code as a list, and for each candidate, 
        # compute min distance to all codewords using vectorized operations.
        
        # Actually, the forbidden array approach requires enumerating neighbors.
        # For N=18 and MIN_DIST=6, the number of words within distance 5 of a given word is:
        # sum_{k=0}^{5} C(18,k) = 1 + 18 + 153 + 816 + 3060 + 8568 = 12616
        # That's manageable!
        
        # Precompute neighbor offsets for each distance
        # Actually, let's precompute for each word, the list of words within distance < MIN_DIST.
        # But that's TOTAL * 12616 entries, which is ~3.3 billion - too much.
        
        # Alternative: for each candidate, compute distance to all codewords in the code.
        # Use vectorized numpy operations.
        
        code_arr = np.array(code, dtype=np.int64)
        
        # Candidate pool: all words not yet forbidden
        available = np.where(~forbidden)[0]
        rng.shuffle(available)
        
        for idx in range(len(available)):
            if time.time() > time_limit:
                break
            
            cand = int(available[idx])
            
            if len(code_arr) == 0:
                min_d = N + 1
            else:
                # Compute Hamming distance from cand to all codewords
                cand_bits = bit_mat_all[cand]  # shape (N,)
                # XOR with all codeword bits
                # code_arr shape: (len(code),)
                # We need bit_mat for code_arr
                code_bits = bit_mat_all[code_arr]  # shape (len(code), N)
                xor = code_bits ^ cand_bits  # shape (len(code), N)
                dists = xor.sum(axis=1)  # shape (len(code),)
                min_d = int(dists.min())
            
            if min_d >= MIN_DIST:
                code.append(cand)
                code_arr = np.append(code_arr, cand)
                forbidden[cand] = True
                # Mark all words within distance < MIN_DIST of cand as forbidden
                # This is the expensive part. Let's skip it for now and just rely on the distance check.
                # Actually, without marking neighbors, we'll keep checking already-rejected candidates.
                # Let's mark them to speed up future iterations.
        
        return code
    
    # Actually, the above approach is still slow because we re-check candidates.
    # Let me try a different strategy: use the forbidden array properly.
    
    def greedy_with_forbidden(seed_words, rng, time_limit):
        """
        Greedy with proper forbidden tracking.
        """
        code = list(seed_words)
        forbidden = np.zeros(TOTAL, dtype=bool)
        
        for w in seed_words:
            forbidden[w] = True
        
        # For efficiency, we'll iterate through all words in random order.
        # When we add a word, we mark its neighbors as forbidden.
        # To mark neighbors, we need to find all words within distance < MIN_DIST.
        
        # Precompute: for each word, the list of neighbors within distance < MIN_DIST.
        # This is too expensive to precompute for all 2^18 words.
        
        # Alternative: when adding a word c, compute neighbors on the fly.
        # For a word c, neighbors within distance d are words that differ in at most d bit positions.
        # We can generate them by trying all subsets of bit positions of size <= d.
        
        def mark_neighbors(c, forbidden):
            """Mark all words within distance < MIN_DIST of c as forbidden."""
            forbidden[c] = True
            # Generate all words at distance 1 to MIN_DIST-1 from c
            # Use bit manipulation
            c_bits = c
            # For each subset of positions to flip, of size 1 to MIN_DIST-1
            # This is expensive. Let's use a different approach.
            
            # Actually, for N=18 and MIN_DIST=6, the number of neighbors is ~12616.
            # We can generate them by iterating through all 2^18 words and checking distance.
            # But that's 262144 checks per added word, which is too slow.
            
            # Better: generate neighbors by flipping subsets of bits.
            # Number of ways to flip k bits out of 18: C(18,k)
            # For k=1: 18, k=2: 153, k=3: 816, k=4: 3060, k=5: 8568
            # Total: 12615 (excluding k=0)
            
            # Generate all combinations of bit positions to flip
            for k in range(1, MIN_DIST):
                # Generate all k-combinations of range(N)
                # This is expensive in Python. Let's use a precomputed list.
                pass
        
        # This is getting complicated. Let me use a simpler but effective approach:
        # Just use the vectorized distance check without forbidden marking.
        # To avoid re-checking, maintain a set of checked words.
        
        code = list(seed_words)
        code_set = set(code)
        checked = set()
        
        all_candidates = np.arange(TOTAL, dtype=np.int64)
        rng.shuffle(all_candidates)
        
        for idx in range(len(all_candidates)):
            if time.time() > time_limit:
                break
            
            cand = int(all_candidates[idx])
            
            if cand in code_set or cand in checked:
                continue
            
            checked.add(cand)
            
            if len(code) == 0:
                min_d = N + 1
            else:
                code_arr = np.array(code, dtype=np.int64)
                code_bits = bit_mat_all[code_arr]
                cand_bits = bit_mat_all[cand]
                xor = code_bits ^ cand_bits
                dists = xor.sum(axis=1)
                min_d = int(dists.min())
            
            if min_d >= MIN_DIST:
                code.append(cand)
                code_set.add(cand)
        
        return code
    
    # Let me try yet another approach: use a precomputed lookup table for distances.
    # For N=18, we can precompute popcount for all 2^18 values.
    popcount = np.zeros(TOTAL, dtype=np.uint8)
    for i in range(1, TOTAL):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    def greedy_fast(seed_words, rng, time_limit):
        """
        Fast greedy using popcount lookup.
        """
        code = list(seed_words)
        code_set = set(code)
        
        all_candidates = np.arange(TOTAL, dtype=np.int64)
        rng.shuffle(all_candidates)
        
        # Convert code to numpy array for vectorized operations
        code_arr = np.array(code, dtype=np.int64)
        
        for idx in range(len(all_candidates)):
            if time.time() > time_limit:
                break
            
            cand = int(all_candidates[idx])
            
            if cand in code_set:
                continue
            
            if len(code_arr) == 0:
                min_d = N + 1
            else:
                # XOR cand with each codeword, then popcount
                xors = code_arr ^ cand
                dists = popcount[xors]
                min_d = int(dists.min())
            
            if min_d >= MIN_DIST:
                code.append(cand)
                code_set.add(cand)
                code_arr = np.append(code_arr, cand)
        
        return code
    
    # Start with a good seed
    seed = reed_muller_seed()
    
    best_code = seed
    best_len = len(seed)
    
    rng = np.random.default_rng(42)
    
    # Try multiple random restarts
    while time.time() < deadline:
        code = greedy_fast(seed, rng, deadline - 2)
        if len(code) > best_len:
            best_code = code
            best_len = len(code)
        
        # If we've found a large code, try to improve with a different seed
        if best_len > 100:
            break
    
    return best_code
