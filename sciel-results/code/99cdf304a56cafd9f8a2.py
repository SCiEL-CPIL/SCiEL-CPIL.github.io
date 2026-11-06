"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for 60s limit
    
    N = 17
    MIN_DIST = 6
    
    # 1. Generate the 256-word seed: Shortened [18,9,6] Quadratic Residue Code
    # The [18,9,6] QR code is cyclic. We can construct it using the primitive polynomial.
    # However, constructing the full QR code from scratch is complex.
    # Instead, we use a known structure or a robust random initialization if QR generation is too slow/complex.
    # But the task mentions "256-word linear reference". 
    # Let's try to construct the [17,8,6] code.
    # Actually, a simpler and faster robust seed is a random greedy construction, 
    # but we want to start with a strong base if possible.
    # Let's use a fast random greedy to get a base of ~150-200 words, then optimize.
    # Or better: Use the structure of the [18,9,6] code if we can hardcode or generate it quickly.
    
    # Let's stick to a very fast random greedy initialization to get a decent base,
    # then use the clique heuristic to expand.
    
    def get_bits(words):
        n_words = len(words)
        if n_words == 0:
            return np.zeros((0, N), dtype=np.uint8)
        bits = np.zeros((n_words, N), dtype=np.uint8)
        # Vectorized bit extraction is tricky in pure numpy without loops, 
        # but for 256 words it's fast enough.
        for i, w in enumerate(words):
            for j in range(N):
                if w & (1 << j):
                    bits[i, j] = 1
        return bits

    def int_to_bits(w):
        bits = np.zeros(N, dtype=np.uint8)
        for j in range(N):
            if w & (1 << j):
                bits[j] = 1
        return bits

    # Fast Random Greedy Initialization
    current_words = []
    current_bits = np.zeros((0, N), dtype=np.uint8)
    
    # Try to add random words until stuck or time limit
    # We aim for a base of ~200 words quickly.
    attempts = 0
    while time.time() < deadline - 10.0 and len(current_words) < 250:
        # Bias towards weight 8 or 9 for better distance properties
        weight = random.choice([8, 9])
        # Generate random word with specific weight
        positions = random.sample(range(N), weight)
        word = 0
        for p in positions:
            word |= (1 << p)
            
        if current_bits.size == 0:
            current_words.append(word)
            current_bits = np.array([int_to_bits(word)], dtype=np.uint8)
        else:
            cand_bits = int_to_bits(word)
            xor = current_bits ^ cand_bits
            dists = xor.sum(axis=1)
            if np.min(dists) >= MIN_DIST:
                current_words.append(word)
                current_bits = np.vstack([current_bits, cand_bits])
                attempts = 0
            else:
                attempts += 1
                if attempts > 1000:
                    break

    best_words = current_words[:]
    best_size = len(current_words)
    
    # 2. Generate Candidate Pool for Clique Search
    # Candidates: All vectors of weight 8 and 9.
    # C(17,8) = 24310, C(17,9) = 24310. Total ~48620 candidates.
    # This is small enough to hold in memory and compute distances.
    
    candidates = []
    for w in range(8, 10):
        # Generate all combinations
        from itertools import combinations
        for comb in combinations(range(N), w):
            word = 0
            for p in comb:
                word |= (1 << p)
            candidates.append(word)
            
    # Remove candidates that are too close to current code
    # We can vectorize this check against the current code
    if current_bits.size > 0:
        cand_bits_arr = np.array([int_to_bits(w) for w in candidates], dtype=np.uint8)
        # Compute distances from each candidate to each current word
        # This is (num_cands, N) XOR (num_code, N) -> (num_cands, num_code)
        # Memory: 48620 * 256 * 8 bytes ~ 100MB. Might be tight but okay.
        # Better: Loop over current words, filter candidates.
        
        valid_cands = []
        valid_bits = []
        
        # Optimization: Check against current code in batches or one by one
        # Since current code is ~250, and candidates ~50k, 
        # 50k * 250 distance checks = 12.5M ops. Fast enough.
        
        for i, cw in enumerate(current_words):
            cw_bits = int_to_bits(cw)
            # We will filter candidates that are >= MIN_DIST from ALL current words
            # It's easier to start with all candidates and remove invalid ones iteratively
            # But 50k * 250 is fast. Let's do it efficiently.
            
        # Actually, let's just filter the candidate list.
        filtered_cands = []
        filtered_bits = []
        
        # Pre-compute bits for candidates
        cand_bits_list = [int_to_bits(w) for w in candidates]
        
        # For each candidate, check against all current words
        # To speed this up, we can use numpy broadcasting in chunks
        chunk_size = 1000
        for start in range(0, len(candidates), chunk_size):
            end = min(start + chunk_size, len(candidates))
            chunk_cands = candidates[start:end]
            chunk_bits = np.array(cand_bits_list[start:end], dtype=np.uint8)
            
            # Distances to all current words
            # chunk_bits: (chunk_size, N)
            # current_bits: (num_code, N)
            # We need min distance to any word in current_code.
            
            # Loop over current words for this chunk (vectorized over chunk)
            min_dists = np.full(len(chunk_cands), 255, dtype=np.uint8)
            
            for j in range(len(current_words)):
                cw_bits = int_to_bits(current_words[j])
                xor = chunk_bits ^ cw_bits
                dists = xor.sum(axis=1)
                min_dists = np.minimum(min_dists, dists)
                
            # Keep candidates where min_dist >= MIN_DIST
            mask = min_dists >= MIN_DIST
            for k in range(len(chunk_cands)):
                if mask[k]:
                    filtered_cands.append(chunk_cands[k])
                    filtered_bits.append(chunk_bits[k])
                    
        candidates = filtered_cands
        cand_bits = np.array(filtered_bits, dtype=np.uint8) if filtered_bits else np.zeros((0, N), dtype=np.uint8)
    else:
        candidates = []
        cand_bits = np.zeros((0, N), dtype=np.uint8)
        
    # 3. Greedy Maximum Clique Expansion
    # We have a set of candidates that are compatible with the base code.
    # Now we want to pick a subset of these candidates that are also mutually compatible (dist >= 6).
    # This is a Max Clique problem on the graph of candidates.
    # Heuristic: Greedy by "confidence" or degree.
    
    if len(candidates) == 0:
        return best_words
        
    # Compute pairwise distances between candidates?
    # 50k candidates -> 50k^2 distances is too big.
    # We only need to add them one by one to the code.
    
    # Strategy: 
    # While time remains:
    #   1. Find a candidate that is valid against current code.
    #   2. Among valid candidates, pick the one with "highest potential" (e.g., lowest average distance to others? or just random?).
    #   Actually, since we already filtered against the base, we just need to ensure the new candidate is valid against the *newly added* ones.
    
    # Let's maintain a list of available candidates.
    # As we add a word, we must remove candidates that are too close to it.
    
    # To make this efficient, we can pre-compute a "compatibility" structure or just re-filter.
    # Re-filtering 50k candidates against 1 new word is fast (50k ops).
    # If we add 100 words, that's 50k * 100 = 5M ops. Very fast.
    
    available_indices = list(range(len(candidates)))
    
    while time.time() < deadline:
        if not available_indices:
            break
            
        # Pick a candidate
        # Heuristic: Random choice from available, or choice with max "slack"?
        # Random is simple and often works well.
        idx = random.choice(available_indices)
        cand_word = candidates[idx]
        cand_bit = cand_bits[idx]
        
        # Check against current code (which includes base + previously added)
        # We need to check against ALL current_words.
        # We already know it's valid against the base (initial current_words).
        # We only need to check against words added in this phase.
        # Let's keep track of words added in this phase.
        
        # For simplicity and robustness, check against all current_words.
        # Vectorized check:
        if current_bits.size > 0:
            xor = current_bits ^ cand_bit
            dists = xor.sum(axis=1)
            if np.min(dists) < MIN_DIST:
                # Invalid, remove from available
                available_indices.remove(idx)
                continue
        else:
            # Should not happen if base is non-empty, but safe
            pass
            
        # Valid! Add to code
        current_words.append(cand_word)
        current_bits = np.vstack([current_bits, cand_bit.reshape(1, -1)])
        
        if len(current_words) > best_size:
            best_size = len(current_words)
            best_words = current_words[:]
            
        # Update available candidates: remove those too close to cand_word
        # We can do this in vectorized chunks or just loop if list is small.
        # If available_indices is large, vectorize.
        
        if len(available_indices) > 100:
            # Get bits of available candidates
            # This requires mapping indices to bits
            # To avoid copying, we can just mark invalid ones in a boolean array
            # But we are using a list of indices.
            
            # Let's filter the list
            new_available = []
            for i in available_indices:
                if i == idx:
                    continue
                b = cand_bits[i]
                xor = b ^ cand_bit
                d = xor.sum()
                if d >= MIN_DIST:
                    new_available.append(i)
            available_indices = new_available
        else:
            new_available = []
            for i in available_indices:
                if i == idx:
                    continue
                b = cand_bits[i]
                xor = b ^ cand_bit
                d = xor.sum()
                if d >= MIN_DIST:
                    new_available.append(i)
            available_indices = new_available

    return best_words
# EVOLVE-BLOCK-END
