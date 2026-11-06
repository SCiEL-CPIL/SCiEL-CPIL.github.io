import numpy as np
import time
from itertools import combinations

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    N = 18
    MIN_DIST = 6
    
    # Start with Reed-Muller [16,5,8] code extended by 2 zero bits -> 32 words, dist 8
    base_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            base_words.append(word)
    
    code = list(base_words)
    code_set = set(code)
    
    # Precompute popcount lookup for 18-bit values
    popcount = np.zeros(2**18, dtype=np.uint8)
    for i in range(2**18):
        popcount[i] = i.bit_count()
    
    # Convert code to numpy array for faster vectorized distance checks
    code_np = np.array(code, dtype=np.uint32)
    
    # Helper: check if a batch of candidates are all at distance >= MIN_DIST from code
    def batch_check(candidates):
        """Return indices of candidates that have min distance >= MIN_DIST to all code words."""
        if len(candidates) == 0:
            return []
        cand_np = np.array(candidates, dtype=np.uint32)
        # Compute XOR between each candidate and each code word
        # This creates a (len(candidates), len(code)) array
        # For memory efficiency, do it in chunks
        valid_indices = []
        chunk_size = 1000
        for start in range(0, len(cand_np), chunk_size):
            end = min(start + chunk_size, len(cand_np))
            chunk = cand_np[start:end]
            # XOR each candidate with each code word
            xor_matrix = chunk[:, None] ^ code_np[None, :]  # (chunk, len(code))
            # Get popcounts
            dist_matrix = popcount[xor_matrix]  # (chunk, len(code))
            # Find min distance for each candidate
            min_dists = np.min(dist_matrix, axis=1)
            # Candidates with min_dist >= MIN_DIST are valid
            valid_in_chunk = np.where(min_dists >= MIN_DIST)[0]
            valid_indices.extend(start + i for i in valid_in_chunk)
        return valid_indices
    
    # Greedy addition with batch processing
    # Generate candidates in a structured order
    
    # First pass: try all words of weight 0 to 9
    candidates_by_weight = {}
    for w in range(0, 10):
        if time.time() > deadline:
            break
        cands = []
        for combo in combinations(range(N), w):
            word = 0
            for bit_pos in combo:
                word |= (1 << bit_pos)
            if word not in code_set:
                cands.append(word)
        candidates_by_weight[w] = cands
    
    for w in range(0, 10):
        if time.time() > deadline:
            break
        
        if w not in candidates_by_weight:
            continue
        
        cands = candidates_by_weight[w]
        if not cands:
            continue
        
        # Process in batches
        batch_size = 500
        for start in range(0, len(cands), batch_size):
            if time.time() > deadline:
                break
            
            batch = cands[start:start+batch_size]
            valid = batch_check(batch)
            
            for idx in valid:
                candidate = batch[idx]
                if candidate not in code_set:
                    code.append(candidate)
                    code_set.add(candidate)
                    code_np = np.append(code_np, np.array([candidate], dtype=np.uint32))
    
    # Second pass: random candidates with batch processing
    rng = np.random.default_rng(42)
    
    for _ in range(50):
        if time.time() > deadline:
            break
        
        # Generate a batch of random candidates
        batch_size = 1000
        batch = [int(x) for x in rng.integers(0, 2**18, size=batch_size)]
        
        # Filter out already-in-code candidates
        batch = [c for c in batch if c not in code_set]
        if not batch:
            continue
        
        valid = batch_check(batch)
        
        for idx in valid:
            candidate = batch[idx]
            if candidate not in code_set:
                code.append(candidate)
                code_set.add(candidate)
                code_np = np.append(code_np, np.array([candidate], dtype=np.uint32))
    
    # Third pass: more random candidates if time allows
    for _ in range(100):
        if time.time() > deadline:
            break
        
        batch_size = 2000
        batch = [int(x) for x in rng.integers(0, 2**18, size=batch_size)]
        batch = [c for c in batch if c not in code_set]
        if not batch:
            continue
        
        valid = batch_check(batch)
        
        for idx in valid:
            candidate = batch[idx]
            if candidate not in code_set:
                code.append(candidate)
                code_set.add(candidate)
                code_np = np.append(code_np, np.array([candidate], dtype=np.uint32))
    
    # Local search: try to improve by swapping codewords
    # This is optional and only done if time remains
    if time.time() < deadline - 10:
        # Try removing one codeword and seeing if we can add two new ones
        # This is complex, so we'll skip for now and rely on the greedy approach
        
        pass
    
    return code
