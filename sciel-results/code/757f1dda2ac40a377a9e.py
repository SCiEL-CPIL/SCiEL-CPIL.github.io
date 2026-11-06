import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 17
    min_dist = 6
    max_size = 340
    
    # Precompute popcounts using NumPy for fast vectorized distance checks
    popcount_np = np.array([bin(i).count('1') for i in range(1 << n)], dtype=np.uint8)
    
    # Construct the [17,8,6] quadratic residue code (shortened from [18,9,6])
    # Generator polynomial for the shortened code
    gen_poly = (1 << 9) | (1 << 7) | (1 << 6) | (1 << 3) | (1 << 0)
    
    def encode(msg):
        cw = 0
        for bit_pos in range(9):
            if (msg >> bit_pos) & 1:
                cw ^= (gen_poly << bit_pos)
        return cw
    
    qr_codewords = []
    seen = set()
    for msg in range(256):
        cw = encode(msg)
        if cw not in seen:
            qr_codewords.append(cw)
            seen.add(cw)
    
    # Verify QR code validity (distance >= 6)
    valid_qr = (len(qr_codewords) == 256)
    if valid_qr:
        qr_np = np.array(qr_codewords, dtype=np.uint32)
        # Check a few random pairs to save time, or all if fast enough
        # 256*256 = 65536 pairs, XOR is fast
        xor_matrix = qr_np[:, None] ^ qr_np[None, :]
        dists = popcount_np[xor_matrix]
        # Exclude self-distances (diagonal is 0)
        np.fill_diagonal(dists, 100) 
        if dists.min() < min_dist:
            valid_qr = False
    
    if valid_qr:
        best_codewords = qr_codewords
        best_size = 256
    else:
        # Fallback: Greedy from 0
        best_codewords = [0]
        best_size = 1
        # Quick greedy fill to get a decent baseline
        C_np = np.array(best_codewords, dtype=np.uint32)
        # Use natural order for initial fill
        for candidate in range(1, 1 << n):
            if time.time() > deadline - 20.0:
                break
            xor = C_np ^ candidate
            if np.min(popcount_np[xor]) >= min_dist:
                best_codewords.append(candidate)
                C_np = np.append(C_np, candidate)
                best_size += 1
                if best_size >= max_size:
                    return best_codewords

    # Vectorized Greedy Expansion
    # We maintain the current set of codewords as a NumPy array
    C_np = np.array(best_codewords, dtype=np.uint32)
    C_list = list(best_codewords)
    C_set = set(C_list)
    
    # We will run multiple rounds of greedy expansion.
    # In each round, we generate a batch of random candidates, check them against C,
    # and add the valid ones.
    # To improve quality, we prioritize candidates with higher "minimum distance" to C.
    
    # Pre-generate a large pool of random candidates to avoid regeneration overhead
    # 131072 is the total space. We can just iterate through a shuffled range.
    # But shuffling 131072 items takes time. 
    # Instead, let's use a strategy of random batches.
    
    # Strategy:
    # 1. Start with QR code (256 words).
    # 2. Generate random batches.
    # 3. For each batch, compute distances to all C.
    # 4. Select candidates with min_dist >= 6.
    # 5. Among valid candidates, pick the one with max(min_dist) to add? 
    #    Or add all valid ones? 
    #    Adding all valid ones is faster but might lead to a "dense" cluster that blocks others.
    #    Picking the "most distant" one (maximin) is better for spreading.
    #    However, if we have many valid ones, adding them all is a good move to grow size quickly.
    #    Let's try adding all valid ones in a batch, then re-evaluate.
    #    Actually, if we add one, it invalidates others in the same batch.
    #    So we must be careful.
    
    # Efficient approach:
    # Keep a list of "candidates" to try.
    # In each step, generate a batch of random words.
    # Filter out those already in C.
    # Compute min distance to C for the rest.
    # If any have min_dist >= 6, select the one with the highest min_dist.
    # Add it to C.
    # Repeat.
    
    # This is O(BatchSize * |C|) per addition.
    # With |C| ~ 256 and BatchSize ~ 1000, that's 256k ops.
    # NumPy can do this in ~1ms.
    # 55 seconds / 1ms = 55000 additions possible.
    # But we only need to go from 256 to ~340. So ~84 additions.
    # This is very fast. We can afford to be more thorough.
    
    # Let's use a "Best Improvement" local search on the set of candidates.
    # Actually, simple greedy with random restarts of the candidate pool might be enough.
    
    # Let's try to add words one by one, picking the best from a random sample.
    
    while time.time() < deadline - 2.0:
        # Generate a batch of random candidates
        batch_size = 5000
        batch = np.random.randint(0, 1 << n, size=batch_size, dtype=np.uint32)
        
        # Filter out existing codewords
        # Using a set lookup is fast for small C, but batch is large.
        # We can just check distance; if distance is 0, it's in C.
        
        # Compute distances from batch to C
        # Shape: (batch_size, len(C))
        xor_matrix = batch[:, None] ^ C_np[None, :]
        dists = popcount_np[xor_matrix]
        
        # Min distance for each candidate
        min_dists = dists.min(axis=1)
        
        # Find valid candidates (min_dist >= 6)
        valid_mask = min_dists >= min_dist
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            # No valid candidates in this batch. 
            # This can happen if we are close to maximal.
            # Try a smaller batch or just continue.
            continue
        
        # Select the best candidate: max of min_dists
        # This maximizes the "safety margin" of the new word.
        best_idx_in_valid = valid_indices[np.argmax(min_dists[valid_indices])]
        new_word = batch[best_idx_in_valid]
        
        # Add it
        C_np = np.append(C_np, new_word)
        C_list.append(new_word)
        C_set.add(new_word)
        
        if len(C_list) >= max_size:
            break
    
    return C_list
