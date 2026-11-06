import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 58.0
    
    n = 17
    min_dist = 6
    max_size = 340
    size = 1 << n  # 131072

    # Precompute popcounts using vectorized operations
    # This is fast: creating array 0..2^17-1, then using bit tricks
    arr = np.arange(size, dtype=np.uint32)
    popcount = np.zeros(size, dtype=np.uint8)
    x = arr.copy()
    for _ in range(n):
        popcount += (x & 1).astype(np.uint8)
        x >>= 1
    
    # Alternatively, use a lookup table for speed
    # But vectorized above is fine. Let's use a more standard efficient method.
    # popcount = np.array([bin(i).count('1') for i in range(size)], dtype=np.uint8) # Too slow
    
    # A faster way for popcount in numpy:
    # popcount = (arr.astype(np.uint64) // 2 % 2) + (arr // 4 % 2) + ... 
    # Or use lookup. Since size is 131072, a lookup table of 256 is best.
    lookup = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    popcount = (lookup[arr & 0xFF] + 
                lookup[(arr >> 8) & 0xFF] + 
                lookup[(arr >> 16) & 0xFF]).astype(np.uint8)

    # Construct the [17,8,6] QR code
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
    
    # Verify QR code
    valid_qr = (len(qr_codewords) == 256)
    if valid_qr:
        qr_arr = np.array(qr_codewords, dtype=np.uint32)
        # Check pairwise distances
        # This is 256x256 matrix, fine to do at once
        xor_matrix = qr_arr[:, None] ^ qr_arr[None, :]
        dist_matrix = popcount[xor_matrix]
        np.fill_diagonal(dist_matrix, 999)
        if np.min(dist_matrix) < min_dist:
            valid_qr = False

    if valid_qr:
        in_code = np.zeros(size, dtype=bool)
        for cw in qr_codewords:
            in_code[cw] = True
        current_code = set(qr_codewords)
        current_size = 256
    else:
        in_code = np.zeros(size, dtype=bool)
        in_code[0] = True
        current_code = {0}
        current_size = 1

    best_size = current_size
    best_codewords = list(current_code)

    # Helper: find valid candidates among a list of indices
    def find_valid_candidates(candidate_indices, code_arr):
        if len(candidate_indices) == 0:
            return []
        cand_arr = np.array(candidate_indices, dtype=np.uint32)
        # Compute distances to all codewords in code_arr
        # To avoid huge memory, do in chunks if necessary, but 256 * 131072 is 32M, which is ~32MB. OK.
        # Actually, we want to find which candidates have min distance >= 6 to ALL codewords.
        
        valid_mask = np.ones(len(candidate_indices), dtype=bool)
        
        # Iterate through codewords in chunks to save memory if needed, but 256 is small.
        for i in range(0, len(code_arr), 1000000): # Chunking not strictly needed for 256 codewords, but good practice
             pass
             
        # Efficient vectorized check:
        # dist[c, k] = popcount[cand_arr[c] ^ code_arr[k]]
        # We need min over k to be >= 6.
        
        # Memory: len(candidate_indices) * len(code_arr)
        # If candidate_indices is large (e.g., 100k) and code_arr is 256, that's 25.6M uint8s = 25MB. Fine.
        
        # Let's do it in chunks of code_arr to be safe with memory limits if code grows.
        for k_start in range(0, len(code_arr), 1000):
            code_chunk = code_arr[k_start:k_start+1000]
            # Shape: (len(cand), len(chunk))
            xor_vals = cand_arr[:, None] ^ code_chunk[None, :]
            dists = popcount[xor_vals]
            min_dists = np.min(dists, axis=1)
            valid_mask &= (min_dists >= min_dist)
            
        return candidate_indices[valid_mask]

    # Main Loop: Greedy Expansion + Swaps
    # We maintain in_code and current_code (set)
    
    # Pre-compute list of all indices not in code
    all_indices = np.arange(size, dtype=np.int32)
    
    while time.time() < deadline - 5.0:
        # 1. Try to Add
        code_arr = np.array(list(current_code), dtype=np.uint32)
        
        # Candidates are those not in in_code
        cand_mask = ~in_code
        cand_indices = all_indices[cand_mask]
        
        if len(cand_indices) > 0:
            valid_cands = find_valid_candidates(cand_indices, code_arr)
            
            if len(valid_cands) > 0:
                # Pick one at random (or the one with highest "something")
                # Let's just pick a random one to explore different parts
                idx = random.randint(0, len(valid_cands) - 1)
                new_word = int(valid_cands[idx])
                
                current_code.add(new_word)
                in_code[new_word] = True
                current_size += 1
                
                if current_size > best_size:
                    best_size = current_size
                    best_codewords = list(current_code)
                    
                continue # Restart loop to try adding more
        
        # 2. Try to Swap (Remove 1, Add 2)
        # This is trickier.
        # We need to find a word w in code such that removing it allows adding 2 new words.
        
        # Heuristic:
        # For each word w in code (sample if too many), remove it temporarily.
        # Find valid candidates in the rest.
        # If we can add 2, keep the swap.
        
        if current_size > 0:
            # Sample a few words to try removing
            code_list = list(current_code)
            random.shuffle(code_list)
            
            # Limit search time for swaps
            swap_deadline = time.time() + 2.0
            found_swap = False
            
            for w in code_list:
                if time.time() > swap_deadline:
                    break
                
                # Remove w
                current_code.remove(w)
                in_code[w] = False
                current_size -= 1
                
                code_arr = np.array(list(current_code), dtype=np.uint32)
                cand_mask = ~in_code
                cand_indices = all_indices[cand_mask]
                
                valid_cands = find_valid_candidates(cand_indices, code_arr)
                
                if len(valid_cands) >= 2:
                    # We can add 2
                    # Pick 2 random
                    idxs = random.sample(range(len(valid_cands)), 2)
                    w1 = int(valid_cands[idxs[0]])
                    w2 = int(valid_cands[idxs[1]])
                    
                    current_code.add(w1)
                    in_code[w1] = True
                    current_code.add(w2)
                    in_code[w2] = True
                    current_size += 2 # Net +1
                    
                    if current_size > best_size:
                        best_size = current_size
                        best_codewords = list(current_code)
                        
                    found_swap = True
                    break
                else:
                    # Revert
                    current_code.add(w)
                    in_code[w] = True
                    current_size += 1
            
            if found_swap:
                continue # Restart main loop
            
            # If no swap worked, we are stuck at local maximum for simple swaps
            # Try 2-for-1 swaps (Remove 2, Add 1)? No, that decreases size.
            # Try 2-for-3? (Remove 2, Add 3)
            # This is computationally expensive.
            
            # If stuck, break to save time? Or try random restart?
            # Given time limit, breaking is safer than risking timeout.
            break

    return best_codewords
