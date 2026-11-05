import numpy as np
import time

def construct():
    N = 17
    MIN_D = 6
    deadline = time.time() + 55  # safety margin before 60s limit
    
    # Step 1: Generate the 256-word linear [17,8,6] code
    # f1(x) = x^8 + x^4 + x^3 + x^2 + 1
    f1 = [1, 0, 0, 1, 1, 1, 0, 0, 1]
    
    basis = []
    for i in range(9):
        vec = [0] * 17
        for j in range(len(f1)):
            if i + j < 17:
                vec[i+j] = f1[j]
        basis.append(vec)
        
    words_17 = []
    for mask in range(512):
        word = [0] * 17
        for i in range(9):
            if mask & (1 << i):
                for j in range(17):
                    word[j] ^= basis[i][j]
        words_17.append(word)
        
    # Extend to 18 bits
    words_18 = []
    for w in words_17:
        parity = sum(w) % 2
        words_18.append(w + [parity])
        
    # Shorten: keep words where bit 0 is 0
    codewords = []
    for w18 in words_18:
        if w18[0] == 0:
            new_word = w18[1:]
            val = 0
            for i, bit in enumerate(new_word):
                if bit:
                    val |= (1 << i)
            codewords.append(val)
    
    # Step 2: Greedily add any 17-bit word at distance >= 6 from all existing
    # Use vectorized distance computation for speed
    
    def can_add(candidate, codes):
        """Check if candidate can be added to codes (all pairwise dist >= MIN_D)"""
        if not codes:
            return True
        arr = np.array(codes, dtype=np.uint8)
        # Convert candidate to bit vector
        cand_bits = np.array([(candidate >> i) & 1 for i in range(N)], dtype=np.uint8)
        # Compute Hamming distances
        dists = np.sum(arr ^ cand_bits[None, :], axis=1)
        return np.min(dists) >= MIN_D
    
    # Precompute all possible 17-bit words as a numpy array for fast distance checks
    all_words = np.arange(2**N, dtype=np.int64)
    bits_all = np.array([[(w >> i) & 1 for i in range(N)] for w in all_words], dtype=np.uint8)
    
    # Current code as bit matrix
    def to_bit_matrix(codes):
        return np.array([[(c >> i) & 1 for i in range(N)] for c in codes], dtype=np.uint8)
    
    code_bits = to_bit_matrix(codewords)
    
    # Iteratively try to add words
    added_any = True
    while added_any and time.time() < deadline:
        added_any = False
        
        # Compute minimum distance from each candidate to the current code
        # dist[c, j] = Hamming distance between bits_all[c] and code_bits[j]
        # Use: sum over bits of (a != b)
        # Vectorized: for each code word, compute distance to all candidates
        # This is expensive: 131072 x 256 x 17 operations
        
        # Better approach: compute for each candidate the min distance to code
        # dist_to_code[c] = min over j of sum_i (bits_all[c,i] != code_bits[j,i])
        
        # Process in batches to manage memory
        batch_size = 10000
        found_new = False
        
        for start in range(0, 2**N, batch_size):
            if time.time() >= deadline:
                break
            end = min(start + batch_size, 2**N)
            batch = bits_all[start:end]
            
            # For each candidate in batch, compute min distance to code
            # dist[b, j] = sum_i (batch[b,i] != code_bits[j,i])
            # = sum_i (batch[b,i] + code_bits[j,i] - 2*batch[b,i]*code_bits[j,i])
            # = weight(batch[b]) + weight(code_bits[j]) - 2*dot(batch[b], code_bits[j])
            
            batch_weights = np.sum(batch, axis=1)  # (batch_size,)
            code_weights = np.sum(code_bits, axis=1)  # (num_codes,)
            
            # dot products: (batch_size, num_codes)
            dots = batch @ code_bits.T
            
            # distances: (batch_size, num_codes)
            dists = batch_weights[:, None] + code_weights[None, :] - 2 * dots
            
            # min distance for each candidate
            min_dists = np.min(dists, axis=1)
            
            # Find candidates with min_dist >= MIN_D
            valid_mask = min_dists >= MIN_D
            valid_indices = np.where(valid_mask)[0]
            
            for idx in valid_indices:
                if time.time() >= deadline:
                    break
                candidate = start + idx
                # Add this word
                codewords.append(candidate)
                code_bits = np.vstack([code_bits, bits_all[candidate][None, :]])
                added_any = True
                found_new = True
            
            if not found_new and start > 0:
                # No new words found in earlier batches, continue checking
                pass
    
    return codewords
