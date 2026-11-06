import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds safety margin
    
    # Seed: First-order Reed-Muller code on 16 coordinates, extended by one zero bit?
    # The seed in the prompt uses 16 points and shifts by point (0..15).
    # The task says N=17.
    # The seed code in the prompt:
    # for point in range(16): bit ... word |= bit << point
    # This creates 16-bit words.
    # The prompt says "17-bit words".
    # The seed program returns words in [0, 2**16).
    # Wait, the prompt says "17-bit binary codeword".
    # The seed program:
    # words.append(word) where word has bits 0..15.
    # This is a 16-bit code.
    # But the task says N=17.
    # "Each integer encodes one N-bit binary codeword".
    # If the seed returns 16-bit words, are they considered 17-bit with a leading 0?
    # "including leading zeros".
    # So 0b010... is a 17-bit word.
    # The seed distance is 8.
    # We need distance >= 6.
    
    # Let's generate the seed words as per the prompt's code, but ensure they are treated as 17-bit.
    # The prompt's seed code:
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # The 17th bit (index 16) is 0.
            words.append(word)
            
    # Convert to NumPy array for fast operations
    # Shape: (K, 17), dtype uint8
    # We need to handle the 17th bit (index 16).
    
    def to_np(words_list):
        arr = np.zeros((len(words_list), 17), dtype=np.uint8)
        for i, w in enumerate(words_list):
            for j in range(17):
                arr[i, j] = (w >> j) & 1
        return arr

    code_np = to_np(words)
    K = len(words)
    
    # Generate candidates: All words of weight 8 and 9
    # This is a good pool for distance 6 codes.
    # Number of weight 8 words: C(17, 8) = 24310
    # Number of weight 9 words: C(17, 9) = 24310
    # Total ~48k.
    
    candidates = []
    # Generate weight 8
    for i in range(17):
        for j in range(i+1, 17):
            for k in range(j+1, 17):
                for l in range(k+1, 17):
                    for m in range(l+1, 17):
                        for n in range(m+1, 17):
                            for o in range(n+1, 17):
                                for p in range(o+1, 17):
                                    w = (1<<i) | (1<<j) | (1<<k) | (1<<l) | (1<<m) | (1<<n) | (1<<o) | (1<<p)
                                    candidates.append(w)
                                    
    # Generate weight 9
    # This nested loop is very slow in Python.
    # Better to use itertools.combinations.
    
    import itertools
    
    candidates = []
    for w in [8, 9]:
        for comb in itertools.combinations(range(17), w):
            word = 0
            for idx in comb:
                word |= (1 << idx)
            candidates.append(word)
            
    random.shuffle(candidates)
    
    batch_size = 500
    idx = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        if idx >= len(candidates):
            # Generate random candidates if we run out
            # Or break if we've tried a lot
            if no_improve_count > 5:
                break
            # Generate 1000 random candidates
            new_cands = [random.randint(0, 2**17 - 1) for _ in range(1000)]
            candidates.extend(new_cands)
            idx = len(candidates) - 1000
            
        end_idx = min(idx + batch_size, len(candidates))
        batch = candidates[idx:end_idx]
        idx = end_idx
        
        # Convert batch to NumPy
        batch_np = np.zeros((len(batch), 17), dtype=np.uint8)
        for i, w in enumerate(batch):
            for j in range(17):
                batch_np[i, j] = (w >> j) & 1
                
        # Compute distances
        # code_np: (K, 17)
        # batch_np: (B, 17)
        # We want dist(batch_np[i], code_np[j])
        
        # Broadcasting: (B, 1, 17) XOR (1, K, 17) -> (B, K, 17)
        # Sum over axis 2 -> (B, K)
        
        # Memory check: B=500, K=256. 500*256*17 bytes = ~2MB. OK.
        # If K grows to 1000, 500*1000*17 = 8.5MB. OK.
        
        diff = np.bitwise_xor(batch_np[:, np.newaxis, :], code_np[np.newaxis, :, :])
        dists = diff.sum(axis=2)
        min_dists = dists.min(axis=1)
        
        valid_mask = min_dists >= 6
        
        if np.any(valid_mask):
            valid_words = [batch[i] for i in range(len(batch)) if valid_mask[i]]
            # Add to code
            for w in valid_words:
                # Check if already in words (should be unique candidates, but safe)
                if w not in words:
                    words.append(w)
            # Update code_np
            # Instead of rebuilding, we can just rebuild if K changes significantly
            # Or append rows
            new_rows = np.zeros((len(valid_words), 17), dtype=np.uint8)
            for i, w in enumerate(valid_words):
                for j in range(17):
                    new_rows[i, j] = (w >> j) & 1
            code_np = np.vstack((code_np, new_rows))
            K += len(valid_words)
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if K > 300:
            # We are beating the linear bound, maybe stop early to save time?
            # Or continue to try for 258+
            pass

    return words

